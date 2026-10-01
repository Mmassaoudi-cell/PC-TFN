"""Untouched-test evaluation of the frozen final model against the full benchmark suite.

Must only be run after FINAL_MODEL_CONFIG.yaml is frozen (Step 13 of the study protocol).
Trains every method on the frozen `train` split, fits any post-hoc calibration on the
frozen `validation` split only, and evaluates once on the frozen `test` split and once
on the untouched `external_test` (London, zero-shot) split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "SOURCE_METHOD_REPRODUCTION"))

from resilient_dc.benchmarks import BENCHMARK_DEFAULT_CONFIGS, run_benchmark
from resilient_dc.modeling import (
    aux_tensors,
    compute_metrics,
    ensure_bundles,
    fit_temperature,
    apply_temperature,
    select_abstention_threshold,
    abstention_mask,
    run_candidate,
    selective_metrics,
    train_deep,
    train_pbt,
    train_pc_tfn,
)
from source_method import fit_thresholds, predict_frame

RESULTS_DIR = ROOT / "results" / "final_evaluation"
MODEL_DIR = ROOT / "models" / "final"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_benchmark_configs() -> dict:
    configs = dict(BENCHMARK_DEFAULT_CONFIGS)
    tuning_dir = ROOT / "results" / "benchmark_tuning"
    for name in list(configs.keys()):
        best_path = tuning_dir / f"tuning_{name}_best.json"
        if best_path.exists():
            configs[name] = json.loads(best_path.read_text())["config"]
    return configs


def train_and_predict(method: str, config: dict, train, validation, test, external, seed: int, aux_train=None):
    """Fits once on `train` (early stopping / checkpoint selection against `validation`
    only), then reuses the fitted artifact to predict on test/external. Test and
    external data are never used for training, early stopping, or model selection."""
    if method == "PC_TFN":
        from resilient_dc.benchmarks import _sequence_predictor

        model, val_probs, details = train_pc_tfn(
            train, validation, seed, config, aux_train, MODEL_DIR / f"{method}_seed{seed}.pt")
        predictor = _sequence_predictor(model)
        return val_probs, predictor(test), predictor(external), details
    if method == "PC_TCN":
        from resilient_dc.benchmarks import _sequence_predictor

        model, val_probs, details = train_deep(method, train, validation, seed, config, MODEL_DIR / f"{method}_seed{seed}.pt")
        predictor = _sequence_predictor(model)
        return val_probs, predictor(test), predictor(external), details
    if method == "PBT":
        model, val_probs, details = train_pbt(train, validation, seed, config, MODEL_DIR / f"{method}_seed{seed}.joblib")
        return val_probs, model.predict_proba(test.tabular), model.predict_proba(external.tabular), details
    val_probs, details, predictor = run_benchmark(method, train, validation, seed, config, MODEL_DIR / "benchmarks")
    return val_probs, predictor(test), predictor(external), details


def evaluate_source_method(sites_frames: dict, split: str) -> dict:
    train_frames = {s: f.loc[f["split"] == "train"] for s, f in sites_frames.items() if s != "london"}
    thresholds = fit_thresholds(train_frames)
    preds = []
    for site, frame in sites_frames.items():
        subset = frame.loc[frame["split"] == split].copy()
        if subset.empty:
            continue
        subset["prediction"] = predict_frame(subset, thresholds)
        preds.append(subset[["target_class_6h", "prediction"]])
    result = pd.concat(preds, ignore_index=True)
    y_true = result["target_class_6h"].astype(int).to_numpy()
    y_pred = result["prediction"].astype(int).to_numpy()
    # The rule reproduction emits hard labels, not calibrated probabilities. We build a
    # one-hot "probability" only so label-based metrics (macro-F1, recall, FPR, ...)
    # can share the same compute_metrics() code path; NLL/Brier/ECE are not meaningful
    # for a method with no genuine confidence output and are overwritten with NaN below.
    probs = np.zeros((len(y_pred), 4))
    probs[np.arange(len(y_pred)), y_pred] = 1.0
    probs = probs * 0.97 + 0.03 / 4  # rows still sum to exactly 1
    metrics = compute_metrics(y_true, probs)
    for key in ("nll", "brier", "ece_10"):
        metrics[key] = float("nan")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[17, 29, 43, 71, 88])
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()

    final_cfg = yaml.safe_load((ROOT / "FINAL_MODEL_CONFIG.yaml").read_text())
    final_name = final_cfg["final_model"]["name"]
    final_config = final_cfg["final_model"]["config"]

    train, validation = ensure_bundles(ROOT, include_test=False)
    _, _ = ensure_bundles(ROOT, include_test=True)
    from resilient_dc.modeling import load_bundle
    cache = ROOT / "data" / "features_v4"
    test = load_bundle(cache / "test.npz")
    external = load_bundle(cache / "external_test.npz")

    benchmark_configs = load_benchmark_configs()
    # PBT is the efficiency-strong runner-up candidate (see MODEL_SELECTION_REPORT.md /
    # FINAL_MODEL_CONFIG.yaml selection rationale) whenever it is not itself the final
    # model; it is trained via train_pbt (not the benchmarks.py registry), so it must be
    # added explicitly here using its own tuned configuration.
    if final_name != "PBT":
        pbt_best = ROOT / "results" / "model_selection" / "tuning_pbt_best.json"
        if pbt_best.exists():
            pbt_config = json.loads(pbt_best.read_text())["config"]
            for int_key in ("n_estimators", "max_depth", "min_child_weight"):
                if int_key in pbt_config:
                    pbt_config[int_key] = int(pbt_config[int_key])
            benchmark_configs["PBT"] = pbt_config
    all_methods = {final_name: final_config, **{k: v for k, v in benchmark_configs.items() if args.methods is None or k in args.methods}}
    if args.methods is not None and final_name not in args.methods:
        all_methods.pop(final_name, None)

    # Preserve any already-computed rows (e.g. a prior run with different seeds or a
    # different --methods subset) rather than overwriting them. Only (method, seed) pairs
    # being recomputed in *this* invocation are dropped from the carry-over, so extending
    # to more seeds or adding a method never discards previously computed results.
    recompute_methods = set(all_methods.keys())
    if args.methods is None or "SOURCE_METHOD_REPRODUCTION" in (args.methods or []):
        recompute_methods.add("SOURCE_METHOD_REPRODUCTION")
    recompute_pairs = {(m, s) for m in recompute_methods for s in args.seeds}
    existing_path = RESULTS_DIR / "seed_level_results.csv"
    rows = []
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        keep_mask = ~existing.apply(lambda r: (r["method"], r["seed"]) in recompute_pairs, axis=1)
        rows = existing[keep_mask].to_dict("records")

    include_source = "SOURCE_METHOD_REPRODUCTION" in recompute_methods
    if include_source:
        sites_frames = {
            site: pd.read_parquet(ROOT / "data" / "processed" / f"{site}.parquet")
            for site in ["san_francisco", "phoenix", "chicago", "dallas", "london"]
        }
        source_test = evaluate_source_method(sites_frames, "test")
        source_external = evaluate_source_method(sites_frames, "external_test")
        for seed in args.seeds:
            for split_name, metrics in [("test", source_test), ("external_test", source_external)]:
                rows.append({
                    "method": "SOURCE_METHOD_REPRODUCTION", "seed": seed, "split": split_name, "wall_seconds": 0.0,
                    "temperature": 1.0, "abstain_rate": 0.0,
                    **{f"raw_{k}": v for k, v in metrics.items()}, **{f"cal_{k}": v for k, v in metrics.items()},
                    "parameters": 0, "train_seconds": 0.0, "epochs_completed": 0,
                })
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "seed_level_results.csv", index=False)
        print("SOURCE_METHOD_REPRODUCTION done (deterministic, no seed variance)")

    # Auxiliary supervision for the proposed model. Built from the training-split rows only;
    # it is a regression target, never an input, so the frozen splits stay untouched.
    aux_train = aux_tensors(ROOT, "train", train) if "PC_TFN" in all_methods else None

    for method, config in all_methods.items():
        for seed in args.seeds:
            started = time.perf_counter()
            val_probs, test_probs, external_probs, details = train_and_predict(
                method, config, train, validation, test, external, seed, aux_train)
            wall = time.perf_counter() - started

            temperature = fit_temperature(val_probs, validation.labels)
            abst_threshold = select_abstention_threshold(apply_temperature(val_probs, temperature), validation.labels)

            for split_name, probs, labels, bundle in [
                ("test", test_probs, test.labels, test),
                ("external_test", external_probs, external.labels, external),
            ]:
                raw_metrics = compute_metrics(labels, probs)
                calibrated = apply_temperature(probs, temperature)
                cal_metrics = compute_metrics(labels, calibrated)
                abstain = abstention_mask(calibrated, abst_threshold)
                row = {
                    "method": method, "seed": seed, "split": split_name, "wall_seconds": wall,
                    "temperature": temperature, "abstain_rate": float(abstain.mean()),
                    **{f"raw_{k}": v for k, v in raw_metrics.items()},
                    **{f"cal_{k}": v for k, v in cal_metrics.items()},
                    **selective_metrics(labels, calibrated, ~abstain),
                    **{k: v for k, v in details.items() if k not in {"history"}},
                }
                rows.append(row)
            pd.DataFrame(rows).to_csv(RESULTS_DIR / "seed_level_results.csv", index=False)
            print(f"{method} seed={seed} done in {wall:.1f}s")

    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.columns if c not in {"method", "seed", "split"}]
    agg = frame.groupby(["method", "split"])[numeric].agg(["mean", "std"])
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg.reset_index().to_csv(RESULTS_DIR / "aggregate_results.csv", index=False)
    print("Wrote", RESULTS_DIR / "aggregate_results.csv")


if __name__ == "__main__":
    main()
