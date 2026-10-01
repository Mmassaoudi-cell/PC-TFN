"""Robustness and low-data generalization experiments for the frozen final model.

Three experiments, each on the frozen test split (and the external London split where
noted), using models already fit under the frozen configuration in FINAL_MODEL_CONFIG.yaml:

1. Low-data fraction curve: retrain on stratified {1,5,10,25,50,100}% of `train`, evaluate on test.
2. Feature-noise robustness: additive Gaussian noise (in standardized units) injected into
   test-time inputs of a model trained on 100% of `train`; no retraining.
3. Missingness robustness: random masking (zero-in-standardized-space, i.e. mean-imputation)
   of test-time inputs of the same 100%-trained model; no retraining.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resilient_dc.modeling import (
    FeatureBundle, aux_tensors, compute_metrics, ensure_bundles, load_bundle, train_deep,
    train_pbt, train_pc_tfn,
)
from resilient_dc.benchmarks import _sequence_predictor

RESULTS_DIR = ROOT / "results" / "robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models" / "robustness"


def stratified_fraction(bundle: FeatureBundle, fraction: float, seed: int) -> tuple[FeatureBundle, np.ndarray]:
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in range(4):
        idx = np.flatnonzero(bundle.labels == cls)
        n = max(1, int(round(len(idx) * fraction)))
        chosen.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
    chosen = np.asarray(chosen)
    rng.shuffle(chosen)
    subset = FeatureBundle(
        sequence=bundle.sequence[chosen], physics=bundle.physics[chosen], tabular=bundle.tabular[chosen],
        labels=bundle.labels[chosen], sites=bundle.sites[chosen], timestamps=bundle.timestamps[chosen],
    )
    return subset, chosen


def add_gaussian_noise(bundle: FeatureBundle, std: float, seed: int, target: str) -> FeatureBundle:
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(bundle)
    if target in ("sequence", "both"):
        out.sequence = (out.sequence + rng.normal(0, std, out.sequence.shape)).astype(np.float32)
    if target in ("physics", "both"):
        out.physics = (out.physics + rng.normal(0, std, out.physics.shape)).astype(np.float32)
    if target == "tabular":
        out.tabular = (out.tabular + rng.normal(0, std, out.tabular.shape)).astype(np.float32)
    return out


def apply_missingness(bundle: FeatureBundle, rate: float, seed: int, target: str) -> FeatureBundle:
    """Standardized features have mean 0; masking to 0 approximates mean-imputation of missing sensors."""
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(bundle)
    if target in ("sequence", "both"):
        keep = (rng.random(out.sequence.shape[:2]) >= rate).astype(np.float32)  # per (sample, timestep)
        out.sequence = (out.sequence * keep[:, :, None]).astype(np.float32)
    if target in ("physics", "both"):
        keep = (rng.random(out.physics.shape) >= rate).astype(np.float32)
        out.physics = (out.physics * keep).astype(np.float32)
    if target == "tabular":
        keep = (rng.random(out.tabular.shape) >= rate).astype(np.float32)
        out.tabular = (out.tabular * keep).astype(np.float32)
    return out


def fit_final(name: str, config: dict, train, validation, seed: int, tag: str, aux_train=None):
    if name == "PC_TFN":
        # the low-data curve subsets `train`, so the auxiliary targets must be subset the same way
        model, _, _ = train_pc_tfn(train, validation, seed, config, aux_train, MODEL_DIR / f"{tag}_seed{seed}.pt")
        return _sequence_predictor(model)
    if name == "PC_TCN":
        model, _, _ = train_deep(name, train, validation, seed, config, MODEL_DIR / f"{tag}_seed{seed}.pt")
        return _sequence_predictor(model)
    model, _, _ = train_pbt(train, validation, seed, config, MODEL_DIR / f"{tag}_seed{seed}.joblib")
    return lambda bundle: model.predict_proba(bundle.tabular)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[17, 29, 43, 71, 88])
    args = parser.parse_args()

    final_cfg = yaml.safe_load((ROOT / "FINAL_MODEL_CONFIG.yaml").read_text())
    name = final_cfg["final_model"]["name"]
    config = final_cfg["final_model"]["config"]
    seq_or_tab = "sequence" if name in {"PC_TCN", "PC_TFN"} else "tabular"

    train, validation = ensure_bundles(ROOT, include_test=False)
    cache = ROOT / "data" / "features_v4"
    test = load_bundle(cache / "test.npz")
    external = load_bundle(cache / "external_test.npz")
    aux_full = aux_tensors(ROOT, "train", train) if name == "PC_TFN" else None

    # 1) Low-data fraction curve.
    low_data_rows = []
    for fraction in [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]:
        for seed in args.seeds:
            if fraction == 1.0:
                train_subset, aux_subset = train, aux_full
            else:
                train_subset, chosen = stratified_fraction(train, fraction, seed)
                aux_subset = None if aux_full is None else {k: v[chosen] for k, v in aux_full.items()}
            predictor = fit_final(name, config, train_subset, validation, seed,
                                  f"lowdata_{int(fraction*100)}pct", aux_subset)
            for split_name, bundle in [("test", test), ("external_test", external)]:
                metrics = compute_metrics(bundle.labels, predictor(bundle))
                low_data_rows.append({"fraction": fraction, "seed": seed, "split": split_name, "train_samples": len(train_subset.labels), **metrics})
        pd.DataFrame(low_data_rows).to_csv(RESULTS_DIR / "low_data_curve.csv", index=False)
        print(f"low-data fraction {fraction} done")

    # Reference model trained once per seed on 100% of train, reused for noise/missingness sweeps.
    predictors = {seed: fit_final(name, config, train, validation, seed, "full", aux_full)
                  for seed in args.seeds}

    noise_rows = []
    for std in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
        for seed in args.seeds:
            noisy_test = add_gaussian_noise(test, std, seed, seq_or_tab)
            metrics = compute_metrics(test.labels, predictors[seed](noisy_test))
            noise_rows.append({"noise_std": std, "seed": seed, "split": "test", **metrics})
        pd.DataFrame(noise_rows).to_csv(RESULTS_DIR / "feature_noise_curve.csv", index=False)
        print(f"noise std {std} done")

    missing_rows = []
    for rate in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
        for seed in args.seeds:
            missing_test = apply_missingness(test, rate, seed, seq_or_tab)
            metrics = compute_metrics(test.labels, predictors[seed](missing_test))
            missing_rows.append({"missing_rate": rate, "seed": seed, "split": "test", **metrics})
        pd.DataFrame(missing_rows).to_csv(RESULTS_DIR / "missingness_curve.csv", index=False)
        print(f"missing rate {rate} done")


if __name__ == "__main__":
    main()
