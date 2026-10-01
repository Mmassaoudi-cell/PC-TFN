"""Backfill coverage-conditioned (selective-prediction) metrics for the benchmark suite.

The benchmarks were already trained under the frozen protocol and their fitted artifacts are
on disk, so their 90%-coverage operating point can be evaluated by reloading and predicting.
Nothing is retrained, no hyperparameter is changed, and the abstention threshold is still the
validation-only entropy quantile already used in the published run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resilient_dc.benchmarks import (  # noqa: E402
    BENCHMARK_DEFAULT_CONFIGS, SEQUENCE_DEEP, SKLEARN_FAMILY, TABULAR_DEEP,
    _sequence_predictor, _tabular_deep_predictor, current_only_features,
)
from resilient_dc.modeling import (  # noqa: E402
    FeatureBundle, apply_temperature, abstention_mask, fit_temperature, load_bundle,
    make_deep_model, select_abstention_threshold, selective_metrics, TabularMLP, FTTransformer,
)

RESULTS = ROOT / "results" / "final_evaluation"
CACHE = ROOT / "data" / "features_v4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sequence_model(path: Path, train: FeatureBundle):
    blob = torch.load(path, map_location=DEVICE, weights_only=False)
    model = make_deep_model(blob["model"], train.sequence.shape[2], train.physics.shape[1], blob["config"])
    model.load_state_dict(blob["state_dict"])
    return model.to(DEVICE).eval()


def load_tabular_deep(path: Path, train: FeatureBundle):
    blob = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg, dim = blob["config"], train.tabular.shape[1]
    if blob["model"] == "MLP":
        model = TabularMLP(dim, cfg.get("hidden", 128), cfg.get("dropout", 0.2))
    else:
        model = FTTransformer(dim, cfg.get("hidden", 32), cfg.get("heads", 4), cfg.get("layers", 2), cfg.get("dropout", 0.15))
    model.load_state_dict(blob["state_dict"])
    return model.to(DEVICE).eval()


def predictor_for(method: str, seed: int, train: FeatureBundle):
    """Rebuild the fitted predictor for (method, seed) from disk, or None if unavailable."""
    final_dir = ROOT / "models" / "final"
    bench_dir = final_dir / "benchmarks"
    if method in SKLEARN_FAMILY:
        path = bench_dir / f"{method}_seed{seed}.joblib"
        if not path.exists():
            return None
        model = joblib.load(path)
        if method == "XGB_current":
            return lambda b: model.predict_proba(current_only_features(b))
        return lambda b: model.predict_proba(b.tabular)
    if method == "PBT":
        path = final_dir / f"PBT_seed{seed}.joblib"
        if not path.exists():
            return None
        model = joblib.load(path)
        return lambda b: model.predict_proba(b.tabular)
    if method in SEQUENCE_DEEP or method in {"PC_TCN", "PC_TFN"}:
        path = (final_dir if method in {"PC_TCN", "PC_TFN"} else bench_dir) / f"{method}_seed{seed}.pt"
        if not path.exists():
            return None
        return _sequence_predictor(load_sequence_model(path, train))
    if method in TABULAR_DEEP:
        path = bench_dir / f"{method}_seed{seed}.pt"
        if not path.exists():
            return None
        return _tabular_deep_predictor(load_tabular_deep(path, train))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-abstain", type=float, default=0.10)
    args = ap.parse_args()

    train = load_bundle(CACHE / "train.npz")
    validation = load_bundle(CACHE / "validation.npz")
    test = load_bundle(CACHE / "test.npz")
    external = load_bundle(CACHE / "external_test.npz")

    frame = pd.read_csv(RESULTS / "seed_level_results.csv")
    methods = [m for m in frame["method"].unique() if m != "SOURCE_METHOD_REPRODUCTION"]
    updates: dict[tuple[str, int, str], dict] = {}

    for method in sorted(methods):
        seeds = sorted(frame.loc[frame["method"].eq(method), "seed"].unique())
        done = 0
        for seed in seeds:
            pred = predictor_for(method, int(seed), train)
            if pred is None:
                continue
            val_probs = pred(validation)
            temperature = fit_temperature(val_probs, validation.labels)
            threshold = select_abstention_threshold(
                apply_temperature(val_probs, temperature), validation.labels, args.target_abstain)
            for split_name, bundle in (("test", test), ("external_test", external)):
                calibrated = apply_temperature(pred(bundle), temperature)
                keep = ~abstention_mask(calibrated, threshold)
                updates[(method, int(seed), split_name)] = selective_metrics(bundle.labels, calibrated, keep)
            done += 1
        print(f"{method:28s} {done}/{len(seeds)} seeds reconstructed")

    sel_cols = sorted({k for v in updates.values() for k in v})
    for col in sel_cols:
        if col not in frame.columns:
            frame[col] = np.nan
    index = {(m, s, sp): i for i, (m, s, sp) in
             enumerate(zip(frame["method"], frame["seed"], frame["split"]))}
    for key, vals in updates.items():
        if key in index:
            for col, v in vals.items():
                frame.at[index[key], col] = v

    frame.to_csv(RESULTS / "seed_level_results.csv", index=False)
    numeric = [c for c in frame.columns if c not in {"method", "seed", "split"}]
    agg = frame.groupby(["method", "split"])[numeric].agg(["mean", "std"])
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg.reset_index().to_csv(RESULTS / "aggregate_results.csv", index=False)
    print("updated", RESULTS / "seed_level_results.csv")


if __name__ == "__main__":
    main()
