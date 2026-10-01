"""Ablation study for the frozen final model, run after Step 15 (untouched test evaluation).

Ablation only removes/replaces components of the ALREADY-FROZEN final architecture; it
never feeds back into architecture selection. Evaluated on the frozen test split and
the external (London) split, 5 seeds per variant.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resilient_dc.modeling import (
    aux_tensors, compute_metrics, ensure_bundles, load_bundle, train_deep, train_pbt, train_pc_tfn,
)
from resilient_dc.benchmarks import _sequence_predictor

RESULTS_DIR = ROOT / "results" / "ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models" / "ablation"


def pc_tcn_variants(base_config: dict) -> dict[str, dict]:
    no_physics = {**base_config, "use_physics": False, "use_gate": False}
    no_gate = {**base_config, "use_physics": True, "use_gate": False}
    no_focal = {**base_config, "focal": False}
    return {
        "Full_PC_TCN": base_config,
        "No_physics_branch": no_physics,
        "No_gate_(concat_fusion)": no_gate,
        "No_focal_loss_(weighted_CE)": no_focal,
    }


def pc_tfn_variants(base_config: dict) -> dict[str, dict]:
    """One mechanism removed at a time from the already-frozen PC-TFN architecture."""
    return {
        "Full_PC_TFN": base_config,
        "No_thermal_forecast_head": {**base_config, "use_thermal": False},
        "No_physics_surrogate": {**base_config, "use_surrogate": False},
        "No_expert_gate_(concat_fusion)": {**base_config, "use_gate": False},
        "Mean_pooling_(no_multi_head_readout)": {**base_config, "pooling": "mean"},
        "No_focal_loss_(weighted_CE)": {**base_config, "focal": False},
    }


def pbt_variants(base_config: dict) -> dict[str, dict]:
    # PBT ablation is implemented as a feature-mask flag consumed by run_ablation's PBT branch.
    return {
        "Full_PBT": {**base_config, "feature_mask": "all"},
        "No_physics_features": {**base_config, "feature_mask": "no_physics"},
        "No_temporal_stats_(current_only)": {**base_config, "feature_mask": "current_only"},
        "Unweighted_(no_class_weighting)": {**base_config, "feature_mask": "all", "no_class_weight": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[17, 29, 43, 71, 88, 101, 113, 127, 139, 151])
    args = parser.parse_args()

    final_cfg = yaml.safe_load((ROOT / "FINAL_MODEL_CONFIG.yaml").read_text())
    final_name = final_cfg["final_model"]["name"]
    final_config = final_cfg["final_model"]["config"]

    train, validation = ensure_bundles(ROOT, include_test=False)
    cache = ROOT / "data" / "features_v4"
    test = load_bundle(cache / "test.npz")
    external = load_bundle(cache / "external_test.npz")

    if final_name == "PC_TFN":
        variants = pc_tfn_variants(final_config)
    elif final_name == "PC_TCN":
        variants = pc_tcn_variants(final_config)
    elif final_name == "PBT":
        variants = pbt_variants(final_config)
    else:
        raise ValueError(f"No ablation plan defined for final model {final_name}")

    aux_train = aux_tensors(ROOT, "train", train) if final_name == "PC_TFN" else None

    rows = []
    for variant_name, config in variants.items():
        for seed in args.seeds:
            if final_name == "PC_TFN":
                cfg = {k: v for k, v in config.items()}
                model, val_probs, details = train_pc_tfn(
                    train, validation, seed, cfg, aux_train, MODEL_DIR / f"{variant_name}_seed{seed}.pt")
                predictor = _sequence_predictor(model)
                test_probs, ext_probs = predictor(test), predictor(external)
            elif final_name == "PC_TCN":
                cfg = {k: v for k, v in config.items()}
                model, val_probs, details = train_deep("PC_TCN", train, validation, seed, cfg, MODEL_DIR / f"{variant_name}_seed{seed}.pt")
                predictor = _sequence_predictor(model)
                test_probs, ext_probs = predictor(test), predictor(external)
            else:
                from resilient_dc.benchmarks import current_only_features
                import numpy as np

                mask = config.pop("feature_mask")
                no_cw = config.pop("no_class_weight", False)
                cfg = {k: v for k, v in config.items()}

                def slice_bundle(bundle):
                    if mask == "all":
                        return bundle.tabular
                    if mask == "current_only":
                        return current_only_features(bundle)
                    if mask == "no_physics":
                        idx = [i for i in range(bundle.tabular.shape[1]) if not (90 <= i < 95)]
                        return bundle.tabular[:, idx]
                    raise ValueError(mask)

                from xgboost import XGBClassifier
                from resilient_dc.modeling import class_weights
                import time

                weights = None if no_cw else class_weights(train.labels)[train.labels]
                model = XGBClassifier(
                    n_estimators=cfg.get("n_estimators", 500), max_depth=cfg.get("max_depth", 6),
                    learning_rate=cfg.get("learning_rate", 0.06), subsample=cfg.get("subsample", 0.85),
                    colsample_bytree=cfg.get("colsample_bytree", 0.85), min_child_weight=cfg.get("min_child_weight", 2),
                    reg_lambda=cfg.get("reg_lambda", 1.0), objective="multi:softprob", num_class=4,
                    eval_metric="mlogloss", tree_method="hist", random_state=seed, n_jobs=8,
                )
                started = time.perf_counter()
                model.fit(slice_bundle(train), train.labels, sample_weight=weights, verbose=False)
                train_seconds = time.perf_counter() - started
                test_probs = model.predict_proba(slice_bundle(test))
                ext_probs = model.predict_proba(slice_bundle(external))
                details = {"train_seconds": train_seconds}
                config["feature_mask"] = mask
                config["no_class_weight"] = no_cw

            for split_name, probs, bundle in [("test", test_probs, test), ("external_test", ext_probs, external)]:
                metrics = compute_metrics(bundle.labels, probs)
                rows.append({"variant": variant_name, "seed": seed, "split": split_name, **metrics})
            pd.DataFrame(rows).to_csv(RESULTS_DIR / "seed_level_results.csv", index=False)
            print(f"{variant_name} seed={seed} done")

    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.columns if c not in {"variant", "seed", "split"}]
    agg = frame.groupby(["variant", "split"])[numeric].agg(["mean", "std"])
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg.reset_index().to_csv(RESULTS_DIR / "aggregate_results.csv", index=False)


if __name__ == "__main__":
    main()
