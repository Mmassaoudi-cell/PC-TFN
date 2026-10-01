from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resilient_dc.modeling import (
    DEFAULT_CONFIGS,
    ensure_bundles,
    run_candidate,
    train_deep,
    train_pbt,
    compute_metrics,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = ROOT / "results" / "model_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models" / "candidates" / "tuning"


def pc_tcn_space(trial: optuna.Trial) -> dict:
    return {
        "channels": trial.suggest_categorical("channels", [32, 48, 64, 96]),
        "blocks": trial.suggest_int("blocks", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.05, 0.35),
        "lr": trial.suggest_float("lr", 3e-4, 2.5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "gamma": trial.suggest_float("gamma", 0.5, 3.0),
        "epochs": 16,
        "patience": 4,
        "focal": True,
        "use_physics": True,
        "use_gate": True,
    }


def pbt_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 250, 900, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.18, log=True),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 9),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 6.0, log=True),
    }


def run_tuning(name: str, train, validation, n_trials: int, seed_for_search: int, top_k: int):
    space_fn = pc_tcn_space if name == "PC_TCN" else pbt_space
    trial_log = []

    def objective(trial: optuna.Trial) -> float:
        config = space_fn(trial)
        started = time.perf_counter()
        if name == "PC_TCN":
            _, probabilities, details = train_deep(name, train, validation, seed_for_search, config)
        else:
            _, probabilities, details = train_pbt(train, validation, seed_for_search, config)
        metrics = compute_metrics(validation.labels, probabilities)
        elapsed = time.perf_counter() - started
        row = {"trial": trial.number, **config, **metrics, "wall_seconds": elapsed}
        trial_log.append(row)
        pd.DataFrame(trial_log).to_csv(OUTPUT_DIR / f"tuning_{name.lower()}_trials.csv", index=False)
        return metrics["macro_f1"]

    sampler = optuna.samplers.TPESampler(seed=2026)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    confirm_rows = []
    # Re-derive top trials directly from the Optuna study (not the CSV) to avoid dtype round-trip issues.
    best_trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:top_k]
    for t in best_trials_sorted:
        config = dict(t.params)
        if name == "PC_TCN":
            config.update({"epochs": 16, "patience": 4, "focal": True, "use_physics": True, "use_gate": True})
        for seed in (17, 29, 43):
            if name == "PC_TCN":
                _, probabilities, details = train_deep(name, train, validation, seed, config)
            else:
                _, probabilities, details = train_pbt(train, validation, seed, config)
            metrics = compute_metrics(validation.labels, probabilities)
            confirm_rows.append({"trial": t.number, "seed": seed, **config, **metrics})
        pd.DataFrame(confirm_rows).to_csv(OUTPUT_DIR / f"tuning_{name.lower()}_confirm.csv", index=False)

    confirm_df = pd.DataFrame(confirm_rows)
    agg = confirm_df.groupby("trial")["macro_f1"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    best_trial_id = agg.index[0]
    best_config_rows = confirm_df[confirm_df["trial"] == best_trial_id]
    best_config = {k: best_config_rows.iloc[0][k] for k in best_trials_sorted[0].params.keys()}
    if name == "PC_TCN":
        best_config.update({"epochs": 16, "patience": 4, "focal": True, "use_physics": True, "use_gate": True})
    summary = {
        "candidate": name,
        "best_trial": int(best_trial_id),
        "mean_macro_f1": float(agg.loc[best_trial_id, "mean"]),
        "std_macro_f1": float(agg.loc[best_trial_id, "std"]),
        "config": best_config,
    }
    (OUTPUT_DIR / f"tuning_{name.lower()}_best.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["PC_TCN", "PBT", "both"], default="both")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    train, validation = ensure_bundles(ROOT, include_test=False)
    targets = ["PC_TCN", "PBT"] if args.candidate == "both" else [args.candidate]
    for name in targets:
        run_tuning(name, train, validation, args.trials, seed_for_search=17, top_k=args.top_k)


if __name__ == "__main__":
    main()
