from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resilient_dc.benchmarks import BENCHMARK_DEFAULT_CONFIGS, run_benchmark
from resilient_dc.modeling import ensure_bundles, compute_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)
OUTPUT_DIR = ROOT / "results" / "benchmark_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models" / "benchmarks" / "tuning"

SPACES = {
    "LogReg": lambda t: {"C": t.suggest_float("C", 0.01, 20.0, log=True)},
    "LinearSVM": lambda t: {"C": t.suggest_float("C", 0.01, 5.0, log=True)},
    "kNN": lambda t: {"k": t.suggest_int("k", 5, 75, step=5), "n_per_class": 15000},
    "RandomForest": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": t.suggest_categorical("max_depth", [None, 10, 16, 24, 32]),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
    },
    "ExtraTrees": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 200, 900, step=100),
        "max_depth": t.suggest_categorical("max_depth", [None, 10, 16, 24, 32]),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
    },
    "HistGB": lambda t: {
        "max_iter": t.suggest_int("max_iter", 150, 500, step=50),
        "max_depth": t.suggest_categorical("max_depth", [None, 6, 10, 16]),
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "l2_regularization": t.suggest_float("l2_regularization", 0.0, 2.0),
    },
    "XGB_current": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 150, 700, step=50),
        "max_depth": t.suggest_int("max_depth", 3, 9),
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.25, log=True),
    },
    "LightGBM": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 200, 900, step=50),
        "num_leaves": t.suggest_int("num_leaves", 15, 150),
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample": t.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.55, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 0.05, 5.0, log=True),
    },
    "CatBoost": lambda t: {
        "iterations": t.suggest_int("iterations", 200, 700, step=50),
        "depth": t.suggest_int("depth", 4, 9),
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 0.5, 8.0, log=True),
    },
    "CNN": lambda t: {
        "channels": t.suggest_categorical("channels", [32, 48, 64, 96]),
        "dropout": t.suggest_float("dropout", 0.05, 0.35),
        "lr": t.suggest_float("lr", 3e-4, 2.5e-3, log=True),
        "batch_size": t.suggest_categorical("batch_size", [256, 512, 1024]),
        "epochs": 12, "focal": False,
    },
    "GRU": lambda t: {
        "hidden": t.suggest_categorical("hidden", [32, 48, 64, 96]),
        "dropout": t.suggest_float("dropout", 0.05, 0.35),
        "lr": t.suggest_float("lr", 3e-4, 2.5e-3, log=True),
        "batch_size": t.suggest_categorical("batch_size", [256, 512, 1024]),
        "epochs": 12, "focal": False,
    },
    "Transformer": lambda t: {
        "hidden": t.suggest_categorical("hidden", [32, 48, 64]),
        "layers": t.suggest_int("layers", 1, 3),
        "dropout": t.suggest_float("dropout", 0.05, 0.35),
        "lr": t.suggest_float("lr", 3e-4, 2e-3, log=True),
        "batch_size": t.suggest_categorical("batch_size", [256, 512, 1024]),
        "epochs": 12, "focal": False,
    },
    "MLP": lambda t: {
        "hidden": t.suggest_categorical("hidden", [64, 128, 192, 256]),
        "dropout": t.suggest_float("dropout", 0.05, 0.45),
        "lr": t.suggest_float("lr", 3e-4, 2.5e-3, log=True),
        "batch_size": t.suggest_categorical("batch_size", [256, 512, 1024]),
        "epochs": 14,
    },
    "FT_Transformer": lambda t: {
        "hidden": t.suggest_categorical("hidden", [16, 24, 32, 48]),
        "layers": t.suggest_int("layers", 1, 3),
        "dropout": t.suggest_float("dropout", 0.05, 0.35),
        "lr": t.suggest_float("lr", 3e-4, 2e-3, log=True),
        "batch_size": t.suggest_categorical("batch_size", [256, 512, 1024]),
        "epochs": 14,
    },
}

TRIAL_BUDGET = {
    "LogReg": 8, "LinearSVM": 6, "kNN": 6, "RandomForest": 10, "ExtraTrees": 10,
    "HistGB": 10, "XGB_current": 10, "LightGBM": 12, "CatBoost": 8,
    "CNN": 10, "GRU": 10, "Transformer": 8, "MLP": 10, "FT_Transformer": 8,
}

# Keys returned by a SPACES[name] closure that are fixed constants, not Optuna-suggested
# values (e.g. epoch budgets, loss-type switches). These must be merged back into the
# winning trial's config since study.best_trial.params only records suggested values.
FIXED_EXTRAS = {
    "LogReg": {}, "LinearSVM": {}, "kNN": {"n_per_class": 15000},
    "RandomForest": {}, "ExtraTrees": {}, "HistGB": {}, "XGB_current": {}, "LightGBM": {}, "CatBoost": {},
    "CNN": {"epochs": 12, "focal": False}, "GRU": {"epochs": 12, "focal": False}, "Transformer": {"epochs": 12, "focal": False},
    "MLP": {"epochs": 14}, "FT_Transformer": {"epochs": 14},
}


def tune_one(name: str, train, validation, seed: int = 17) -> dict:
    trial_log = []

    def objective(trial: optuna.Trial) -> float:
        config = SPACES[name](trial)
        probabilities, details, _ = run_benchmark(name, train, validation, seed, config, MODEL_DIR / "_scratch")
        metrics = compute_metrics(validation.labels, probabilities)
        row = {"trial": trial.number, **{k: v for k, v in config.items()}, **metrics}
        trial_log.append(row)
        pd.DataFrame(trial_log).to_csv(OUTPUT_DIR / f"tuning_{name}_trials.csv", index=False)
        return metrics["macro_f1"]

    sampler = optuna.samplers.TPESampler(seed=2026)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=TRIAL_BUDGET[name], show_progress_bar=False)
    best_config = {**dict(study.best_trial.params), **FIXED_EXTRAS[name]}
    summary = {"benchmark": name, "best_trial": study.best_trial.number, "best_macro_f1": study.best_value, "config": best_config}
    (OUTPUT_DIR / f"tuning_{name}_best.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="*", default=list(SPACES.keys()))
    args = parser.parse_args()
    train, validation = ensure_bundles(ROOT, include_test=False)
    for name in args.benchmarks:
        tune_one(name, train, validation)


if __name__ == "__main__":
    main()
