"""Benchmark suite: classical, boosting, and generic-deep baselines.

All benchmarks are trained once on the `train` split, with any early stopping or
model checkpointing decided against the `validation` split only. Each `train_*`
function returns `(validation_probabilities, details, predictor)`, where `predictor`
is a callable that maps a `FeatureBundle` (test, external, or otherwise) to class
probabilities using the *already-fitted* model — so evaluating on the frozen test
split never re-enters training or re-uses test/external data for model selection.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from .modeling import FeatureBundle, class_weights, predict_deep, train_deep, train_tabular_deep

Predictor = Callable[[FeatureBundle], np.ndarray]

SEQUENCE_FEATURE_COUNT = 15
CURRENT_FEATURE_IDX = list(range(0, SEQUENCE_FEATURE_COUNT)) + list(range(6 * SEQUENCE_FEATURE_COUNT + 5, 6 * SEQUENCE_FEATURE_COUNT + 5 + 8))


def current_only_features(bundle: FeatureBundle) -> np.ndarray:
    """Current-hour sensor snapshot + calendar/site one-hot, excluding 24h history stats and the explicit physics risk score."""
    return bundle.tabular[:, CURRENT_FEATURE_IDX]


def stratified_subsample(bundle: FeatureBundle, n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in range(4):
        idx = np.flatnonzero(bundle.labels == cls)
        chosen.extend(rng.choice(idx, min(n_per_class, len(idx)), replace=False).tolist())
    chosen = np.asarray(chosen)
    rng.shuffle(chosen)
    return bundle.tabular[chosen], bundle.labels[chosen]


def _save(model, output_path: Path | None) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)


def _tabular_sklearn(model, output_path: Path | None) -> Predictor:
    _save(model, output_path)
    return lambda bundle: model.predict_proba(bundle.tabular)


def train_logreg(train: FeatureBundle, validation: FeatureBundle, seed: int, config: dict, output_path: Path | None = None):
    model = LogisticRegression(C=config.get("C", 1.0), max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed, n_jobs=8)
    started = time.perf_counter()
    model.fit(train.tabular, train.labels)
    train_seconds = time.perf_counter() - started
    predictor = _tabular_sklearn(model, output_path)
    details = {"train_seconds": train_seconds, "parameters": model.coef_.size + model.intercept_.size, "epochs_completed": 1}
    return predictor(validation), details, predictor


def train_linear_svm(train: FeatureBundle, validation: FeatureBundle, seed: int, config: dict, output_path: Path | None = None):
    base = LinearSVC(C=config.get("C", 0.5), class_weight="balanced", random_state=seed, max_iter=5000, dual="auto")
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    started = time.perf_counter()
    model.fit(train.tabular, train.labels)
    train_seconds = time.perf_counter() - started
    predictor = _tabular_sklearn(model, output_path)
    details = {"train_seconds": train_seconds, "parameters": train.tabular.shape[1] * 4 * 3, "epochs_completed": 1}
    return predictor(validation), details, predictor


def train_knn(train: FeatureBundle, validation: FeatureBundle, seed: int, config: dict, output_path: Path | None = None):
    n_per_class = config.get("n_per_class", 15000)
    X_train, y_train = stratified_subsample(train, n_per_class, seed)
    model = KNeighborsClassifier(n_neighbors=config.get("k", 25), weights="distance", n_jobs=8)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started
    predictor = _tabular_sklearn(model, output_path)
    details = {"train_seconds": train_seconds, "parameters": len(y_train), "epochs_completed": 1}
    return predictor(validation), details, predictor


def train_random_forest(train: FeatureBundle, validation: FeatureBundle, seed: int, config: dict, output_path: Path | None = None):
    model = RandomForestClassifier(
        n_estimators=config.get("n_estimators", 400), max_depth=config.get("max_depth", None),
        min_samples_leaf=config.get("min_samples_leaf", 2), class_weight="balanced_subsample", random_state=seed, n_jobs=8,
    )
    started = time.perf_counter()
    model.fit(train.tabular, train.labels)
    train_seconds = time.perf_counter() - started
    predictor = _tabular_sklearn(model, output_path)
    details = {"train_seconds": train_seconds, "parameters": sum(t.tree_.node_count for t in model.estimators_), "epochs_completed": 1}
    return predictor(validation), details, predictor


def train_extra_trees(train: FeatureBundle, validation: FeatureBundle, seed: int, config: dict, output_path: Path | None = None):
    model = ExtraTreesClassifier(
        n_estimators=config.get("n_estimators", 500), max_depth=config.get("max_depth", None),
        min_samples_leaf=config.get("min_samples_leaf", 2), class_weight="balanced_subsample", random_state=seed, n_jobs=8,
    )
    started = time.perf_counter()
    model.fit(train.tabular, train.labels)
    train_seconds = time.perf_counter() - started
    predictor = _tabular_sklearn(model, output_path)
    details = {"train_seconds": train_seconds, "parameters": sum(t.tree_.node_count for t in model.estimators_), "epochs_completed": 1}
    return predictor(validation), details, predictor


def train_hist_gb(train: FeatureBundle, validation: FeatureBundle, seed: int, config: dict, output_path: Path | None = None):
    weights = class_weights(train.labels)[train.labels]
    model = HistGradientBoostingClassifier(
        max_iter=config.get("max_iter", 300), max_depth=config.get("max_depth", None),
        learning_rate=config.get("learning_rate", 0.08), l2_regularization=config.get("l2_regularization", 0.0), random_state=seed,
    )
    started = time.perf_counter()
    model.fit(train.tabular, train.labels, sample_weight=weights)
    train_seconds = time.perf_counter() - started
    predictor = _tabular_sklearn(model, output_path)
    details = {"train_seconds": train_seconds, "parameters": model.n_iter_ * 31, "epochs_completed": model.n_iter_}
    return predictor(validation), details, predictor


def train_xgb_current(train: FeatureBundle, validation: FeatureBundle, seed: int, config: dict, output_path: Path | None = None):
    weights = class_weights(train.labels)[train.labels]
    model = XGBClassifier(
        n_estimators=config.get("n_estimators", 400), max_depth=config.get("max_depth", 5),
        learning_rate=config.get("learning_rate", 0.1), subsample=0.85, colsample_bytree=0.85,
        objective="multi:softprob", num_class=4, eval_metric="mlogloss", tree_method="hist", random_state=seed, n_jobs=8,
    )
    started = time.perf_counter()
    model.fit(current_only_features(train), train.labels, sample_weight=weights, verbose=False)
    train_seconds = time.perf_counter() - started
    _save(model, output_path)
    predictor: Predictor = lambda bundle: model.predict_proba(current_only_features(bundle))
    details = {"train_seconds": train_seconds, "parameters": int(sum(t.count("leaf") for t in model.get_booster().get_dump())), "epochs_completed": config.get("n_estimators", 400)}
    return predictor(validation), details, predictor


def train_lightgbm(train: FeatureBundle, validation: FeatureBundle, seed: int, config: dict, output_path: Path | None = None):
    weights = class_weights(train.labels)[train.labels]
    model = LGBMClassifier(
        n_estimators=config.get("n_estimators", 500), max_depth=config.get("max_depth", -1),
        num_leaves=config.get("num_leaves", 63), learning_rate=config.get("learning_rate", 0.06),
        subsample=config.get("subsample", 0.85), colsample_bytree=config.get("colsample_bytree", 0.85),
        reg_lambda=config.get("reg_lambda", 1.0), objective="multiclass", num_class=4, random_state=seed, n_jobs=8, verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(train.tabular, train.labels, sample_weight=weights)
    train_seconds = time.perf_counter() - started
    predictor = _tabular_sklearn(model, output_path)
    details = {"train_seconds": train_seconds, "parameters": model.booster_.num_trees() * 63, "epochs_completed": model.booster_.num_trees()}
    return predictor(validation), details, predictor


def train_catboost(train: FeatureBundle, validation: FeatureBundle, seed: int, config: dict, output_path: Path | None = None):
    weights = class_weights(train.labels)[train.labels]
    model = CatBoostClassifier(
        iterations=config.get("iterations", 500), depth=config.get("depth", 6),
        learning_rate=config.get("learning_rate", 0.08), l2_leaf_reg=config.get("l2_leaf_reg", 3.0),
        loss_function="MultiClass", random_seed=seed, verbose=False, thread_count=8,
    )
    started = time.perf_counter()
    model.fit(train.tabular, train.labels, sample_weight=weights)
    train_seconds = time.perf_counter() - started
    predictor = _tabular_sklearn(model, output_path)
    details = {"train_seconds": train_seconds, "parameters": model.tree_count_ * 63, "epochs_completed": model.tree_count_}
    return predictor(validation), details, predictor


SKLEARN_FAMILY = {
    "LogReg": train_logreg, "LinearSVM": train_linear_svm, "kNN": train_knn,
    "RandomForest": train_random_forest, "ExtraTrees": train_extra_trees, "HistGB": train_hist_gb,
    "XGB_current": train_xgb_current, "LightGBM": train_lightgbm, "CatBoost": train_catboost,
}

SEQUENCE_DEEP = {"CNN", "GRU", "Transformer", "PG_GRU", "PTT"}
TABULAR_DEEP = {"MLP", "FT_Transformer"}


def _sequence_predictor(model) -> Predictor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predictor(bundle: FeatureBundle) -> np.ndarray:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(bundle.sequence), torch.from_numpy(bundle.physics), torch.from_numpy(bundle.labels)),
            batch_size=2048, shuffle=False,
        )
        return predict_deep(model, loader, device)

    return predictor


def _tabular_deep_predictor(model) -> Predictor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predictor(bundle: FeatureBundle) -> np.ndarray:
        model.eval()
        outputs = []
        loader = DataLoader(TensorDataset(torch.from_numpy(bundle.tabular)), batch_size=2048, shuffle=False)
        with torch.no_grad():
            for (x,) in loader:
                outputs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        return np.concatenate(outputs)

    return predictor


def run_benchmark(name: str, train: FeatureBundle, validation: FeatureBundle, seed: int, config: dict, model_dir: Path):
    """Returns (validation_probabilities, details, predictor). `predictor(bundle)` reuses the fitted model."""
    if name in SKLEARN_FAMILY:
        path = model_dir / f"{name}_seed{seed}.joblib"
        return SKLEARN_FAMILY[name](train, validation, seed, config, path)
    if name in SEQUENCE_DEEP:
        path = model_dir / f"{name}_seed{seed}.pt"
        model, val_probs, details = train_deep(name, train, validation, seed, config, path)
        return val_probs, {k: v for k, v in details.items() if k != "history"}, _sequence_predictor(model)
    if name in TABULAR_DEEP:
        path = model_dir / f"{name}_seed{seed}.pt"
        model, val_probs, details = train_tabular_deep(name, train, validation, seed, config, path)
        return val_probs, {k: v for k, v in details.items() if k != "history"}, _tabular_deep_predictor(model)
    raise ValueError(name)


BENCHMARK_DEFAULT_CONFIGS = {
    "LogReg": {"C": 1.0},
    "LinearSVM": {"C": 0.5},
    "kNN": {"k": 25, "n_per_class": 15000},
    "RandomForest": {"n_estimators": 400, "min_samples_leaf": 2},
    "ExtraTrees": {"n_estimators": 500, "min_samples_leaf": 2},
    "HistGB": {"max_iter": 300, "learning_rate": 0.08},
    "XGB_current": {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.1},
    "LightGBM": {"n_estimators": 500, "num_leaves": 63, "learning_rate": 0.06},
    "CatBoost": {"iterations": 500, "depth": 6, "learning_rate": 0.08},
    "CNN": {"channels": 48, "dropout": 0.15, "epochs": 12, "lr": 8e-4, "batch_size": 512, "focal": False},
    "GRU": {"hidden": 48, "dropout": 0.15, "epochs": 12, "lr": 8e-4, "batch_size": 512, "focal": False},
    "Transformer": {"hidden": 48, "heads": 4, "layers": 2, "dropout": 0.15, "epochs": 12, "lr": 7e-4, "batch_size": 512, "focal": False},
    "MLP": {"hidden": 128, "dropout": 0.2, "epochs": 14, "lr": 1e-3, "batch_size": 512},
    "FT_Transformer": {"hidden": 32, "heads": 4, "layers": 2, "dropout": 0.15, "epochs": 14, "lr": 8e-4, "batch_size": 512},
    # Physics-hybrid competitors: screened with 3 seeds in candidate selection
    # (results/model_selection/screen_aggregate.csv) but not carried into the
    # top-2 Optuna tuning budget (see MODEL_SELECTION_REPORT.md). Kept here as
    # benchmarks with their screened default configuration, not re-tuned, so the
    # comparison against the final model is explicitly disclosed as "screened,
    # not further tuned" rather than presented as a fully tuned competitor.
    "PG_GRU": {"hidden": 48, "dropout": 0.15, "epochs": 10, "lr": 8e-4, "batch_size": 512, "gamma": 1.5},
    "PTT": {"hidden": 48, "heads": 4, "layers": 2, "dropout": 0.15, "epochs": 10, "lr": 7e-4, "batch_size": 512, "gamma": 1.5},
}
