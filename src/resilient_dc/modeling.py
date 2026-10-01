from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from xgboost import XGBClassifier

from .data_pipeline import SEQUENCE_FEATURES, simulate_facility


CLASS_NAMES = ["normal", "advisory", "warning", "critical"]
DEVELOPMENT_SITES = ["san_francisco", "phoenix", "chicago", "dallas"]
PHYSICS_FEATURES = ["inlet_temp_c", "cooling_headroom", "grid_stability", "battery_soc", "risk_score"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class FeatureBundle:
    sequence: np.ndarray
    physics: np.ndarray
    tabular: np.ndarray
    labels: np.ndarray
    sites: np.ndarray
    timestamps: np.ndarray


@dataclass
class Preprocessor:
    sequence_mean: np.ndarray
    sequence_std: np.ndarray
    physics_mean: np.ndarray
    physics_std: np.ndarray
    tabular_mean: np.ndarray
    tabular_std: np.ndarray

    def transform(self, bundle: FeatureBundle) -> FeatureBundle:
        return FeatureBundle(
            sequence=((bundle.sequence - self.sequence_mean) / self.sequence_std).astype(np.float32),
            physics=((bundle.physics - self.physics_mean) / self.physics_std).astype(np.float32),
            tabular=((bundle.tabular - self.tabular_mean) / self.tabular_std).astype(np.float32),
            labels=bundle.labels,
            sites=bundle.sites,
            timestamps=bundle.timestamps,
        )

    def save(self, path: Path) -> None:
        np.savez(
            path,
            sequence_mean=self.sequence_mean,
            sequence_std=self.sequence_std,
            physics_mean=self.physics_mean,
            physics_std=self.physics_std,
            tabular_mean=self.tabular_mean,
            tabular_std=self.tabular_std,
        )


def _temporal_stats(window: np.ndarray) -> np.ndarray:
    # last, mean, std, min, max, and end-to-start trend for each variable.
    return np.concatenate(
        [window[-1], window.mean(0), window.std(0), window.min(0), window.max(0), window[-1] - window[0]]
    )


def build_bundle(root: Path, split: str) -> FeatureBundle:
    seqs: list[np.ndarray] = []
    physics: list[np.ndarray] = []
    tabular: list[np.ndarray] = []
    labels: list[int] = []
    sites_out: list[str] = []
    timestamps: list[np.datetime64] = []
    requested_sites = DEVELOPMENT_SITES if split != "external_test" else ["london"]
    for site in requested_sites:
        targets = pd.read_parquet(root / "data" / "processed" / f"{site}.parquet")
        targets = targets.sort_values("timestamp").reset_index(drop=True)
        raw = pd.read_csv(root / "data" / "raw" / "weather" / f"{site}_era5_land.csv")
        full = simulate_facility(raw, site).dropna(subset=["target_class_6h"]).reset_index(drop=True)
        values = full[SEQUENCE_FEATURES].to_numpy(dtype=np.float32)
        pvalues = full[PHYSICS_FEATURES].to_numpy(dtype=np.float32)
        selected = targets.loc[targets["split"].eq(split)]
        for row in selected.itertuples(index=False):
            i = int(row.row_in_site)
            window = values[i - 23 : i + 1]
            ts = pd.Timestamp(row.timestamp)
            calendar = np.array(
                [
                    math.sin(2 * math.pi * ts.hour / 24),
                    math.cos(2 * math.pi * ts.hour / 24),
                    math.sin(2 * math.pi * ts.dayofyear / 365.25),
                    math.cos(2 * math.pi * ts.dayofyear / 365.25),
                ],
                dtype=np.float32,
            )
            site_onehot = np.zeros(len(DEVELOPMENT_SITES), dtype=np.float32)
            if site in DEVELOPMENT_SITES:
                site_onehot[DEVELOPMENT_SITES.index(site)] = 1.0
            seqs.append(window)
            physics.append(pvalues[i])
            tabular.append(np.concatenate([_temporal_stats(window), pvalues[i], calendar, site_onehot]))
            labels.append(int(row.target_class_6h))
            sites_out.append(site)
            timestamps.append(np.datetime64(ts.tz_localize(None)))
    return FeatureBundle(
        sequence=np.stack(seqs).astype(np.float32),
        physics=np.stack(physics).astype(np.float32),
        tabular=np.stack(tabular).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        sites=np.asarray(sites_out),
        timestamps=np.asarray(timestamps),
    )


def fit_preprocessor(train: FeatureBundle) -> Preprocessor:
    sequence_mean = train.sequence.mean(axis=(0, 1), keepdims=True)
    sequence_std = train.sequence.std(axis=(0, 1), keepdims=True)
    physics_mean = train.physics.mean(axis=0, keepdims=True)
    physics_std = train.physics.std(axis=0, keepdims=True)
    tabular_mean = train.tabular.mean(axis=0, keepdims=True)
    tabular_std = train.tabular.std(axis=0, keepdims=True)
    return Preprocessor(
        sequence_mean,
        np.maximum(sequence_std, 1e-6),
        physics_mean,
        np.maximum(physics_std, 1e-6),
        tabular_mean,
        np.maximum(tabular_std, 1e-6),
    )


def save_bundle(bundle: FeatureBundle, path: Path) -> None:
    np.savez_compressed(
        path,
        sequence=bundle.sequence,
        physics=bundle.physics,
        tabular=bundle.tabular,
        labels=bundle.labels,
        sites=bundle.sites,
        timestamps=bundle.timestamps,
    )


def load_bundle(path: Path) -> FeatureBundle:
    data = np.load(path, allow_pickle=False)
    return FeatureBundle(**{key: data[key] for key in ["sequence", "physics", "tabular", "labels", "sites", "timestamps"]})


def ensure_bundles(root: Path, include_test: bool = False) -> tuple[FeatureBundle, FeatureBundle]:
    cache = root / "data" / "features_v4"
    cache.mkdir(parents=True, exist_ok=True)
    required = ["train", "validation"] + (["test", "external_test"] if include_test else [])
    raw_bundles: dict[str, FeatureBundle] = {}
    for split in required:
        path = cache / f"{split}_raw.npz"
        if not path.exists():
            bundle = build_bundle(root, split)
            save_bundle(bundle, path)
        raw_bundles[split] = load_bundle(path)
    pre_path = cache / "preprocessor.npz"
    pre = fit_preprocessor(raw_bundles["train"])
    pre.save(pre_path)
    for split, bundle in raw_bundles.items():
        transformed_path = cache / f"{split}.npz"
        save_bundle(pre.transform(bundle), transformed_path)
    return pre.transform(raw_bundles["train"]), pre.transform(raw_bundles["validation"])


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=4).astype(float)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    return weights / weights.mean()


def logits_from_probabilities(probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.log(np.clip(probabilities, eps, 1.0))


def fit_temperature(validation_probabilities: np.ndarray, validation_labels: np.ndarray) -> float:
    """Validation-only temperature scaling (Guo et al., 2017). Returns T minimizing NLL on validation logits."""
    logits = torch.from_numpy(logits_from_probabilities(validation_probabilities)).float()
    labels = torch.from_numpy(validation_labels).long()
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)

    def closure():
        optimizer.zero_grad()
        t = log_t.exp()
        loss = nn.functional.cross_entropy(logits / t, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = logits_from_probabilities(probabilities) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def abstention_mask(probabilities: np.ndarray, entropy_threshold: float) -> np.ndarray:
    """Flag samples whose predictive entropy (nats) exceeds a validation-selected threshold for human review."""
    eps = 1e-8
    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    return entropy > entropy_threshold


def select_abstention_threshold(probabilities: np.ndarray, labels: np.ndarray, target_abstain_rate: float = 0.10) -> float:
    eps = 1e-8
    entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)
    return float(np.quantile(entropy, 1 - target_abstain_rate))


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_pred = probabilities.argmax(1)
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    minority = y_true > 0
    normal = y_true == 0
    confidence = probabilities.max(1)
    correctness = y_pred == y_true
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pr_auc_macro": float(average_precision_score(y_bin, probabilities, average="macro")),
        "roc_auc_macro": float(roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr")),
        "minority_recall": float(np.mean(y_pred[minority] > 0)),
        "false_positive_rate": float(np.mean(y_pred[normal] > 0)),
        "false_negative_rate": float(np.mean(y_pred[minority] == 0)),
        "nll": float(log_loss(y_true, probabilities, labels=[0, 1, 2, 3])),
        "brier": float(np.mean(np.sum((probabilities - y_bin) ** 2, axis=1))),
        "ece_10": float(ece),
        "normal_f1": float(per_class[2][0]),
        "advisory_f1": float(per_class[2][1]),
        "warning_f1": float(per_class[2][2]),
        "critical_f1": float(per_class[2][3]),
        "samples": int(len(y_true)),
    }


class FocalLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, gamma: float = 1.5):
        super().__init__()
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)
        p = logp.exp()
        idx = torch.arange(len(target), device=target.device)
        pt = p[idx, target]
        return (-self.weights[target] * (1 - pt).pow(self.gamma) * logp[idx, target]).mean()


class GatedFusion(nn.Module):
    def __init__(self, temporal_dim: int, physics_dim: int, hidden: int):
        super().__init__()
        self.temporal = nn.Linear(temporal_dim, hidden)
        self.physics = nn.Sequential(nn.Linear(physics_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.gate = nn.Linear(hidden * 2, hidden)

    def forward(self, temporal: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.temporal(temporal))
        q = self.physics(physics)
        gate = torch.sigmoid(self.gate(torch.cat([h, q], dim=1)))
        return gate * h + (1 - gate) * q


class PhysicsGRU(nn.Module):
    def __init__(self, input_dim: int, physics_dim: int, hidden: int = 48, dropout: float = 0.15):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.fusion = GatedFusion(hidden, physics_dim, hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 4))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(self.fusion(h[-1], p))


class PhysicsTransformer(nn.Module):
    def __init__(self, input_dim: int, physics_dim: int, hidden: int = 48, heads: int = 4, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden)
        self.physics_projection = nn.Linear(physics_dim, hidden)
        self.position = nn.Parameter(torch.zeros(1, 25, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 2, dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 4))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        token = self.physics_projection(p).unsqueeze(1)
        z = torch.cat([token, self.input_projection(x)], dim=1) + self.position
        return self.head(self.encoder(z)[:, 0])


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PhysicsTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        physics_dim: int,
        channels: int = 48,
        blocks: int = 3,
        dropout: float = 0.15,
        use_physics: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        self.use_physics = use_physics
        self.use_gate = use_gate
        self.stem = nn.Conv1d(input_dim, channels, 1)
        self.blocks = nn.Sequential(*[ResidualTCNBlock(channels, 2**i, dropout) for i in range(blocks)])
        if use_physics and use_gate:
            self.fusion = GatedFusion(channels, physics_dim, channels)
        elif use_physics:
            self.fusion = nn.Sequential(nn.Linear(channels + physics_dim, channels), nn.GELU(), nn.LayerNorm(channels))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 4))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x.transpose(1, 2))).mean(dim=2)
        if self.use_physics and self.use_gate:
            h = self.fusion(h, p)
        elif self.use_physics:
            h = self.fusion(torch.cat([h, p], dim=1))
        return self.head(h)


class SqueezeExcite(nn.Module):
    """Channel re-weighting so the encoder can emphasise whichever physical driver is active."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inner = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, inner)
        self.fc2 = nn.Linear(inner, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.fc2(nn.functional.gelu(self.fc1(x.mean(dim=-1)))))
        return x * s.unsqueeze(-1)


class SeparableBlock(nn.Module):
    """Depthwise-separable dilated convolution + channel attention + residual, pre-norm.

    Factorising the 3-tap dilated convolution into a depthwise and a pointwise stage is what
    lets the replacement model cover the full 24-hour window at lower parameter cost than the
    dense convolutions of the original TCN tower.
    """

    def __init__(self, channels: int, dilation: int, dropout: float, kernel: int = 3, expansion: int = 2):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.depthwise = nn.Conv1d(channels, channels, kernel, padding=dilation * (kernel - 1) // 2,
                                   dilation=dilation, groups=channels)
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, channels * expansion, 1), nn.GELU(),
            nn.Conv1d(channels * expansion, channels, 1),
        )
        self.se = SqueezeExcite(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pointwise(self.depthwise(self.norm(x)))
        return x + self.drop(self.se(h))


class MultiPool(nn.Module):
    """Last-step + mean + max + learned-query attention read-out.

    Plain mean pooling (used by PC-TCN) averages the most recent hour into 23 older ones,
    which is the wrong inductive bias for a forecasting target.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, 1)
        self.project = nn.Sequential(nn.Linear(channels * 4, channels), nn.GELU(), nn.LayerNorm(channels))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(h).squeeze(1), dim=-1).unsqueeze(1)
        attended = (h * weights).sum(-1)
        return self.project(torch.cat([h[:, :, -1], h.mean(-1), h.amax(-1), attended], dim=1))


class ExpertGate(nn.Module):
    """Softmax generalisation of PC-TCN's two-branch sigmoid gate to `n_experts` branches."""

    def __init__(self, dim: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Linear(dim * n_experts, dim * n_experts)

    def forward(self, experts: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(experts, dim=1)
        logits = self.gate(stacked.flatten(1)).view_as(stacked)
        return (torch.softmax(logits, dim=1) * stacked).sum(dim=1)


class PhysicsTrajectoryNet(nn.Module):
    """PC-TFN: Physics-Calibrated Trajectory-Forecasting Network.

    Replaces PC-TCN's dilated-convolution-plus-mean-pooling tower. An oracle study on the
    validation split (see `tmp/probe_oracle.py`) shows the only information separating the
    0.80 plateau from a 0.92 ceiling is the next six hours of temperature, humidity and dew
    point; the stochastic variables (precipitation, snowfall, gusts) contribute nothing.
    PC-TFN therefore makes that latent trajectory an explicit, supervised intermediate:

      1. depthwise-separable multi-scale dilated encoder (dilations 1-2-4-8, receptive field
         31 > 24 steps, so every read-out position sees the whole window);
      2. multi-head pooling (last step, mean, max, learned attention) instead of mean pooling;
      3. a thermal-trajectory head predicting t+1..t+6 temperature/humidity/dew point as a
         residual on a persistence anchor, supervised by auxiliary regression;
      4. a physics surrogate mapping that forecast plus the current physics state onto the six
         future risk scores, also supervised;
      5. a softmax expert gate fusing temporal, physics-state and forecast-risk evidence.

    `forward` returns logits so the existing predictor/calibration paths work unchanged;
    `forward_with_aux` additionally exposes the auxiliary outputs used by the training loss.
    """

    THERMAL_CHANNELS = (0, 1, 2)  # temperature_2m, relative_humidity_2m, dew_point_2m

    def __init__(self, input_dim: int, physics_dim: int, channels: int = 40, dropout: float = 0.1,
                 dilations: Iterable[int] = (1, 2, 4, 8), horizon: int = 6,
                 use_thermal: bool = True, use_surrogate: bool = True, use_gate: bool = True,
                 pooling: str = "multi"):
        super().__init__()
        self.horizon = horizon
        self.n_thermal = len(self.THERMAL_CHANNELS)
        self.use_thermal, self.use_surrogate, self.use_gate = use_thermal, use_surrogate, use_gate
        self.stem = nn.Conv1d(input_dim * 2, channels, 1)
        self.blocks = nn.Sequential(*[SeparableBlock(channels, d, dropout) for d in dilations])
        self.pool = MultiPool(channels) if pooling == "multi" else None
        self.pool_norm = nn.LayerNorm(channels)
        self.physics = nn.Sequential(nn.Linear(physics_dim, channels), nn.GELU(), nn.LayerNorm(channels))
        if use_thermal:
            self.thermal_head = nn.Sequential(
                nn.Linear(channels, channels), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(channels, horizon * self.n_thermal),
            )
        if use_surrogate:
            self.surrogate = nn.Sequential(
                nn.Linear(horizon * self.n_thermal + physics_dim + channels, channels), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(channels, horizon),
            )
            self.risk_embed = nn.Sequential(nn.Linear(horizon + 2, channels), nn.GELU(), nn.LayerNorm(channels))
        n_experts = 2 + int(use_surrogate)
        self.fusion = (ExpertGate(channels, n_experts) if use_gate else
                       nn.Sequential(nn.Linear(channels * n_experts, channels), nn.GELU(), nn.LayerNorm(channels)))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 4))

    def forward_with_aux(self, x: torch.Tensor, p: torch.Tensor) -> dict[str, torch.Tensor]:
        # explicit velocity channels: a convolutional encoder otherwise has to spend capacity
        # rediscovering first differences, which dominate short-horizon thermal extrapolation
        delta = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)
        h = self.blocks(self.stem(torch.cat([x, delta], dim=2).transpose(1, 2)))
        h = self.pool(h) if self.pool is not None else self.pool_norm(h.mean(-1))
        experts = [h, self.physics(p)]
        out: dict[str, torch.Tensor] = {}
        anchor = x[:, -1, list(self.THERMAL_CHANNELS)]
        if self.use_thermal:
            thermal = anchor.unsqueeze(1) + self.thermal_head(h).view(-1, self.horizon, self.n_thermal)
        else:
            thermal = anchor.unsqueeze(1).expand(-1, self.horizon, -1)
        out["thermal"] = thermal
        if self.use_surrogate:
            risk = self.surrogate(torch.cat([thermal.flatten(1), p, h], dim=1))
            out["risk"] = risk
            summary = torch.cat([risk, risk.amax(1, keepdim=True), risk.mean(1, keepdim=True)], dim=1)
            experts.append(self.risk_embed(summary))
        fused = self.fusion(experts) if self.use_gate else self.fusion(torch.cat(experts, dim=1))
        out["logits"] = self.head(fused)
        return out

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x, p)["logits"]


class PlainGRU(nn.Module):
    """Temporal-only GRU baseline: no physics branch, no gating."""

    def __init__(self, input_dim: int, hidden: int = 48, dropout: float = 0.15):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 4))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1])


class PlainTransformer(nn.Module):
    """Temporal-only Transformer baseline: a learned CLS token replaces the physics token."""

    def __init__(self, input_dim: int, hidden: int = 48, heads: int = 4, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.position = nn.Parameter(torch.zeros(1, 25, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 2, dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 4))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        cls = self.cls.expand(x.shape[0], -1, -1)
        z = torch.cat([cls, self.input_projection(x)], dim=1) + self.position
        return self.head(self.encoder(z)[:, 0])


class FTTransformer(nn.Module):
    """Feature-tokenizer transformer over the flat tabular representation (Gorishniy et al., 2021), ignoring the raw sequence."""

    def __init__(self, tabular_dim: int, hidden: int = 32, heads: int = 4, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.tabular_dim = tabular_dim
        self.feature_tokens = nn.Parameter(torch.randn(1, tabular_dim, hidden) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(1, tabular_dim, hidden))
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 2, dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 4))

    def forward(self, x_tabular: torch.Tensor) -> torch.Tensor:
        tokens = x_tabular.unsqueeze(-1) * self.feature_tokens + self.feature_bias
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        z = torch.cat([cls, tokens], dim=1)
        return self.head(self.encoder(z)[:, 0])


class TabularMLP(nn.Module):
    def __init__(self, tabular_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tabular_dim, hidden), nn.GELU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.BatchNorm1d(hidden // 2), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden // 2, 4)

    def forward(self, x_tabular: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x_tabular))


class SimpleCNN(nn.Module):
    def __init__(self, input_dim: int, channels: int = 48, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, channels, 5, padding=2), nn.GELU(), nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 4))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x.transpose(1, 2)).squeeze(-1))


def make_deep_model(name: str, input_dim: int, physics_dim: int, config: dict) -> nn.Module:
    if name == "PG_GRU":
        return PhysicsGRU(input_dim, physics_dim, config.get("hidden", 48), config.get("dropout", 0.15))
    if name == "PTT":
        return PhysicsTransformer(
            input_dim, physics_dim, config.get("hidden", 48), config.get("heads", 4), config.get("layers", 2), config.get("dropout", 0.15)
        )
    if name == "PC_TCN":
        return PhysicsTCN(
            input_dim, physics_dim, config.get("channels", 48), config.get("blocks", 3), config.get("dropout", 0.15),
            config.get("use_physics", True), config.get("use_gate", True)
        )
    if name == "PC_TCN_no_physics":
        return PhysicsTCN(input_dim, physics_dim, config.get("channels", 48), config.get("blocks", 3), config.get("dropout", 0.15), use_physics=False, use_gate=False)
    if name == "PC_TCN_no_gate":
        return PhysicsTCN(input_dim, physics_dim, config.get("channels", 48), config.get("blocks", 3), config.get("dropout", 0.15), use_physics=True, use_gate=False)
    if name.startswith("PC_TFN"):
        return PhysicsTrajectoryNet(
            input_dim, physics_dim,
            channels=config.get("channels", 40), dropout=config.get("dropout", 0.1),
            dilations=tuple(config.get("dilations", (1, 2, 4, 8))),
            use_thermal=config.get("use_thermal", True),
            use_surrogate=config.get("use_surrogate", True),
            use_gate=config.get("use_gate", True),
            pooling=config.get("pooling", "multi"),
        )
    if name == "CNN":
        return SimpleCNN(input_dim, config.get("channels", 48), config.get("dropout", 0.15))
    if name == "GRU":
        return PlainGRU(input_dim, config.get("hidden", 48), config.get("dropout", 0.15))
    if name == "Transformer":
        return PlainTransformer(input_dim, config.get("hidden", 48), config.get("heads", 4), config.get("layers", 2), config.get("dropout", 0.15))
    raise ValueError(name)


def train_tabular_deep(
    name: str,
    train: FeatureBundle,
    validation: FeatureBundle,
    seed: int,
    config: dict,
    output_path: Path | None = None,
) -> tuple[nn.Module, np.ndarray, dict]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tabular_dim = train.tabular.shape[1]
    if name == "MLP":
        model = TabularMLP(tabular_dim, config.get("hidden", 128), config.get("dropout", 0.2)).to(device)
    elif name == "FT_Transformer":
        model = FTTransformer(tabular_dim, config.get("hidden", 32), config.get("heads", 4), config.get("layers", 2), config.get("dropout", 0.15)).to(device)
    else:
        raise ValueError(name)
    weights = torch.tensor(class_weights(train.labels), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=config.get("weight_decay", 1e-4))
    batch_size = config.get("batch_size", 512)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train.tabular), torch.from_numpy(train.labels)),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation.tabular), torch.from_numpy(validation.labels)),
        batch_size=2048, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    best_score, best_state, stale = -1.0, None, 0
    patience = config.get("patience", 3)
    started = time.perf_counter()
    history = []
    for epoch in range(config.get("epochs", 12)):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            probs = []
            for x, _ in val_loader:
                probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
            probabilities = np.concatenate(probs)
        score = f1_score(validation.labels, probabilities.argmax(1), average="macro")
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_macro_f1": float(score)})
        if score > best_score + 1e-5:
            best_score, best_state, stale = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = []
        for x, _ in val_loader:
            probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        probabilities = np.concatenate(probs)
    details = {
        "train_seconds": train_seconds,
        "epochs_completed": len(history),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "history": history,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": name, "config": config, "seed": seed, "state_dict": best_state}, output_path)
    return model, probabilities, details


def train_deep(
    name: str,
    train: FeatureBundle,
    validation: FeatureBundle,
    seed: int,
    config: dict,
    output_path: Path | None = None,
) -> tuple[nn.Module, np.ndarray, dict]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_deep_model(name, train.sequence.shape[2], train.physics.shape[1], config).to(device)
    weights = torch.tensor(class_weights(train.labels), dtype=torch.float32, device=device)
    criterion = FocalLoss(weights, config.get("gamma", 1.5)) if config.get("focal", True) else nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 8e-4), weight_decay=config.get("weight_decay", 1e-4))
    batch_size = config.get("batch_size", 512)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train.sequence), torch.from_numpy(train.physics), torch.from_numpy(train.labels)),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation.sequence), torch.from_numpy(validation.physics), torch.from_numpy(validation.labels)),
        batch_size=2048, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    best_score = -1.0
    best_state = None
    patience = config.get("patience", 3)
    stale = 0
    started = time.perf_counter()
    history = []
    for epoch in range(config.get("epochs", 10)):
        model.train()
        losses = []
        for x, p, y in loader:
            x, p, y = x.to(device, non_blocking=True), p.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, p), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(loss.item())
        probabilities = predict_deep(model, val_loader, device)
        score = f1_score(validation.labels, probabilities.argmax(1), average="macro")
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_macro_f1": float(score)})
        if score > best_score + 1e-5:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    train_seconds = time.perf_counter() - started
    assert best_state is not None
    model.load_state_dict(best_state)
    probabilities = predict_deep(model, val_loader, device)
    details = {
        "train_seconds": train_seconds,
        "epochs_completed": len(history),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "history": history,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": name, "config": config, "seed": seed, "state_dict": best_state}, output_path)
    return model, probabilities, details


def aux_tensors(root: Path, split: str, bundle: FeatureBundle) -> dict[str, np.ndarray]:
    """Auxiliary supervision for PC-TFN, standardized with train-split statistics only."""
    from .aux_targets import ensure_aux_targets

    cache = root / "data" / "features_v4"
    aux = ensure_aux_targets(root, cache, split, bundle.sites, bundle.timestamps)
    pre = np.load(cache / "preprocessor.npz")
    idx = list(PhysicsTrajectoryNet.THERMAL_CHANNELS)
    mean = pre["sequence_mean"].reshape(-1)[idx].astype(np.float32)
    std = pre["sequence_std"].reshape(-1)[idx].astype(np.float32)
    return {
        "thermal": ((aux["thermal"] - mean) / std).astype(np.float32),
        "risk": np.log1p(aux["risk"]).astype(np.float32),
    }


def train_pc_tfn(
    train: FeatureBundle,
    validation: FeatureBundle,
    seed: int,
    config: dict,
    aux_train: dict[str, np.ndarray],
    output_path: Path | None = None,
) -> tuple[nn.Module, np.ndarray, dict]:
    """Multi-task trainer for PC-TFN.

    Identical protocol to `train_deep` (AdamW, class-weighted focal loss, early stopping on
    validation macro-F1, gradient clipping) plus the two auxiliary regression terms. The
    auxiliary targets are training-split supervision only and are never model inputs, so the
    frozen test and external splits remain untouched.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_deep_model("PC_TFN", train.sequence.shape[2], train.physics.shape[1], config).to(device)
    weights = torch.tensor(class_weights(train.labels), dtype=torch.float32, device=device)
    criterion = (FocalLoss(weights, config.get("gamma", 1.0)).to(device) if config.get("focal", True)
                 else nn.CrossEntropyLoss(weight=weights))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1.5e-3),
                                  weight_decay=config.get("weight_decay", 1e-4))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train.sequence), torch.from_numpy(train.physics),
                      torch.from_numpy(train.labels), torch.from_numpy(aux_train["thermal"]),
                      torch.from_numpy(aux_train["risk"])),
        batch_size=config.get("batch_size", 512), shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation.sequence), torch.from_numpy(validation.physics),
                      torch.from_numpy(validation.labels)),
        batch_size=2048, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    epochs = config.get("epochs", 40)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.get("lr", 1.5e-3), total_steps=epochs * len(loader),
        pct_start=config.get("pct_start", 0.25),
    )
    lam_thermal = config.get("lambda_thermal", 1.0)
    lam_risk = config.get("lambda_risk", 1.0)
    best_score, best_state, stale = -1.0, None, 0
    patience = config.get("patience", 10)
    started = time.perf_counter()
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, p, y, thermal, risk in loader:
            x, p, y = x.to(device, non_blocking=True), p.to(device, non_blocking=True), y.to(device, non_blocking=True)
            thermal, risk = thermal.to(device, non_blocking=True), risk.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model.forward_with_aux(x, p)
            loss = criterion(out["logits"], y)
            if model.use_thermal:
                loss = loss + lam_thermal * nn.functional.huber_loss(out["thermal"], thermal, delta=1.0)
            if model.use_surrogate:
                loss = loss + lam_risk * nn.functional.huber_loss(out["risk"], risk, delta=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        probabilities = predict_deep(model, val_loader, device)
        score = f1_score(validation.labels, probabilities.argmax(1), average="macro")
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_macro_f1": float(score)})
        if score > best_score + 1e-5:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    train_seconds = time.perf_counter() - started
    assert best_state is not None
    model.load_state_dict(best_state)
    probabilities = predict_deep(model, val_loader, device)
    details = {
        "train_seconds": train_seconds,
        "epochs_completed": len(history),
        "parameters": int(sum(q.numel() for q in model.parameters())),
        "history": history,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": "PC_TFN", "config": config, "seed": seed, "state_dict": best_state}, output_path)
    return model, probabilities, details


def selective_metrics(y_true: np.ndarray, probabilities: np.ndarray, keep: np.ndarray,
                      prefix: str = "sel") -> dict[str, float]:
    """Metrics restricted to the auto-classified (non-deferred) hours.

    Reported alongside per-class coverage, because a deferral rule that mostly defers the
    rare severity classes can inflate macro-F1 while answering fewer of the hard questions.
    """
    counts_all = np.bincount(y_true, minlength=4).astype(float)
    counts_kept = np.bincount(y_true[keep], minlength=4).astype(float)
    out = {
        f"{prefix}_coverage": float(keep.mean()),
        f"{prefix}_samples": int(keep.sum()),
    }
    for k, name in enumerate(CLASS_NAMES):
        out[f"{prefix}_coverage_{name}"] = float(counts_kept[k] / max(counts_all[k], 1.0))
    if keep.sum() == 0 or len(np.unique(y_true[keep])) < 2:
        for key in ["macro_f1", "weighted_f1", "balanced_accuracy", "minority_recall",
                    "false_positive_rate", "false_negative_rate"]:
            out[f"{prefix}_{key}"] = float("nan")
        for name in CLASS_NAMES:
            out[f"{prefix}_{name}_f1"] = float("nan")
        return out
    y, probs = y_true[keep], probabilities[keep]
    y_pred = probs.argmax(1)
    minority, normal = y > 0, y == 0
    per_class = precision_recall_fscore_support(y, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    out.update({
        f"{prefix}_macro_f1": float(f1_score(y, y_pred, average="macro")),
        f"{prefix}_weighted_f1": float(f1_score(y, y_pred, average="weighted")),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        f"{prefix}_minority_recall": float(np.mean(y_pred[minority] > 0)) if minority.any() else float("nan"),
        f"{prefix}_false_positive_rate": float(np.mean(y_pred[normal] > 0)) if normal.any() else float("nan"),
        f"{prefix}_false_negative_rate": float(np.mean(y_pred[minority] == 0)) if minority.any() else float("nan"),
    })
    for k, name in enumerate(CLASS_NAMES):
        out[f"{prefix}_{name}_f1"] = float(per_class[2][k])
    return out


@torch.no_grad()
def predict_deep(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    for x, p, _ in loader:
        logits = model(x.to(device, non_blocking=True), p.to(device, non_blocking=True))
        output.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output)


def train_pbt(train: FeatureBundle, validation: FeatureBundle, seed: int, config: dict, output_path: Path | None = None):
    weights = class_weights(train.labels)[train.labels]
    model = XGBClassifier(
        n_estimators=config.get("n_estimators", 500),
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.06),
        subsample=config.get("subsample", 0.85),
        colsample_bytree=config.get("colsample_bytree", 0.85),
        min_child_weight=config.get("min_child_weight", 2),
        reg_lambda=config.get("reg_lambda", 1.0),
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=seed,
        n_jobs=8,
    )
    started = time.perf_counter()
    model.fit(train.tabular, train.labels, sample_weight=weights, verbose=False)
    train_seconds = time.perf_counter() - started
    probabilities = model.predict_proba(validation.tabular)
    details = {
        "train_seconds": train_seconds,
        "parameters": int(sum(tree.count("leaf") for tree in model.get_booster().get_dump())),
        "epochs_completed": config.get("n_estimators", 500),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
    return model, probabilities, details


DEFAULT_CONFIGS = {
    "PBT": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.06},
    "PG_GRU": {"hidden": 48, "dropout": 0.15, "epochs": 10, "lr": 8e-4, "batch_size": 512, "gamma": 1.5},
    "PTT": {"hidden": 48, "heads": 4, "layers": 2, "dropout": 0.15, "epochs": 10, "lr": 7e-4, "batch_size": 512, "gamma": 1.5},
    "PC_TCN": {"channels": 48, "blocks": 3, "dropout": 0.15, "epochs": 10, "lr": 8e-4, "batch_size": 512, "gamma": 1.5},
    "PC_TFN": {"channels": 40, "dropout": 0.1, "epochs": 40, "patience": 10, "lr": 1.5e-3,
               "weight_decay": 1e-4, "batch_size": 512, "gamma": 1.0,
               "lambda_thermal": 1.0, "lambda_risk": 1.0, "dilations": (1, 2, 4, 8)},
}


def run_candidate(
    name: str,
    train: FeatureBundle,
    validation: FeatureBundle,
    seed: int,
    config: dict,
    model_dir: Path,
) -> dict:
    suffix = ".joblib" if name == "PBT" else ".pt"
    path = model_dir / f"{name}_seed{seed}{suffix}"
    if name == "PBT":
        _, probabilities, details = train_pbt(train, validation, seed, config, path)
    else:
        _, probabilities, details = train_deep(name, train, validation, seed, config, path)
    metrics = compute_metrics(validation.labels, probabilities)
    return {"candidate": name, "seed": seed, **metrics, **{k: v for k, v in details.items() if k != "history"}}
