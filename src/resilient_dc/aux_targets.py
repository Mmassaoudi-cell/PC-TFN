"""Auxiliary supervision targets for the forecast-then-classify architecture.

The published label y_t = max_{h=1..6} class(risk_{t+h}) collapses the whole six-hour
risk trajectory into one ordinal symbol. An oracle study on the validation split shows
that handing a model the *true* future thermal trajectory (temperature, relative
humidity, dew point over t+1..t+6) lifts macro-F1 from 0.80 to 0.92, while the true
future stochastic variables (precipitation, snowfall, gusts) add nothing. The thermal
trajectory is therefore the binding latent variable, and this module materialises it as
auxiliary regression targets so the encoder can be supervised to forecast it.

Targets are derived from the same disclosed simulator that defines the label and are
consumed only as training-split supervision; they are never inputs at inference time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_pipeline import simulate_facility

HORIZON = 6
THERMAL_TARGETS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m"]
RISK_CUTS = np.array([0.08, 0.36, 0.72], dtype=np.float32)
ALL_SITES = ["san_francisco", "phoenix", "chicago", "dallas", "london"]


def risk_to_class(risk: np.ndarray) -> np.ndarray:
    """Disclosed severity cut-points: normal / advisory / warning / critical."""
    return np.searchsorted(RISK_CUTS, risk, side="right").astype(np.int64)


def _future_matrix(values: np.ndarray) -> np.ndarray:
    """(n, HORIZON, d) stack of values at t+1 .. t+HORIZON, NaN-padded at the tail."""
    n = len(values)
    out = np.full((n, HORIZON) + values.shape[1:], np.nan, dtype=np.float32)
    for h in range(1, HORIZON + 1):
        out[: n - h, h - 1] = values[h:]
    return out


def _site_targets(root: Path, site: str) -> pd.DataFrame:
    raw = pd.read_csv(root / "data" / "raw" / "weather" / f"{site}_era5_land.csv")
    sim = simulate_facility(raw, site)
    thermal = _future_matrix(sim[THERMAL_TARGETS].to_numpy(dtype=np.float32))
    risk = _future_matrix(sim["risk_score"].to_numpy(dtype=np.float32).reshape(-1, 1))[:, :, 0]
    frame = pd.DataFrame({
        "site": site,
        "timestamp": sim["timestamp"].dt.tz_localize(None).to_numpy("datetime64[ns]"),
    })
    for h in range(HORIZON):
        for j, name in enumerate(THERMAL_TARGETS):
            frame[f"{name}_t{h + 1}"] = thermal[:, h, j]
        frame[f"risk_t{h + 1}"] = risk[:, h]
    return frame


def _key(sites: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    ts = pd.Series(timestamps).astype("datetime64[ns]").astype("int64").astype(str)
    return (pd.Series(sites).astype(str) + "|" + ts).to_numpy()


def build_aux_targets(root: Path, sites: np.ndarray, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    """Return {'thermal': (n, 6, 3), 'risk': (n, 6)} aligned to the given sample keys."""
    table = pd.concat([_site_targets(root, s) for s in ALL_SITES], ignore_index=True)
    table["key"] = _key(table["site"].to_numpy(), table["timestamp"].to_numpy())
    table = table.set_index("key")
    rows = table.reindex(_key(sites, timestamps))
    thermal = np.stack(
        [np.stack([rows[f"{n}_t{h + 1}"].to_numpy(dtype=np.float32) for n in THERMAL_TARGETS], axis=1)
         for h in range(HORIZON)], axis=1)
    risk = np.stack([rows[f"risk_t{h + 1}"].to_numpy(dtype=np.float32) for h in range(HORIZON)], axis=1)
    if not np.isfinite(thermal).all() or not np.isfinite(risk).all():
        raise RuntimeError("auxiliary targets contain non-finite values; sample keys did not align")
    return {"thermal": thermal, "risk": risk}


def ensure_aux_targets(root: Path, cache_dir: Path, split: str, sites: np.ndarray,
                       timestamps: np.ndarray) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}_aux.npz"
    if path.exists():
        data = np.load(path)
        if len(data["risk"]) == len(sites):
            return {"thermal": data["thermal"], "risk": data["risk"]}
    aux = build_aux_targets(root, sites, timestamps)
    np.savez_compressed(path, **aux)
    return aux
