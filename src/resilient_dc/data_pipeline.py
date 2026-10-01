from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests


OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "snowfall",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
]

SITES = {
    "san_francisco": {"latitude": 37.7749, "longitude": -122.4194, "climate": "marine"},
    "phoenix": {"latitude": 33.4484, "longitude": -112.0740, "climate": "hot_arid"},
    "chicago": {"latitude": 41.8781, "longitude": -87.6298, "climate": "continental"},
    "dallas": {"latitude": 32.7767, "longitude": -96.7970, "climate": "hot_humid"},
    "london": {"latitude": 51.5074, "longitude": -0.1278, "climate": "external_marine"},
}

DEVELOPMENT_SITES = {"san_francisco", "phoenix", "chicago", "dallas"}
EXTERNAL_SITE = "london"


@dataclass(frozen=True)
class SplitPolicy:
    train_end: str = "2021-12-31 17:00:00+00:00"
    validation_start: str = "2022-01-02 00:00:00+00:00"
    validation_end: str = "2022-12-31 17:00:00+00:00"
    test_start: str = "2023-01-02 00:00:00+00:00"
    test_end: str = "2024-12-31 17:00:00+00:00"
    history_hours: int = 24
    horizon_hours: int = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_weather(raw_dir: Path, start: str = "2018-01-01", end: str = "2024-12-31") -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "accessed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "endpoint": OPEN_METEO_URL,
        "start_date": start,
        "end_date": end,
        "hourly_variables": HOURLY_VARIABLES,
        "timezone": "UTC",
        "sites": SITES,
        "files": {},
    }
    for site, meta in SITES.items():
        target = raw_dir / f"{site}_era5_land.csv"
        if not target.exists():
            params = {
                "latitude": meta["latitude"],
                "longitude": meta["longitude"],
                "start_date": start,
                "end_date": end,
                "hourly": ",".join(HOURLY_VARIABLES),
                "timezone": "UTC",
                "models": "era5_land",
            }
            response = requests.get(OPEN_METEO_URL, params=params, timeout=180)
            response.raise_for_status()
            payload = response.json()
            hourly = pd.DataFrame(payload["hourly"])
            hourly.insert(0, "site", site)
            hourly.to_csv(target, index=False)
            time.sleep(1.0)
        provenance["files"][target.name] = {
            "sha256": _sha256(target),
            "bytes": target.stat().st_size,
        }
    (raw_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def _site_parameters(site: str) -> dict[str, float]:
    # Fixed before model fitting. Values define a transparent research simulator,
    # not a claim about any named operator's actual facility.
    return {
        "san_francisco": {"it_capacity_mw": 36.0, "cooling_capacity_mw": 9.5, "battery_mwh": 32.0},
        "phoenix": {"it_capacity_mw": 44.0, "cooling_capacity_mw": 15.5, "battery_mwh": 45.0},
        "chicago": {"it_capacity_mw": 40.0, "cooling_capacity_mw": 10.5, "battery_mwh": 38.0},
        "dallas": {"it_capacity_mw": 46.0, "cooling_capacity_mw": 14.0, "battery_mwh": 48.0},
        "london": {"it_capacity_mw": 38.0, "cooling_capacity_mw": 10.0, "battery_mwh": 34.0},
    }[site]


def simulate_facility(weather: pd.DataFrame, site: str) -> pd.DataFrame:
    df = weather.copy()
    t = pd.to_datetime(df["time"], utc=True)
    hour = t.dt.hour.to_numpy()
    dow = t.dt.dayofweek.to_numpy()
    temp = df["temperature_2m"].astype(float).interpolate(limit_direction="both").to_numpy()
    rh = df["relative_humidity_2m"].astype(float).interpolate(limit_direction="both").to_numpy()
    dew_series = df["dew_point_2m"].astype(float).interpolate(limit_direction="both")
    dew_series = dew_series.fillna(pd.Series(temp - (100 - rh) / 5.0, index=df.index))
    dew = dew_series.to_numpy()
    precip = df["precipitation"].astype(float).fillna(0).to_numpy()
    snow = df["snowfall"].astype(float).fillna(0).to_numpy()
    wind = df["wind_speed_10m"].astype(float).interpolate(limit_direction="both").fillna(0).to_numpy()
    gust = df["wind_gusts_10m"].astype(float).interpolate(limit_direction="both").fillna(0).to_numpy()
    solar = df["shortwave_radiation"].astype(float).fillna(0).to_numpy()
    p = _site_parameters(site)

    daily = 0.08 * np.sin(2 * np.pi * (hour - 10) / 24) + 0.035 * np.sin(4 * np.pi * (hour - 8) / 24)
    weekday = np.where(dow < 5, 0.035, -0.02)
    seasonal = 0.025 * np.sin(2 * np.pi * (t.dt.dayofyear.to_numpy() - 15) / 365.25)
    it_fraction = np.clip(0.69 + daily + weekday + seasonal, 0.52, 0.91)
    it_load = p["it_capacity_mw"] * it_fraction

    wet_bulb_proxy = temp - (100 - rh) / 5.0
    cooling_multiplier = 0.16 + 0.0105 * np.maximum(temp - 16, 0) + 0.0020 * np.maximum(wet_bulb_proxy - 18, 0)
    cooling_demand = it_load * cooling_multiplier
    capacity_derate = np.clip(1 - 0.018 * np.maximum(temp - 32, 0) - 0.0015 * np.maximum(rh - 85, 0), 0.52, 1.0)
    available_cooling = p["cooling_capacity_mw"] * capacity_derate
    cooling_headroom = (available_cooling - cooling_demand) / available_cooling
    overload = np.maximum(cooling_demand - available_cooling, 0) / np.maximum(available_cooling, 1e-6)
    inlet_temp = 22.0 + 0.055 * np.maximum(temp - 24, 0) + 15.0 * overload
    pue = 1 + cooling_demand / np.maximum(it_load, 1e-6) + 0.03

    heat_extreme = np.maximum(temp - 35, 0) / 12
    cold_extreme = np.maximum(-10 - temp, 0) / 18
    weather_stress = 0.10 * np.minimum(1, precip / 8) + 0.10 * np.minimum(1, gust / 80) + 0.10 * np.minimum(1, snow / 3)
    demand_stress = 0.10 * np.maximum(it_fraction - 0.75, 0) / 0.16
    grid_stability = np.clip(0.97 - weather_stress - demand_stress - 0.18 * heat_extreme - 0.20 * cold_extreme, 0.35, 1.0)

    soc = np.empty(len(df), dtype=np.float32)
    soc[0] = 0.72
    for i in range(1, len(df)):
        solar_charge = 0.018 * min(1.0, solar[i] / 500)
        reserve_discharge = 0.028 if grid_stability[i] < 0.78 else 0.003
        night_recharge = 0.007 if 1 <= hour[i] <= 5 and grid_stability[i] > 0.9 else 0
        soc[i] = np.clip(soc[i - 1] + solar_charge + night_recharge - reserve_discharge, 0.08, 0.98)

    thermal_component = np.maximum((inlet_temp - 24.0) / 10.0, 0)
    headroom_component = np.maximum((0.25 - cooling_headroom) / 0.28, 0)
    grid_component = np.maximum((0.90 - grid_stability) / 0.42, 0)
    reserve_component = np.where(grid_stability < 0.78, np.maximum((0.30 - soc) / 0.30, 0), 0)
    risk_score = np.maximum.reduce([thermal_component, headroom_component, grid_component, reserve_component])
    current_class = np.select(
        [risk_score < 0.08, risk_score < 0.36, risk_score < 0.72],
        [0, 1, 2],
        default=3,
    ).astype(np.int8)

    df["timestamp"] = t
    df["temperature_2m"] = temp.astype(np.float32)
    df["relative_humidity_2m"] = rh.astype(np.float32)
    df["dew_point_2m"] = dew.astype(np.float32)
    df["precipitation"] = precip.astype(np.float32)
    df["snowfall"] = snow.astype(np.float32)
    df["wind_speed_10m"] = wind.astype(np.float32)
    df["wind_gusts_10m"] = gust.astype(np.float32)
    df["shortwave_radiation"] = solar.astype(np.float32)
    df["it_load_mw"] = it_load.astype(np.float32)
    df["cooling_demand_mw"] = cooling_demand.astype(np.float32)
    df["cooling_headroom"] = cooling_headroom.astype(np.float32)
    df["inlet_temp_c"] = inlet_temp.astype(np.float32)
    df["pue"] = pue.astype(np.float32)
    df["grid_stability"] = grid_stability.astype(np.float32)
    df["battery_soc"] = soc
    df["risk_score"] = risk_score.astype(np.float32)
    df["current_risk_class"] = current_class
    # Proactive target: most severe class in t+1 ... t+6. The current hour is excluded.
    future = pd.Series(current_class[::-1]).rolling(6, min_periods=6).max().shift(1).to_numpy()[::-1]
    df["target_class_6h"] = future
    return df


SEQUENCE_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "snowfall",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "it_load_mw",
    "cooling_demand_mw",
    "cooling_headroom",
    "inlet_temp_c",
    "pue",
    "grid_stability",
    "battery_soc",
]


def _assign_split(site: str, timestamp: pd.Timestamp, policy: SplitPolicy) -> str | None:
    if site == EXTERNAL_SITE:
        if pd.Timestamp(policy.test_start) <= timestamp <= pd.Timestamp(policy.test_end):
            return "external_test"
        return None
    if site not in DEVELOPMENT_SITES:
        return None
    if timestamp <= pd.Timestamp(policy.train_end):
        return "train"
    if pd.Timestamp(policy.validation_start) <= timestamp <= pd.Timestamp(policy.validation_end):
        return "validation"
    if pd.Timestamp(policy.test_start) <= timestamp <= pd.Timestamp(policy.test_end):
        return "test"
    return None


def build_processed_dataset(raw_dir: Path, processed_dir: Path, manifest_path: Path) -> pd.DataFrame:
    processed_dir.mkdir(parents=True, exist_ok=True)
    policy = SplitPolicy()
    all_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    for site in SITES:
        raw = pd.read_csv(raw_dir / f"{site}_era5_land.csv")
        sim = simulate_facility(raw, site)
        sim = sim.dropna(subset=["target_class_6h"]).reset_index(drop=True)
        sim["site"] = site
        sim["row_in_site"] = np.arange(len(sim), dtype=np.int32)
        split = [_assign_split(site, ts, policy) for ts in sim["timestamp"]]
        sim["split"] = split
        # Enforce complete 24-hour history inside the same split. This creates temporal gaps.
        valid = sim["split"].notna()
        for lag in range(1, policy.history_hours):
            valid &= sim["split"].eq(sim["split"].shift(lag))
        sim = sim.loc[valid].copy()
        keep_cols = ["site", "timestamp", "row_in_site", "split", "target_class_6h"] + SEQUENCE_FEATURES + ["risk_score", "current_risk_class"]
        sim = sim[keep_cols]
        out = processed_dir / f"{site}.parquet"
        try:
            sim.to_parquet(out, index=False)
        except ImportError:
            out = processed_dir / f"{site}.csv.gz"
            sim.to_csv(out, index=False, compression="gzip")
        all_frames.append(sim)
        for row in sim[["site", "timestamp", "row_in_site", "split", "target_class_6h"]].itertuples(index=False):
            sample_key = f"{row.site}|{row.timestamp.isoformat()}"
            manifest_rows.append(
                {
                    "sample_id": hashlib.sha1(sample_key.encode("utf-8")).hexdigest()[:16],
                    "site": row.site,
                    "timestamp_utc": row.timestamp.isoformat(),
                    "row_in_site": int(row.row_in_site),
                    "split": row.split,
                    "target_class_6h": int(row.target_class_6h),
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    if manifest["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample identifiers detected")
    manifest.to_csv(manifest_path, index=False)
    summary = manifest.groupby(["split", "target_class_6h"]).size().rename("samples").reset_index()
    summary.to_csv(processed_dir / "split_summary.csv", index=False)
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw" / "weather"
    processed_dir = root / "data" / "processed"
    fetch_weather(raw_dir)
    summary = build_processed_dataset(raw_dir, processed_dir, root / "DATA_SPLIT_MANIFEST.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
