from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resilient_dc.modeling import DEFAULT_CONFIGS, FeatureBundle, ensure_bundles, run_candidate


def subset(bundle: FeatureBundle, indices: np.ndarray) -> FeatureBundle:
    return FeatureBundle(
        sequence=bundle.sequence[indices],
        physics=bundle.physics[indices],
        tabular=bundle.tabular[indices],
        labels=bundle.labels[indices],
        sites=bundle.sites[indices],
        timestamps=bundle.timestamps[indices],
    )


def stratified_subset(bundle: FeatureBundle, per_class: int, seed: int) -> FeatureBundle:
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in range(4):
        candidates = np.flatnonzero(bundle.labels == cls)
        chosen.extend(rng.choice(candidates, min(per_class, len(candidates)), replace=False).tolist())
    chosen = np.asarray(chosen)
    rng.shuffle(chosen)
    return subset(bundle, chosen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "screen"], required=True)
    args = parser.parse_args()
    train, validation = ensure_bundles(ROOT, include_test=False)
    candidates = ["PBT", "PG_GRU", "PTT", "PC_TCN"]
    output_dir = ROOT / "results" / "model_selection"
    model_dir = ROOT / "models" / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.stage == "smoke":
        train_use = stratified_subset(train, per_class=5000, seed=17)
        val_use = stratified_subset(validation, per_class=1000, seed=18)
        seeds = [17]
    else:
        train_use, val_use = train, validation
        seeds = [17, 29, 43]
    for candidate in candidates:
        for seed in seeds:
            config = dict(DEFAULT_CONFIGS[candidate])
            if args.stage == "smoke":
                if candidate == "PBT":
                    config["n_estimators"] = 120
                else:
                    config["epochs"] = 3
                    config["patience"] = 2
            row = run_candidate(candidate, train_use, val_use, seed, config, model_dir / args.stage)
            rows.append(row)
            print(json.dumps(row, indent=2))
            pd.DataFrame(rows).to_csv(output_dir / f"{args.stage}_seed_results.csv", index=False)
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.columns if c not in {"candidate", "seed"}]
    aggregate = frame.groupby("candidate")[numeric].agg(["mean", "std"])
    aggregate.columns = ["_".join(col).rstrip("_") for col in aggregate.columns]
    aggregate.reset_index().to_csv(output_dir / f"{args.stage}_aggregate.csv", index=False)


if __name__ == "__main__":
    main()
