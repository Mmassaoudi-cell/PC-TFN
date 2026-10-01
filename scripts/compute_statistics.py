"""Paired statistical comparison of the final model against every benchmark.

Reads `results/final_evaluation/seed_level_results.csv`. For each split and each
benchmark, runs a paired Wilcoxon signed-rank test (seed-matched) on macro-F1,
applies Holm-Bonferroni correction across all benchmarks within a split, and
classifies each comparison as statistically superior / practically superior /
statistically indistinguishable / inferior. Writes BENCHMARK_WTL.csv.

Note: temperature scaling is a monotonic rescaling of probabilities, so it never
changes the argmax prediction; raw_macro_f1 and cal_macro_f1 are therefore
identical for every method, and this script uses raw_macro_f1 (any label-based
metric would give the same comparison) as the primary comparison metric.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "final_evaluation"
OUT_DIR = ROOT / "results" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRACTICAL_MARGIN = 0.01  # macro-F1 points; below this and non-significant => indistinguishable


def paired_effect_size(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation."""
    n = len(diff)
    if n == 0 or np.allclose(diff, 0):
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    return float((pos - neg) / (n * (n + 1) / 2))


def classify(p_corrected: float, mean_diff: float, alpha: float = 0.05) -> str:
    if p_corrected < alpha:
        return "statistically superior" if mean_diff > 0 else "statistically inferior"
    if abs(mean_diff) >= PRACTICAL_MARGIN:
        return "practically superior" if mean_diff > 0 else "practically inferior"
    return "statistically indistinguishable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="methods to drop from the benchmark set (e.g. a superseded architecture)")
    args = parser.parse_args()

    seed_results = pd.read_csv(RESULTS_DIR / "seed_level_results.csv")
    if args.exclude:
        seed_results = seed_results[~seed_results["method"].isin(args.exclude)]
    import yaml
    final_name = yaml.safe_load((ROOT / "FINAL_MODEL_CONFIG.yaml").read_text())["final_model"]["name"]

    all_rows = []
    for split in seed_results["split"].unique():
        split_df = seed_results[seed_results["split"] == split]
        final_df = split_df[split_df["method"] == final_name].set_index("seed")["raw_macro_f1"]
        benchmarks = [m for m in split_df["method"].unique() if m != final_name]

        raw_pvalues = []
        rows = []
        for bench in benchmarks:
            bench_df = split_df[split_df["method"] == bench].set_index("seed")["raw_macro_f1"]
            common_seeds = sorted(set(final_df.index) & set(bench_df.index))
            final_vals = final_df.loc[common_seeds].to_numpy()
            bench_vals = bench_df.loc[common_seeds].to_numpy()
            diff = final_vals - bench_vals
            mean_diff = float(diff.mean())
            if np.allclose(diff, 0):
                p_value = 1.0
            else:
                try:
                    _, p_value = stats.wilcoxon(final_vals, bench_vals)
                except ValueError:
                    p_value = 1.0
            effect = paired_effect_size(diff)
            raw_pvalues.append(p_value)
            rows.append({
                "split": split, "benchmark": bench, "final_model": final_name,
                "final_mean_macro_f1": float(final_vals.mean()), "benchmark_mean_macro_f1": float(bench_vals.mean()),
                "mean_diff": mean_diff, "wilcoxon_p_raw": p_value, "effect_size_rank_biserial": effect,
                "n_seeds": len(common_seeds),
            })
        if raw_pvalues:
            reject, p_corrected, _, _ = multipletests(raw_pvalues, alpha=0.05, method="holm")
            for row, p_c, rej in zip(rows, p_corrected, reject):
                row["wilcoxon_p_holm"] = float(p_c)
                row["verdict"] = classify(p_c, row["mean_diff"])
        all_rows.extend(rows)

    stats_df = pd.DataFrame(all_rows)
    stats_df.to_csv(OUT_DIR / "pairwise_significance.csv", index=False)

    wtl_rows = []
    for split in stats_df["split"].unique():
        split_stats = stats_df[stats_df["split"] == split]
        wins = (split_stats["verdict"].isin(["statistically superior", "practically superior"])).sum()
        ties = (split_stats["verdict"] == "statistically indistinguishable").sum()
        losses = (split_stats["verdict"].isin(["statistically inferior", "practically inferior"])).sum()
        wtl_rows.append({"split": split, "wins": int(wins), "ties": int(ties), "losses": int(losses), "n_benchmarks": len(split_stats)})
        for _, r in split_stats.iterrows():
            wtl_rows.append({
                "split": split, "benchmark": r["benchmark"], "verdict": r["verdict"],
                "final_mean_macro_f1": r["final_mean_macro_f1"], "benchmark_mean_macro_f1": r["benchmark_mean_macro_f1"],
                "wilcoxon_p_holm": r["wilcoxon_p_holm"], "effect_size_rank_biserial": r["effect_size_rank_biserial"],
            })
    pd.DataFrame(wtl_rows).to_csv(ROOT / "BENCHMARK_WTL.csv", index=False)
    print(stats_df.groupby("split")["verdict"].value_counts())


if __name__ == "__main__":
    main()
