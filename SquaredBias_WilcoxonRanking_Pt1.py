# This script was used to rank all resampling methods using the Wilcoxon signed rank test.
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, rankdata
from itertools import combinations

FILE = "PLSDA_SqBias_Detailed_MacroF1_Dataset.csv"
df = pd.read_csv(FILE)
ALPHA = 0.05

OUT_RANKS_CSV = "SqBias_Method_Rankings_by_LV_Dataset.csv"
OUT_PAIRWISE_CSV = "SqBias_Pairwise_Wilcoxon_by_LV_Dataset.csv"

# ============================================================
# Helpers
# ============================================================
def holm_adjust(pvals):
    """
    Holm-Bonferroni adjusted p-values.
    Returns adjusted p-values in original order.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for i, p in enumerate(ranked):
        val = (m - i) * p
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)

    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


def paired_wilcoxon_table(wide_df, alpha=0.05):
    """
    wide_df: rows are paired observations (splits), columns are methods, values are sq bias.
    Lower squared bias is better.
    """
    methods = list(wide_df.columns)
    rows = []

    for a, b in combinations(methods, 2):
        pair = wide_df[[a, b]].dropna()
        x = pair[a].values
        y = pair[b].values
        d = x - y  # negative means A better than B

        # Descriptive pairing summaries
        n_pairs = len(d)
        n_nonzero = np.sum(np.abs(d) > 0)
        med_diff = np.median(d) if n_pairs > 0 else np.nan

        # If all paired differences are zero, Wilcoxon is undefined in the usual sense
        if n_nonzero == 0:
            p_two = 1.0
            p_a_better = 1.0
            p_b_better = 1.0
            stat_two = np.nan
        else:
            # Two-sided
            res_two = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox", correction=False)
            p_two = float(res_two.pvalue)
            stat_two = float(res_two.statistic)

            # One-sided: A better than B means x < y because lower sq bias is better
            res_a_better = wilcoxon(x, y, alternative="less", zero_method="wilcox", correction=False)
            p_a_better = float(res_a_better.pvalue)

            # One-sided: B better than A means x > y
            res_b_better = wilcoxon(x, y, alternative="greater", zero_method="wilcox", correction=False)
            p_b_better = float(res_b_better.pvalue)

        rows.append({
            "Method_A": a,
            "Method_B": b,
            "n_pairs": n_pairs,
            "n_nonzero_diffs": n_nonzero,
            "median_diff_A_minus_B": med_diff,
            "wilcoxon_stat_two_sided": stat_two,
            "p_two_sided": p_two,
            "p_A_better_than_B_one_sided": p_a_better,
            "p_B_better_than_A_one_sided": p_b_better,
        })

    pairwise = pd.DataFrame(rows)

    # Holm adjust separately for each family of tests
    if not pairwise.empty:
        pairwise["p_two_sided_holm"] = holm_adjust(pairwise["p_two_sided"].values)
        pairwise["p_A_better_than_B_one_sided_holm"] = holm_adjust(pairwise["p_A_better_than_B_one_sided"].values)
        pairwise["p_B_better_than_A_one_sided_holm"] = holm_adjust(pairwise["p_B_better_than_A_one_sided"].values)

        def outcome(row):
            if row["p_A_better_than_B_one_sided_holm"] < alpha:
                return "A_better"
            elif row["p_B_better_than_A_one_sided_holm"] < alpha:
                return "B_better"
            else:
                return "Tie"
        pairwise["outcome"] = pairwise.apply(outcome, axis=1)

    return pairwise


def summarize_and_rank_within_lv(sub_df, alpha=0.05):
    """
    sub_df must contain one LV only.
    Returns:
      summary table per method
      pairwise Wilcoxon table
    """
    # Wide format: rows = splits, cols = methods
    wide = sub_df.pivot_table(
        index="OuterSplit",
        columns="Strategy",
        values="SqBias_MacroF1",
        aggfunc="first"
    ).sort_index()

    # Descriptive stats per method
    long_vals = (
        wide.reset_index()
            .melt(id_vars="OuterSplit", var_name="Strategy", value_name="SqBias_MacroF1")
            .dropna()
    )

    desc = long_vals.groupby("Strategy")["SqBias_MacroF1"].agg(
        n="count",
        mean="mean",
        sd="std",
        median="median",
        q1=lambda x: np.percentile(x, 25),
        q3=lambda x: np.percentile(x, 75),
        min="min",
        max="max"
    ).reset_index()

    desc["IQR"] = desc["q3"] - desc["q1"]
    desc["SE"] = desc["sd"] / np.sqrt(desc["n"])

    # Within-split ranks (lower sq bias = better rank 1)
    rank_rows = []
    for split_id, row in wide.iterrows():
        row_nonmissing = row.dropna()
        if len(row_nonmissing) == 0:
            continue
        ranks = rankdata(row_nonmissing.values, method="average")  # ascending by default
        for method, rk, val in zip(row_nonmissing.index, ranks, row_nonmissing.values):
            rank_rows.append({
                "OuterSplit": split_id,
                "Strategy": method,
                "SqBias_MacroF1": val,
                "WithinSplitRank": rk
            })

    rank_df = pd.DataFrame(rank_rows)
    mean_ranks = rank_df.groupby("Strategy")["WithinSplitRank"].agg(
        mean_rank="mean",
        median_rank="median"
    ).reset_index()

    # Pairwise Wilcoxon
    pairwise = paired_wilcoxon_table(wide, alpha=alpha)

    # Win/loss/tie counts
    methods = sorted(wide.columns.tolist())
    wl = {m: {"wins": 0, "losses": 0, "ties": 0} for m in methods}

    for _, r in pairwise.iterrows():
        a, b, outcome = r["Method_A"], r["Method_B"], r["outcome"]
        if outcome == "A_better":
            wl[a]["wins"] += 1
            wl[b]["losses"] += 1
        elif outcome == "B_better":
            wl[b]["wins"] += 1
            wl[a]["losses"] += 1
        else:
            wl[a]["ties"] += 1
            wl[b]["ties"] += 1

    wl_df = pd.DataFrame([
        {"Strategy": m, **wl[m]} for m in methods
    ])

    # Merge summaries
    summary = (
        desc.merge(mean_ranks, on="Strategy", how="left")
            .merge(wl_df, on="Strategy", how="left")
    )

    # Publish-friendly string
    summary["median_IQR_str"] = summary.apply(
        lambda r: f"{r['median']:.6f} [{r['q1']:.6f}, {r['q3']:.6f}]",
        axis=1
    )
    summary["mean_SE_str"] = summary.apply(
        lambda r: f"{r['mean']:.6f} \u00B1 {r['SE']:.6f}",
        axis=1
    )
    summary["win_loss_tie"] = summary.apply(
        lambda r: f"{int(r['wins'])}-{int(r['losses'])}-{int(r['ties'])}",
        axis=1
    )

    # Final rank:
    # 1) more significant wins
    # 2) fewer losses
    # 3) lower mean rank
    # 4) lower median squared bias
    summary = summary.sort_values(
        by=["wins", "losses", "mean_rank", "median"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)
    summary["FinalRank"] = np.arange(1, len(summary) + 1)

    return summary, pairwise, wide, rank_df

# Main
required = {"OuterSplit", "Strategy", "LV", "SqBias_MacroF1"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Make sure types are sensible
df["OuterSplit"] = pd.to_numeric(df["OuterSplit"], errors="coerce")
df["LV"] = pd.to_numeric(df["LV"], errors="coerce")
df["SqBias_MacroF1"] = pd.to_numeric(df["SqBias_MacroF1"], errors="coerce")
df = df.dropna(subset=["OuterSplit", "LV", "Strategy", "SqBias_MacroF1"]).copy()

all_lv_summaries = []
all_pairwise = []

for lv in sorted(df["LV"].unique()):
    sub = df[df["LV"] == lv].copy()

    # Skip LVs with fewer than 2 methods or fewer than 2 paired splits
    if sub["Strategy"].nunique() < 2:
        continue

    summary, pairwise, wide, rank_df = summarize_and_rank_within_lv(sub, alpha=ALPHA)

    summary.insert(0, "LV", lv)
    pairwise.insert(0, "LV", lv)

    all_lv_summaries.append(summary)
    all_pairwise.append(pairwise)

    print("\n" + "=" * 90)
    print(f"LV = {lv}")
    print("=" * 90)
    print(summary[[
        "FinalRank", "Strategy", "n",
        "median_IQR_str", "mean_SE_str",
        "mean_rank", "win_loss_tie"
    ]].to_string(index=False))

    print("\nPairwise Wilcoxon results (Holm-adjusted):")
    if not pairwise.empty:
        print(pairwise[[
            "Method_A", "Method_B",
            "median_diff_A_minus_B",
            "p_two_sided_holm",
            "p_A_better_than_B_one_sided_holm",
            "p_B_better_than_A_one_sided_holm",
            "outcome"
        ]].to_string(index=False))
    else:
        print("No pairwise comparisons available.")

# Save outputs
if all_lv_summaries:
    ranks_out = pd.concat(all_lv_summaries, ignore_index=True)
    ranks_out.to_csv(OUT_RANKS_CSV, index=False)
    print(f"\nSaved ranked summaries to: {OUT_RANKS_CSV}")

if all_pairwise:
    pairwise_out = pd.concat(all_pairwise, ignore_index=True)
    pairwise_out.to_csv(OUT_PAIRWISE_CSV, index=False)
    print(f"Saved pairwise Wilcoxon table to: {OUT_PAIRWISE_CSV}")