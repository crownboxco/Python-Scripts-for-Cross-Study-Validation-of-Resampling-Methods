# This script was used to aggregate results from Squared Bias Wilcoxon tests.
import pandas as pd
import numpy as np

FILE = "SqBias_Method_Rankings_by_LV_FenNails.csv"
df = pd.read_csv(FILE)

# Check required columns
required_cols = {"Strategy", "LV", "mean", "median", "SE", "FinalRank"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {FILE}: {missing}")

# Ensure numeric columns are numeric
for col in ["LV", "mean", "median", "SE", "FinalRank"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Strategy", "LV", "mean", "median", "SE", "FinalRank"]).copy()

# Summarize by strategy over all LVs
rows = []

for strategy, sub in df.groupby("Strategy"):
    sub = sub.sort_values("LV").copy()

    n_lv = len(sub)

    # ---- squared bias summaries across LV-level summaries ----
    # Uses the per-LV "mean" squared bias and per-LV "median" squared bias columns already in the file
    sqbias_mean_mean = sub["mean"].mean()
    sqbias_mean_median = sub["mean"].median()
    sqbias_mean_se = sub["mean"].std(ddof=1) / np.sqrt(n_lv) if n_lv > 1 else np.nan

    sqbias_median_mean = sub["median"].mean()
    sqbias_median_median = sub["median"].median()
    sqbias_median_se = sub["median"].std(ddof=1) / np.sqrt(n_lv) if n_lv > 1 else np.nan

    # ---- final rank summaries across LVs ----
    rank_mean = sub["FinalRank"].mean()
    rank_median = sub["FinalRank"].median()
    rank_se = sub["FinalRank"].std(ddof=1) / np.sqrt(n_lv) if n_lv > 1 else np.nan

    rows.append({
        "Strategy": strategy,
        "n_LVs": n_lv,

        # Based on per-LV MEAN squared bias
        "SqBias_mean_across_LVs": sqbias_mean_mean,
        "SqBias_median_of_means_across_LVs": sqbias_mean_median,
        "SqBias_SE_of_means_across_LVs": sqbias_mean_se,

        # Based on per-LV MEDIAN squared bias
        "SqBias_mean_of_medians_across_LVs": sqbias_median_mean,
        "SqBias_median_across_LVs": sqbias_median_median,
        "SqBias_SE_of_medians_across_LVs": sqbias_median_se,

        # Final rank across LVs
        "FinalRank_mean": rank_mean,
        "FinalRank_median": rank_median,
        "FinalRank_SE": rank_se,
    })

summary = pd.DataFrame(rows)

# Sort by average final rank (lower is better), then median squared bias
summary = summary.sort_values(
    by=["FinalRank_mean", "SqBias_median_across_LVs"],
    ascending=[True, True]
).reset_index(drop=True)

# Print results
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

print("\n" + "=" * 120)
print("SUMMARY BY STRATEGY OVER ALL LVs")
print("=" * 120)
print(summary.to_string(index=False))

print("\n" + "=" * 120)
print("COMPACT REPORT")
print("=" * 120)

for _, r in summary.iterrows():
    print(
        f"{r['Strategy']}: "
        f"n_LVs={int(r['n_LVs'])} | "
        f"SqBias median={r['SqBias_median_across_LVs']:.6f}, "
        f"mean={r['SqBias_mean_across_LVs']:.6f}, "
        f"SE={r['SqBias_SE_of_means_across_LVs']:.6f} | "
        f"FinalRank median={r['FinalRank_median']:.6f}, "
        f"mean={r['FinalRank_mean']:.6f}, "
        f"SE={r['FinalRank_SE']:.6f}"
    )