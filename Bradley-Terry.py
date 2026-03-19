# This script was used for Bradley-Terry modelling.
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy.stats import beta

INPUT_FILE = "Summary_Results.xlsx" #<-- Here enter your summary results.
df = pd.read_excel(INPUT_FILE)
OUTPUT_FILE = "BradleyTerry_Results.xlsx"

# Required columns
STUDY_COL = "Study"
METHOD_COL = "Method"
BIAS_COL = "SqBias_MacroF1"

# How to collapse repeated rows within each Study × Method
# Choose "mean" or "median"
AGG_FUNC = "mean"

# Output figure files
FIG1_FILE = "BradleyTerry_SkillRanking.png"
FIG2_FILE = "BradleyTerry_PBest.png"

# 95% CrI limits
ALPHA = 0.05
LOW_Q = 100 * (ALPHA / 2)       # 2.5
HIGH_Q = 100 * (1 - ALPHA / 2)  # 97.5

# Tie tolerance for study-level comparisons
TIE_ATOL = 1e-15
TIE_RTOL = 0.0

def summarize_study_method(df, study_col, method_col, bias_col, agg_func="mean"):
    if agg_func not in {"mean", "median"}:
        raise ValueError("AGG_FUNC must be 'mean' or 'median'.")

    grouped = (
        df.groupby([study_col, method_col], as_index=False)[bias_col]
        .agg(agg_func)
        .rename(columns={bias_col: "BiasValue"})
    )
    return grouped

required_cols = {STUDY_COL, METHOD_COL, BIAS_COL}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}. "
        f"Columns found = {df.columns.tolist()}"
    )

df = df[[STUDY_COL, METHOD_COL, BIAS_COL]].copy()
df[STUDY_COL] = df[STUDY_COL].astype(str).str.strip()
df[METHOD_COL] = df[METHOD_COL].astype(str).str.strip()
df["BiasValue"] = pd.to_numeric(df[BIAS_COL], errors="coerce")

df = df.loc[(df[STUDY_COL] != "") & (df[METHOD_COL] != "")].copy()
df = df.dropna(subset=["BiasValue"]).reset_index(drop=True)

if df.empty:
    raise ValueError("No valid rows remain after filtering.")

# Collapse repeated rows within Study × Method
study_method = summarize_study_method(
    df=df,
    study_col=STUDY_COL,
    method_col=METHOD_COL,
    bias_col="BiasValue",
    agg_func=AGG_FUNC
)

if study_method.empty:
    raise ValueError("No Study × Method summaries could be created.")

print(f"Collapsed repeated rows using AGG_FUNC = '{AGG_FUNC}'")
print(f"Original rows: {len(df)}")
print(f"Study × Method rows after collapse: {len(study_method)}")

# Extract methods and indexing
methods = sorted(study_method[METHOD_COL].unique())
M = len(methods)

if M < 2:
    raise ValueError("Need at least two unique methods to fit a Bradley–Terry model.")

method_idx = {m: i for i, m in enumerate(methods)}

# Build study-level pairwise comparisons
# For each study:
#   compare every pair of methods present in that study once
#   lower BiasValue = win
# Then aggregate across studies into Binomial counts.
pair_win_counts = {}   # key = (i, j), value = wins for i over j
pair_trial_counts = {} # key = (i, j), value = number of non-tied studies

studies = sorted(study_method[STUDY_COL].unique())
study_pair_rows = []

for study in studies:
    sub = study_method.loc[study_method[STUDY_COL] == study].copy()

    # Map method -> summarized bias for this study
    method_to_bias = dict(zip(sub[METHOD_COL], sub["BiasValue"]))

    study_methods = sorted(method_to_bias.keys())
    if len(study_methods) < 2:
        continue

    for m1, m2 in itertools.combinations(study_methods, 2):
        b1 = method_to_bias[m1]
        b2 = method_to_bias[m2]

        if np.isclose(b1, b2, atol=TIE_ATOL, rtol=TIE_RTOL):
            continue

        i = method_idx[m1]
        j = method_idx[m2]

        # orient as (i, j) where win counts mean "i beats j"
        key = (i, j)

        pair_trial_counts[key] = pair_trial_counts.get(key, 0) + 1
        pair_win_counts[key] = pair_win_counts.get(key, 0) + int(b1 < b2)

        study_pair_rows.append({
            "Study": study,
            "Method_1": m1,
            "Method_2": m2,
            "Bias_1": b1,
            "Bias_2": b2,
            "Winner": m1 if b1 < b2 else m2
        })

if len(pair_trial_counts) == 0:
    raise ValueError("No non-tied study-level pairwise comparisons could be constructed.")

pair_i = []
pair_j = []
n_trials = []
n_wins = []

for (i, j), total_ij in sorted(pair_trial_counts.items()):
    wins_ij = pair_win_counts[(i, j)]
    pair_i.append(i)
    pair_j.append(j)
    n_trials.append(total_ij)
    n_wins.append(wins_ij)

pair_i = np.array(pair_i, dtype=int)
pair_j = np.array(pair_j, dtype=int)
n_trials = np.array(n_trials, dtype=int)
n_wins = np.array(n_wins, dtype=int)

print(f"Number of methods: {M}")
print(f"Methods: {methods}")
print(f"Number of study-level method-pair rows used: {len(n_trials)}")
print(f"Total non-tied study-level contests represented: {int(n_trials.sum())}")

# Bayesian Bradley–Terry Model in PyMC
if __name__ == "__main__":

    with pm.Model() as bt_model:

        # Sum-to-zero identifiable skill parameters
        lambda_raw = pm.Normal("lambda_raw", mu=0, sigma=1, shape=M)
        lambda_ = pm.Deterministic("lambda", lambda_raw - pm.math.mean(lambda_raw))

        # Pairwise win probability
        p = pm.Deterministic(
            "p",
            pm.math.sigmoid(lambda_[pair_i] - lambda_[pair_j])
        )

        # Study-level aggregated Binomial observations
        y = pm.Binomial("y", n=n_trials, p=p, observed=n_wins)

        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=1,
            target_accept=0.99,
            init="jitter+adapt_diag",
            random_seed=1818
        )

    # Posterior skill table
    post = az.summary(trace, var_names=["lambda"])
    post["Method"] = methods

    lambda_samples = trace.posterior["lambda"].stack(samples=("chain", "draw")).values

    # Probability each method is best
    is_best = (lambda_samples == lambda_samples.max(axis=0)).astype(int)
    best_counts = is_best.sum(axis=1)
    n_post = is_best.shape[1]

    post["P_Best"] = (best_counts + 0.5) / (n_post + 1)

    lambda_low = np.percentile(lambda_samples, LOW_Q, axis=1)
    lambda_high = np.percentile(lambda_samples, HIGH_Q, axis=1)
    post["lambda_low"] = lambda_low
    post["lambda_high"] = lambda_high

    # 95% intervals for P(Best) using Jeffreys-style Beta posterior
    post["P_Best_low"] = beta.ppf(ALPHA / 2, best_counts + 0.5, n_post - best_counts + 0.5)
    post["P_Best_high"] = beta.ppf(1 - ALPHA / 2, best_counts + 0.5, n_post - best_counts + 0.5)

    # Save results to Excel
    interval_cols = [c for c in post.columns if c.startswith("hdi_")]
    if len(interval_cols) < 2:
        raise ValueError(
            f"Expected two HDI columns from ArviZ summary, got: {post.columns.tolist()}"
        )

    post_out = post[
        [
            "Method",
            "mean",
            interval_cols[0],
            interval_cols[1],
            "lambda_low",
            "lambda_high",
            "P_Best",
            "P_Best_low",
            "P_Best_high",
        ]
    ].copy()
    post_out = post_out.sort_values("mean", ascending=False).reset_index(drop=True)

    # Pairwise probability matrix
    pair_probs = pd.DataFrame(index=methods, columns=methods, dtype=float)

    for i, m in enumerate(methods):
        for j, mp in enumerate(methods):
            if i == j:
                pair_probs.loc[m, mp] = np.nan
            else:
                pair_probs.loc[m, mp] = (lambda_samples[i, :] > lambda_samples[j, :]).mean()

    # Study-level descriptives using collapsed Study × Method table
    descriptives = (
        study_method.groupby(METHOD_COL)["BiasValue"]
        .agg(n_studies="count", mean_sqbias="mean", median_sqbias="median", sd_sqbias="std")
        .reset_index()
        .rename(columns={METHOD_COL: "Method"})
        .sort_values("mean_sqbias", ascending=True)
        .reset_index(drop=True)
    )

    study_pair_df = pd.DataFrame(study_pair_rows)

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        post_out.to_excel(writer, index=False, sheet_name="BT_Skills")
        pair_probs.to_excel(writer, sheet_name="Pairwise_Probabilities")
        descriptives.to_excel(writer, index=False, sheet_name="Method_Descriptives")
        study_method.to_excel(writer, index=False, sheet_name="Study_Method_Summary")
        study_pair_df.to_excel(writer, index=False, sheet_name="Study_Level_Contests")

    print(f"Bradley–Terry results saved to: {OUTPUT_FILE}")

    # Plot 1 — Skill parameter ranking with 95% CrIs
    plot1_df = post_out.sort_values("mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(plot1_df))

    ax.errorbar(
        x=plot1_df["mean"].to_numpy(),
        y=y_pos,
        xerr=np.vstack([
            plot1_df["mean"].to_numpy() - plot1_df["lambda_low"].to_numpy(),
            plot1_df["lambda_high"].to_numpy() - plot1_df["mean"].to_numpy()
        ]),
        fmt="o",
        capsize=4,
        elinewidth=1.2,
        markersize=6,
        ecolor="black"
    )

    ax.axvline(0, color="red", linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot1_df["Method"])
    ax.invert_yaxis()
    ax.set_xlabel("Skill Parameter λ (Higher = Better)", fontsize=10)
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG1_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 2 — Posterior probability of being best with 95% CrIs
    plot2_df = post_out.sort_values("P_Best", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    x_pos = np.arange(len(plot2_df))
    bar_heights = plot2_df["P_Best"].to_numpy()

    ax.bar(x_pos, bar_heights, edgecolor="black", linewidth=1.2)

    ax.errorbar(
        x=x_pos,
        y=bar_heights,
        yerr=np.vstack([
            bar_heights - plot2_df["P_Best_low"].to_numpy(),
            plot2_df["P_Best_high"].to_numpy() - bar_heights
        ]),
        fmt="none",
        capsize=4,
        elinewidth=1.2,
        ecolor="black"
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot2_df["Method"], rotation=45, ha="right")
    ax.set_ylabel("Posterior P(Least Squared-Biased Macro F1)", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIG2_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure: {FIG1_FILE}")
    print(f"Saved figure: {FIG2_FILE}")