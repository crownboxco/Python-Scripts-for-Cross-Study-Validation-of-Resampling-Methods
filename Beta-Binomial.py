# This script is for Beta-Binomial modelling.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
from matplotlib.gridspec import GridSpec

INPUT_FILE = "Summary_Beta.xlsx" #<-- Here enter your summary results file.
df = pd.read_excel(INPUT_FILE)

PRIOR_LABEL = "Beta(0.5,0.5)"
MIN_POST_PROB_HELP = 0.80
MIN_PARSIMONY_RATE = 0.50

SELECTION_COL = "Selection"   # must contain values like "1SE" and "High"

required_cols = {"Study", "Method", "High_Low_NS", "Baseline_Comp", SELECTION_COL}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["Method", "High_Low_NS", "Baseline_Comp", SELECTION_COL]).copy()

# Clean whitespace
for c in ["Method", "High_Low_NS", "Baseline_Comp", SELECTION_COL]:
    df[c] = df[c].astype(str).str.strip()

# Optional sanity checks
valid_macro = {"High", "Low", "NS"}
valid_baseline = {"High", "Low", "NS"}
valid_selection = {"1SE", "High"}

bad_macro = set(df["High_Low_NS"].unique()) - valid_macro
bad_base = set(df["Baseline_Comp"].unique()) - valid_baseline
bad_sel = set(df[SELECTION_COL].unique()) - valid_selection

if bad_macro:
    raise ValueError(f"Unexpected values found in High_Low_NS: {bad_macro}")
if bad_base:
    raise ValueError(f"Unexpected values found in Baseline_Comp: {bad_base}")
if bad_sel:
    raise ValueError(f"Unexpected values found in {SELECTION_COL}: {bad_sel}")

method_order = df["Method"].drop_duplicates().tolist()

# Helper: Bayesian posterior for Bernoulli outcomes
def bayes_bernoulli(k_success, n_total, a=0.5, b=0.5):
    """
    Returns posterior mean, 95% credible interval, and P(p > 0.5).
    """
    a_post = a + k_success
    b_post = b + (n_total - k_success)
    post_mean = a_post / (a_post + b_post)
    ci_low, ci_high = beta.ppf([0.025, 0.975], a_post, b_post)
    prob_gt_half = 1 - beta.cdf(0.5, a_post, b_post)
    return post_mean, ci_low, ci_high, prob_gt_half

# Summary builder
def build_summary(data, method_order, prior_label=PRIOR_LABEL):
    rows = []

    for method, g in data.groupby("Method"):
        # A. Macro F1 non-worse posterior
        macro_nonworse = int(g["High_Low_NS"].isin(["High", "NS"]).sum())
        macro_total = len(g)

        macro_mean, macro_low, macro_high, macro_prob = bayes_bernoulli(
            macro_nonworse, macro_total
        )

        # B. Conditional parsimony posterior
        g_macro_subset = g[g["High_Low_NS"].isin(["High", "NS"])]

        if len(g_macro_subset) > 0:
            pars_success = int(g_macro_subset["Baseline_Comp"].isin(["Low", "NS"]).sum())
            pars_total = len(g_macro_subset)

            pars_mean, pars_low, pars_high, pars_prob = bayes_bernoulli(
                pars_success, pars_total
            )
        else:
            pars_success = 0
            pars_total = 0
            pars_mean = np.nan
            pars_low = np.nan
            pars_high = np.nan
            pars_prob = np.nan

        rows.append({
            "Method": method,
            "N Studies": macro_total,

            "MacroF1_NonWorse": macro_nonworse,
            f"PosteriorMean_MacroF1 ({prior_label})": macro_mean,
            f"MacroF1_CI_Low ({prior_label})": macro_low,
            f"MacroF1_CI_High ({prior_label})": macro_high,
            f"PostProb_MacroF1>0.5 ({prior_label})": macro_prob,

            "Pars_NonWorseLV": pars_success,
            "Pars_Total": pars_total,
            f"PosteriorMean_ParsGivenMacroF1 ({prior_label})": pars_mean,
            f"Pars_CI_Low ({prior_label})": pars_low,
            f"Pars_CI_High ({prior_label})": pars_high,
            f"PostProb_Pars>0.5_GivenMacroF1 ({prior_label})": pars_prob,
        })

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    summary["__ord"] = summary["Method"].apply(lambda m: method_order.index(m))
    summary = summary.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)
    return summary


# Plotting
def plot_posterior_panel(ax, summary, title, prior_label=PRIOR_LABEL, show_legend=True):
    if summary.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    methods = summary["Method"].tolist()
    x = np.arange(len(methods))
    palette = sns.color_palette()

    # Macro F1 posterior mean + 95% CrI
    macro_mean = summary[f"PosteriorMean_MacroF1 ({prior_label})"].values
    macro_low = summary[f"MacroF1_CI_Low ({prior_label})"].values
    macro_high = summary[f"MacroF1_CI_High ({prior_label})"].values

    macro_mean_pct = macro_mean * 100
    macro_err_low = (macro_mean - macro_low) * 100
    macro_err_high = (macro_high - macro_mean) * 100

    ax.bar(
        x,
        macro_mean_pct,
        color=palette[2],
        edgecolor="black",
        label="P(Macro F1 Non-Worse)",
        zorder=2
    )

    ax.errorbar(
        x,
        macro_mean_pct,
        yerr=np.vstack([macro_err_low, macro_err_high]),
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        zorder=4
    )

    # Parsimony posterior mean + 95% CrI
    pars_mean = summary[f"PosteriorMean_ParsGivenMacroF1 ({prior_label})"].values
    pars_low = summary[f"Pars_CI_Low ({prior_label})"].values
    pars_high = summary[f"Pars_CI_High ({prior_label})"].values

    pars_mean_pct = pars_mean * 100
    pars_err_low = (pars_mean - pars_low) * 100
    pars_err_high = (pars_high - pars_mean) * 100

    ax.plot(
        x,
        pars_mean_pct,
        marker="o",
        linestyle="--",
        linewidth=1.8,
        color="red",
        label="P(Parsimonious | Macro F1 Non-Worse)",
        zorder=5
    )

    valid = ~np.isnan(pars_mean_pct)
    if valid.any():
        ax.errorbar(
            x[valid],
            pars_mean_pct[valid],
            yerr=np.vstack([pars_err_low[valid], pars_err_high[valid]]),
            fmt="none",
            ecolor="red",
            elinewidth=1.2,
            capsize=3,
            zorder=6
        )

    ax.axhline(50, color="red", linestyle=":", linewidth=1.2)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Posterior Mean Probability (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 105)

    if show_legend:
        leg = ax.legend(frameon=True, fontsize=8, loc="lower right")
        leg.get_frame().set_alpha(0.8)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.75)


def plot_proportion_panel(ax, data, method_order, title, show_legend):
    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    palette = sns.color_palette()
    methods = [m for m in method_order if m in data["Method"].unique()]

    cat_pct = (
        data.groupby(["Method", "High_Low_NS"])
            .size()
            .unstack(fill_value=0)
    )

    for c in ["Low", "NS", "High"]:
        if c not in cat_pct.columns:
            cat_pct[c] = 0

    cat_pct = cat_pct.div(cat_pct.sum(axis=1), axis=0) * 100
    cat_pct = cat_pct.reindex(methods).fillna(0)

    x = np.arange(len(methods))

    worse_vals = cat_pct["Low"].values
    unchanged_vals = cat_pct["NS"].values
    improved_vals = cat_pct["High"].values

    ax.bar(
        x, worse_vals,
        label="Macro F1 Decreased",
        color=palette[3], edgecolor="black"
    )
    ax.bar(
        x, unchanged_vals,
        bottom=worse_vals,
        label="No Change",
        color=palette[4], edgecolor="black"
    )
    ax.bar(
        x, improved_vals,
        bottom=worse_vals + unchanged_vals,
        label="Macro F1 Improved",
        color=palette[2], edgecolor="black"
    )

    lv_reduced = (data["Baseline_Comp"] == "Low").groupby(data["Method"]).mean() * 100
    lv_reduced = lv_reduced.reindex(methods).fillna(0)

    ax.plot(
        x, lv_reduced.values,
        marker="o", linestyle="--", linewidth=1.8,
        color="black", label="Reduced LVs (%)"
    )

    lv_no_change = (data["Baseline_Comp"] == "NS").groupby(data["Method"]).mean() * 100
    lv_no_change = lv_no_change.reindex(methods).fillna(0)

    ax.plot(
        x, lv_no_change.values,
        marker="s", linestyle=":", linewidth=1.8,
        color="black", label="No Change LVs (%)"
    )

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Percent of Studies (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 105)

    if show_legend:
        leg = ax.legend(frameon=True, fontsize=8, loc="lower right")
        leg.get_frame().set_alpha(0.8)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.75)

# Build summaries
df_1se = df[df[SELECTION_COL] == "1SE"].copy()
df_high = df[df[SELECTION_COL] == "High"].copy()

summary_all = build_summary(df, method_order, PRIOR_LABEL)
summary_1se = build_summary(df_1se, method_order, PRIOR_LABEL)
summary_high = build_summary(df_high, method_order, PRIOR_LABEL)

# Print Bayesian summaries
print("\n=== Bayesian Summary: COMBINED ===")
print(summary_all.to_string(index=False))

print("\n=== Bayesian Summary: 1SE ===")
print(summary_1se.to_string(index=False))

print("\n=== Bayesian Summary: High ===")
print(summary_high.to_string(index=False))


# ===========================================================
# FIGURE 1:
# Bayesian posterior panels
# Top-left  : 1SE
# Top-right : High
# Bottom    : Combined (1SE + High)
# ===========================================================
fig1 = plt.figure(figsize=(10, 7))
gs1 = GridSpec(2, 2, height_ratios=[1, 1.35], hspace=0.42, wspace=0.22)

ax1 = fig1.add_subplot(gs1[0, 0])
ax2 = fig1.add_subplot(gs1[0, 1])
ax3 = fig1.add_subplot(gs1[1, :])

plot_posterior_panel(ax1, summary_1se, title="1SE Selection", show_legend=False)
plot_posterior_panel(ax2, summary_high, title="HMF1 Selection", show_legend=False)
plot_posterior_panel(ax3, summary_all, title="Combined 1SE + HMF1 Selections", show_legend=True)

plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.show()

# ===========================================================
# FIGURE 2:
# Empirical proportions
# Top-left  : 1SE
# Top-right : High
# Bottom    : Combined (1SE + High)
# ===========================================================
fig2 = plt.figure(figsize=(10, 7))
gs2 = GridSpec(2, 2, height_ratios=[1, 1.35], hspace=0.42, wspace=0.22)

bx1 = fig2.add_subplot(gs2[0, 0])
bx2 = fig2.add_subplot(gs2[0, 1])
bx3 = fig2.add_subplot(gs2[1, :])

plot_proportion_panel(bx1, df_1se, method_order, title="1SE Selection", show_legend=False)
plot_proportion_panel(bx2, df_high, method_order, title="HMF1 Selection", show_legend=False)
plot_proportion_panel(bx3, df, method_order, title="Combined 1SE + HMF1 Selections", show_legend=True)

plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.show()

# ===========================================================
# Save Excel Report
# ===========================================================
output_file = "Phase1_Bayesian_Summary_Report.xlsx"

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    summary_all.to_excel(writer, index=False, sheet_name="Summary_Combined")
    summary_1se.to_excel(writer, index=False, sheet_name="Summary_1SE")
    summary_high.to_excel(writer, index=False, sheet_name="Summary_High")
    df.to_excel(writer, index=False, sheet_name="Input_Data")

print(f"\nSummary report saved to: {output_file}\n")