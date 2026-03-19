# This is the main script for 1SE modelling and squared bias calculations.
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from imblearn.over_sampling import RandomOverSampler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# -------------------- Config --------------------
DATA_FILE = "Preprocessed_Dataset.csv" #<-- Here enter the name of your preprocessed dataset.
SAMPLE_COL = "Sample"
ID_COL = "ID"

SEED = 1818
N_OUTER_SPLITS = 10
TEST_SIZE = 0.5
LV_GRID = list(range(2, 21))
BASELINE_LV = 1 #<-- Here enter the baseline LV model for your preprocessed dataset.

APPLY_OVERSAMPLE = True
INCLUDE_LOOCV = True

KFOLDS = [3, 5, 10]
VB_SPLITS = [3, 5, 10]
BOOT_B = [100, 1000]
JK_SPLITS = [100, 1000]

OUT_DETAILED_CSV = "PLSDA_SqBias_Detailed_MacroF1_Dataset.csv"
OUT_SUMMARY_CSV = "PLSDA_SqBias_Summary_MacroF1_Dataset.csv"
OUT_1SE_CSV = "PLSDA_1SE_SelectedLV_MacroF1_Dataset.csv"


# -------------------- Utilities --------------------
def se(x):
    """Standard error."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def progress(tag, i, total):
    """Print ~10% progress."""
    step = max(1, int(np.ceil(total / 10)))
    if i == 1 or i == total or i % step == 0:
        print(f"[{tag}] {i}/{total} ({100 * i / total:.0f}%)")


def can_do_stratified_kfold(y, k):
    """Check class support for K-fold."""
    return np.min(np.unique(y, return_counts=True)[1]) >= k


def wilcoxon_lr(x, y):
    """One-sided paired Wilcoxon p-values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    d = x[m] - y[m]
    n = len(d)

    if n < 2:
        return np.nan, np.nan, n, np.median(d) if n else np.nan
    if np.allclose(d, 0):
        return np.nan, np.nan, n, float(np.median(d))

    return (
        float(wilcoxon(d, alternative="greater").pvalue),
        float(wilcoxon(d, alternative="less").pvalue),
        n,
        float(np.median(d)),
    )


# -------------------- PLS-DA helpers --------------------
def fit_predict_pls(X_train, y_train, X_eval, n_comp, onehot):
    """Fit PLS-DA and return class scores."""
    Y_train = onehot.transform(y_train.reshape(-1, 1))
    model = PLSRegression(n_components=int(n_comp))
    model.fit(X_train, Y_train)
    return model.predict(X_eval)


def aggregate_unit_scores(y_true, y_scores, units, n_classes):
    """Average spectrum scores by sample-unit."""
    df_tmp = pd.DataFrame({"unit": units, "y": y_true})
    for k in range(n_classes):
        df_tmp[f"s{k}"] = y_scores[:, k]

    score_cols = [f"s{k}" for k in range(n_classes)]
    agg = df_tmp.groupby("unit", as_index=False).agg(
        {"y": "first", **{c: "mean" for c in score_cols}}
    )
    return agg["y"].to_numpy(int), agg[score_cols].to_numpy(float)


def sample_macro_f1(y_true, y_scores, units, n_classes):
    """Sample-level Macro F1."""
    y_u, s_u = aggregate_unit_scores(y_true, y_scores, units, n_classes)
    y_pred = np.argmax(s_u, axis=1)
    return float(
        f1_score(
            y_u,
            y_pred,
            average="macro",
            labels=np.arange(n_classes),
            zero_division=0,
        )
    )


def choose_lv_1se(inner_scores):
    """Pick smallest LV within 1 SE of best mean validation Macro F1."""
    stats = {}
    for lv, vals in inner_scores.items():
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        stats[int(lv)] = (mean, sd, len(vals))

    if not stats:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    best_lv = min(stats, key=lambda k: (-stats[k][0], k))
    best_mean, best_sd, k_best = stats[best_lv]
    best_se = best_sd / np.sqrt(k_best) if k_best > 1 else 0.0
    threshold = best_mean - best_se
    chosen_lv = min(lv for lv, (m, _, _) in stats.items() if m >= threshold)

    return chosen_lv, best_lv, best_mean, best_sd, best_se, threshold


# -------------------- Split generators --------------------
def venetian_blinds_splits(n, s):
    """Ordered venetian blinds splits."""
    idx = np.arange(n)
    fold = idx % s
    return [(idx[fold != i], idx[fold == i]) for i in range(s)]


def stratified_bootstrap_splits(y, B, seed):
    """Stratified bootstrap with OOB validation."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx_all = np.arange(len(y))
    classes = np.unique(y)
    out = []

    while len(out) < B:
        tr = np.concatenate([
            rng.choice(np.where(y == c)[0], size=np.sum(y == c), replace=True)
            for c in classes
        ])
        va = np.setdiff1d(idx_all, np.unique(tr))
        if len(va):
            out.append((tr, va))

    return out


def stratified_jackknife_splits(y, S, seed, n_classes):
    """Stratified delete-p jackknife."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    idx_all = np.arange(n)

    classes, counts = np.unique(y, return_counts=True)
    p = max(n_classes, n // S)
    out = []

    while len(out) < S:
        p_c = np.floor(p * counts / n).astype(int)
        deficit = p - p_c.sum()
        for i in np.argsort(-counts):
            if deficit <= 0:
                break
            p_c[i] += 1
            deficit -= 1
        p_c = np.minimum(p_c, counts)
        while p_c.sum() > p:
            p_c[np.argmax(p_c)] -= 1

        va = np.unique(np.concatenate([
            rng.choice(np.where(y == c)[0], size=int(m), replace=False)
            for c, m in zip(classes, p_c) if m > 0
        ]))
        tr = np.setdiff1d(idx_all, va)

        if len(tr) and len(va):
            out.append((tr, va))

    return out, int(p)


# -------------------- Training-set builder --------------------
def build_train_spectra(
    tr_unit_idx,
    unit_ids,
    y_units,
    unit_to_rows,
    X_spec,
    y_spec,
    seed,
    oversample=True,
):
    """Map unit indices to spectrum rows; optionally oversample units."""
    tr_unit_idx = np.asarray(tr_unit_idx, dtype=int)
    units = unit_ids[tr_unit_idx]
    y_train_units = y_units[tr_unit_idx]
    row_lists = [unit_to_rows[u] for u in units]

    if not oversample:
        rows = np.concatenate(row_lists)
        return X_spec[rows], y_spec[rows]

    unit_means = np.vstack([X_spec[r].mean(axis=0) for r in row_lists])
    ros = RandomOverSampler(random_state=seed)
    ros.fit_resample(unit_means, y_train_units)
    rows = np.concatenate([row_lists[i] for i in ros.sample_indices_])
    return X_spec[rows], y_spec[rows]


# -------------------- Load data --------------------
df = pd.read_csv(DATA_FILE)

required = {SAMPLE_COL, ID_COL}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

feature_cols = [c for c in df.columns if c not in [SAMPLE_COL, ID_COL]]
if not feature_cols:
    raise ValueError("No feature columns found.")

df["SampleUnit"] = df[SAMPLE_COL].astype(str) + "||" + df[ID_COL].astype(str)

X_spec = df[feature_cols].to_numpy(float)
y_spec_str = df[ID_COL].astype(str).to_numpy()
sampleunit_spec = df["SampleUnit"].astype(str).to_numpy()

units_df = df[["SampleUnit", ID_COL]].drop_duplicates().reset_index(drop=True)
unit_ids_all = units_df["SampleUnit"].to_numpy(str)

le = LabelEncoder()
y_units_all = le.fit_transform(units_df[ID_COL].astype(str).to_numpy())
y_spec = le.transform(y_spec_str)
n_classes = len(le.classes_)

onehot = OneHotEncoder(sparse_output=False)
onehot.fit(y_units_all.reshape(-1, 1))

unit_to_rows = {
    u: np.asarray(rows, dtype=int)
    for u, rows in df.groupby("SampleUnit").groups.items()
}

n_units = len(unit_ids_all)

print("\n" + "=" * 80)
print("PLS-DA resampling comparison: 1SE LV selection + squared-bias analysis")
print("=" * 80)
print(f"Dataset: {DATA_FILE}")
print(f"Spectra: {X_spec.shape[0]} | Features: {X_spec.shape[1]}")
print(f"Sample-units: {n_units} | Classes: {n_classes}")
print(f"Outer splits: {N_OUTER_SPLITS} | Test size: {TEST_SIZE}")
print(f"Seed: {SEED} | Baseline LV: {BASELINE_LV}")
print(f"Oversampling: {'ON' if APPLY_OVERSAMPLE else 'OFF'}")
print(f"LV grid: {LV_GRID[0]}..{LV_GRID[-1]}")
print("=" * 80)


# -------------------- Outer resampling --------------------
sss = StratifiedShuffleSplit(
    n_splits=N_OUTER_SPLITS,
    test_size=TEST_SIZE,
    random_state=SEED,
)

detailed_rows = []
chosen_rows = []

for outer_i, (tr_idx_all, te_idx_all) in enumerate(sss.split(np.zeros(n_units), y_units_all), start=1):
    progress("OUTER 50/50", outer_i, N_OUTER_SPLITS)

    train_units = unit_ids_all[tr_idx_all]
    test_units = unit_ids_all[te_idx_all]
    y_train_units = y_units_all[tr_idx_all]
    method_seed = SEED + outer_i

    # Test set
    te_rows = np.concatenate([unit_to_rows[u] for u in test_units])
    X_te, y_te = X_spec[te_rows], y_spec[te_rows]
    su_te = sampleunit_spec[te_rows]

    # Internal methods
    methods = []

    if INCLUDE_LOOCV:
        idx = np.arange(len(train_units))
        methods.append(("LOOCV", [(np.delete(idx, i), np.array([i])) for i in idx]))

    for k in KFOLDS:
        if can_do_stratified_kfold(y_train_units, k):
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=method_seed)
            methods.append((f"KFCV (K={k})", list(skf.split(np.zeros(len(train_units)), y_train_units))))
        else:
            min_ct = int(np.min(np.unique(y_train_units, return_counts=True)[1]))
            print(f"[Skip] KFCV (K={k}) | min class count in train = {min_ct}")

    for s in VB_SPLITS:
        methods.append((f"VBCV (Blinds={s})", venetian_blinds_splits(len(train_units), s)))

    for B in BOOT_B:
        methods.append((f"BS (B={B})", stratified_bootstrap_splits(y_train_units, B, method_seed)))

    for S in JK_SPLITS:
        splits, p = stratified_jackknife_splits(y_train_units, S, method_seed, n_classes)
        methods.append((f"JK (S={S}, p={p})", splits))

    # Full-train test F1 by LV
    X_tr_full, y_tr_full = build_train_spectra(
        np.arange(len(train_units)),
        train_units,
        y_train_units,
        unit_to_rows,
        X_spec,
        y_spec,
        method_seed,
        oversample=APPLY_OVERSAMPLE,
    )

    max_lv_full = min(X_tr_full.shape[0] - 1, X_tr_full.shape[1])
    lv_grid_full = [lv for lv in LV_GRID if lv <= max_lv_full]

    test_f1_by_lv = {}
    for lv in lv_grid_full:
        y_score_te = fit_predict_pls(X_tr_full, y_tr_full, X_te, lv, onehot)
        test_f1_by_lv[lv] = sample_macro_f1(y_te, y_score_te, su_te, n_classes)

    baseline_test_f1 = float(test_f1_by_lv.get(BASELINE_LV, np.nan))

    # Inner tuning for each method
    for strategy, splits in methods:
        inner_scores = {lv: [] for lv in LV_GRID}

        for inner_j, (tr_idx, va_idx) in enumerate(splits, start=1):
            va_units = train_units[va_idx]
            va_rows = np.concatenate([unit_to_rows[u] for u in va_units])
            X_va, y_va = X_spec[va_rows], y_spec[va_rows]
            su_va = sampleunit_spec[va_rows]

            X_tr, y_tr = build_train_spectra(
                tr_idx,
                train_units,
                y_train_units,
                unit_to_rows,
                X_spec,
                y_spec,
                method_seed + inner_j,
                oversample=APPLY_OVERSAMPLE,
            )

            max_lv_inner = min(X_tr.shape[0] - 1, X_tr.shape[1])

            for lv in LV_GRID:
                if lv > max_lv_inner:
                    continue
                y_score_va = fit_predict_pls(X_tr, y_tr, X_va, lv, onehot)
                inner_scores[lv].append(sample_macro_f1(y_va, y_score_va, su_va, n_classes))

        chosen_lv, best_lv, best_mean, best_sd, best_se, threshold = choose_lv_1se(inner_scores)
        chosen_vals = np.asarray(inner_scores.get(chosen_lv, []), dtype=float)
        chosen_val_mean = float(np.mean(chosen_vals)) if len(chosen_vals) else np.nan
        chosen_test_f1 = float(test_f1_by_lv.get(chosen_lv, np.nan)) if np.isfinite(chosen_lv) else np.nan

        chosen_rows.append(
            {
                "OuterSplit": outer_i,
                "Strategy": strategy,
                "ChosenLV_1SE": chosen_lv,
                "BestMeanLV": best_lv,
                "BestMean_ValF1": best_mean,
                "BestMean_ValF1_SD": best_sd,
                "BestMean_ValF1_SE": best_se,
                "1SE_Threshold": threshold,
                "ChosenLV_ValF1_MeanAcrossInner": chosen_val_mean,
                "ChosenLV_TestF1": chosen_test_f1,
                "BaselineLV": BASELINE_LV,
                "Baseline_TestF1": baseline_test_f1,
            }
        )

        for lv in LV_GRID:
            val_mean = float(np.mean(inner_scores[lv])) if inner_scores[lv] else np.nan
            test_f1 = float(test_f1_by_lv.get(lv, np.nan))
            sq_bias = (val_mean - test_f1) ** 2 if np.isfinite(val_mean) and np.isfinite(test_f1) else np.nan

            detailed_rows.append(
                {
                    "OuterSplit": outer_i,
                    "Strategy": strategy,
                    "LV": int(lv),
                    "Val_MacroF1": val_mean,
                    "Test_MacroF1": test_f1,
                    "SqBias_MacroF1": sq_bias,
                }
            )


# -------------------- Save detailed and LV summary --------------------
detailed_df = pd.DataFrame(detailed_rows)
detailed_df.to_csv(OUT_DETAILED_CSV, index=False)

summary_df = (
    detailed_df.groupby(["Strategy", "LV"], as_index=False)
    .agg(
        k=("SqBias_MacroF1", lambda x: int(np.isfinite(np.asarray(x, dtype=float)).sum())),
        Mean_SqBias_MacroF1=("SqBias_MacroF1", "mean"),
        SE_SqBias_MacroF1=("SqBias_MacroF1", se),
        Mean_Val_MacroF1=("Val_MacroF1", "mean"),
        SE_Val_MacroF1=("Val_MacroF1", se),
        Mean_Test_MacroF1=("Test_MacroF1", "mean"),
        SE_Test_MacroF1=("Test_MacroF1", se),
    )
    .sort_values(["Strategy", "LV"])
    .reset_index(drop=True)
)
summary_df.to_csv(OUT_SUMMARY_CSV, index=False)


# -------------------- Chosen-LV summary --------------------
chosen_df = pd.DataFrame(chosen_rows)

chosen_summary = (
    chosen_df.groupby("Strategy", as_index=False)
    .agg(
        OuterSplits=("OuterSplit", "count"),
        BaselineLV=("BaselineLV", "first"),
        ChosenLV_Median=("ChosenLV_1SE", "median"),
        ChosenLV_SE=("ChosenLV_1SE", se),
        ChosenValF1_Median=("ChosenLV_ValF1_MeanAcrossInner", "median"),
        ChosenValF1_SE=("ChosenLV_ValF1_MeanAcrossInner", se),
        ChosenTestF1_Median=("ChosenLV_TestF1", "median"),
        ChosenTestF1_SE=("ChosenLV_TestF1", se),
        BaselineTestF1_Median=("Baseline_TestF1", "median"),
        BaselineTestF1_SE=("Baseline_TestF1", se),
    )
    .sort_values("Strategy")
    .reset_index(drop=True)
)

# Wilcoxon tests
test_stats = []
lv_stats = []

for strategy in chosen_summary["Strategy"]:
    sub = chosen_df.loc[chosen_df["Strategy"] == strategy].sort_values("OuterSplit")
    test_stats.append(wilcoxon_lr(sub["ChosenLV_TestF1"], sub["Baseline_TestF1"]))
    lv_stats.append(wilcoxon_lr(sub["ChosenLV_1SE"], sub["BaselineLV"]))

chosen_summary["wilcoxon_TestF1_n_pairs_used"] = [x[2] for x in test_stats]
chosen_summary["median_diff_TestF1_(chosen-baseline)"] = [x[3] for x in test_stats]
chosen_summary["wilcoxon_TestF1_p_right_(H1_chosen>baseline)"] = [x[0] for x in test_stats]
chosen_summary["wilcoxon_TestF1_p_left_(H1_chosen<baseline)"] = [x[1] for x in test_stats]

chosen_summary["wilcoxon_LV_n_pairs_used"] = [x[2] for x in lv_stats]
chosen_summary["median_diff_LV_(chosen-baseline)"] = [x[3] for x in lv_stats]
chosen_summary["wilcoxon_LV_p_right_(H1_chosenLV>baselineLV)"] = [x[0] for x in lv_stats]
chosen_summary["wilcoxon_LV_p_left_(H1_chosenLV<baselineLV)"] = [x[1] for x in lv_stats]

chosen_summary.to_csv(OUT_1SE_CSV, index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Saved detailed: {OUT_DETAILED_CSV}")
print(f"Saved summary:  {OUT_SUMMARY_CSV}")
print(f"Saved 1SE file: {OUT_1SE_CSV}")
print("\nChosen-LV summary:")
print(chosen_summary.to_string(index=False))