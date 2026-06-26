"""
study_a_dml.py  —  Project A: DML Partialling-out
───────────────────────────────────────────────────
Model:
  Y = D·θ + g(Z) + ε       (partially linear)
  D = m(Z) + v

  Y = car_m1_p1_win   (CAR[-1,+1])
  D = sue_win          (SUE, standardized unexpected earnings)
  Z = text features + firm controls  (confounders to partial out)

Estimator:
  1. K-fold cross-fitting (K=5)
  2. Nuisance models: Lasso (default) + Random Forest (robustness)
  3. Final step: OLS of Ỹ on D̃ (residuals)
  4. Semiparametric SE (influence-function based)

Also runs:
  - OLS benchmark (without partialling out) for bias comparison
  - Sensitivity: θ across different nuisance model choices

Output:
  results/study_a_dml_results.csv
  results/study_a_dml_summary.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model   import LassoCV, RidgeCV
from sklearn.ensemble       import RandomForestRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline       import Pipeline

BASE    = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

np.random.seed(42)

# ── Load panel ────────────────────────────────────────────────────────────────
print("Loading event panel …")
panel = pd.read_parquet(BASE / "event_panel.parquet")

txt_path = BASE / "text_features.parquet"
if txt_path.exists() and "prep_lm_tone" not in panel.columns:
    txt = pd.read_parquet(txt_path, columns=[
        "transcriptid",
        "prep_lm_tone", "prep_lm_uncertainty", "prep_lm_negative",
        "prep_fog_index", "prep_word_count",
        "qa_lm_tone",   "qa_lm_uncertainty",
        "qa_fog_index",  "qa_word_count",
    ])
    panel = panel.merge(txt, on="transcriptid", how="left")

# ── Variable definitions ──────────────────────────────────────────────────────
Y_col = "car_m1_p1_win"
D_col = "sue_win"

FIRM_CONTROLS = ["size", "btm", "leverage", "roa", "loss", "eps_vol"]
TEXT_VARS     = [
    "prep_lm_tone", "prep_lm_uncertainty", "prep_lm_negative", "prep_fog_index",
    "qa_lm_tone",   "qa_lm_uncertainty",   "qa_fog_index",
]

# Log word counts
panel["log_prep_wc"] = np.log1p(panel.get("prep_word_count", 0).fillna(0))
panel["log_qa_wc"]   = np.log1p(panel.get("qa_word_count",   0).fillna(0))
TEXT_VARS += ["log_prep_wc", "log_qa_wc"]

# Time dummies (year-quarter)
panel["ym"] = pd.to_datetime(panel["call_date"]).dt.to_period("Q").astype(str)
time_dummies = pd.get_dummies(panel["ym"], prefix="t", drop_first=True)

Z_COLS = FIRM_CONTROLS + [v for v in TEXT_VARS if v in panel.columns]

# ── Build estimation sample ───────────────────────────────────────────────────
req = [Y_col, D_col] + Z_COLS
sample = panel.dropna(subset=req).copy().reset_index(drop=True)
print(f"Estimation sample: {len(sample):,} obs, {sample['gvkey'].nunique():,} firms")
print(f"Z dimension: {len(Z_COLS)} variables (+ time dummies)")

# Build Z matrix (controls + text + time dummies) — force float64
Z_raw  = sample[Z_COLS].astype(float).values
t_dum  = pd.get_dummies(sample["ym"], prefix="t", drop_first=True).astype(float).values
Z_full = np.hstack([Z_raw, t_dum]).astype(np.float64)

Y = sample[Y_col].astype(float).values
D = sample[D_col].astype(float).values

print(f"Z matrix shape: {Z_full.shape}")

# ── Core DML function ─────────────────────────────────────────────────────────
def dml_crossfit(Y, D, Z, n_folds=5, model="lasso"):
    """
    DML partialling-out with K-fold cross-fitting.
    Returns theta, se, t, p, Y_resid, D_resid
    """
    n = len(Y)
    Y_resid = np.zeros(n)
    D_resid = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(Z)):
        Z_tr, Z_te = Z[train_idx], Z[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]

        if model == "lasso":
            pipe_Y = Pipeline([("scaler", StandardScaler()),
                               ("lasso", LassoCV(cv=5, n_jobs=-1, max_iter=5000))])
            pipe_D = Pipeline([("scaler", StandardScaler()),
                               ("lasso", LassoCV(cv=5, n_jobs=-1, max_iter=5000))])
        elif model == "ridge":
            pipe_Y = Pipeline([("scaler", StandardScaler()),
                               ("ridge", RidgeCV(cv=5))])
            pipe_D = Pipeline([("scaler", StandardScaler()),
                               ("ridge", RidgeCV(cv=5))])
        elif model == "rf":
            pipe_Y = RandomForestRegressor(n_estimators=200, max_depth=6,
                                           n_jobs=-1, random_state=42)
            pipe_D = RandomForestRegressor(n_estimators=200, max_depth=6,
                                           n_jobs=-1, random_state=42)

        pipe_Y.fit(Z_tr, Y_tr)
        pipe_D.fit(Z_tr, D_tr)

        Y_resid[test_idx] = Y_te - pipe_Y.predict(Z_te)
        D_resid[test_idx] = D_te - pipe_D.predict(Z_te)

        print(f"    Fold {fold+1}/{n_folds} done")

    # Final OLS: Y_resid ~ D_resid (no intercept, partialling-out already centering)
    theta = np.dot(D_resid, Y_resid) / np.dot(D_resid, D_resid)

    # Influence-function SE
    psi   = (Y_resid - theta * D_resid) * D_resid
    v_hat = np.mean(D_resid**2)
    se    = np.sqrt(np.mean(psi**2) / (v_hat**2) / n)

    t_val = theta / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_val), df=n - Z.shape[1] - 1))

    return theta, se, t_val, p_val, Y_resid, D_resid

# ── OLS benchmark (naive, no partialling) ────────────────────────────────────
print("\n── OLS benchmark (no DML) ──")
from numpy.linalg import lstsq
X_ols = np.column_stack([np.ones(len(D)), D, Z_full])
beta_ols, _, _, _ = lstsq(X_ols, Y, rcond=None)
theta_ols = beta_ols[1]
resid_ols = Y - X_ols @ beta_ols
n, k = X_ols.shape
sigma2 = resid_ols @ resid_ols / (n - k)
# Clustered SE by firm
firms = sample["gvkey"].values
unique_firms = np.unique(firms)
meat = np.zeros((k, k))
for g in unique_firms:
    idx = firms == g
    score = X_ols[idx].T @ resid_ols[idx]
    meat += np.outer(score, score)
bread = np.linalg.inv(X_ols.T @ X_ols)
V_cl = n / (n-k) * len(unique_firms) / (len(unique_firms)-1) * bread @ meat @ bread
se_ols = np.sqrt(np.diag(V_cl)[1])
t_ols  = theta_ols / se_ols
p_ols  = 2 * (1 - stats.t.cdf(abs(t_ols), df=n-k))
print(f"  θ_OLS = {theta_ols:.4f}  SE = {se_ols:.4f}  t = {t_ols:.2f}  p = {p_ols:.4f}")

# ── DML: Lasso ────────────────────────────────────────────────────────────────
print("\n── DML with Lasso nuisance (K=5) ──")
theta_lasso, se_lasso, t_lasso, p_lasso, Yr_lasso, Dr_lasso = \
    dml_crossfit(Y, D, Z_full, n_folds=5, model="lasso")
print(f"  θ_DML(Lasso) = {theta_lasso:.4f}  SE = {se_lasso:.4f}  "
      f"t = {t_lasso:.2f}  p = {p_lasso:.4f}")

# ── DML: Ridge (robustness) ───────────────────────────────────────────────────
print("\n── DML with Ridge nuisance (K=5) ──")
theta_ridge, se_ridge, t_ridge, p_ridge, _, _ = \
    dml_crossfit(Y, D, Z_full, n_folds=5, model="ridge")
print(f"  θ_DML(Ridge) = {theta_ridge:.4f}  SE = {se_ridge:.4f}  "
      f"t = {t_ridge:.2f}  p = {p_ridge:.4f}")

# ── DML: Random Forest (robustness) ──────────────────────────────────────────
print("\n── DML with Random Forest nuisance (K=5) ──")
theta_rf, se_rf, t_rf, p_rf, _, _ = \
    dml_crossfit(Y, D, Z_full, n_folds=5, model="rf")
print(f"  θ_DML(RF) = {theta_rf:.4f}  SE = {se_rf:.4f}  "
      f"t = {t_rf:.2f}  p = {t_rf:.4f}")

# ── Results table ─────────────────────────────────────────────────────────────
def stars(p):
    return "***" if p<.01 else "**" if p<.05 else "*" if p<.10 else ""

results = pd.DataFrame([
    {"Model": "OLS (no partial-out)", "Nuisance": "—",
     "θ": theta_ols, "SE": se_ols, "t": t_ols, "p": p_ols,
     "Significance": stars(p_ols), "N": len(Y)},
    {"Model": "DML Partialling-out", "Nuisance": "Lasso",
     "θ": theta_lasso, "SE": se_lasso, "t": t_lasso, "p": p_lasso,
     "Significance": stars(p_lasso), "N": len(Y)},
    {"Model": "DML Partialling-out", "Nuisance": "Ridge",
     "θ": theta_ridge, "SE": se_ridge, "t": t_ridge, "p": p_ridge,
     "Significance": stars(p_ridge), "N": len(Y)},
    {"Model": "DML Partialling-out", "Nuisance": "Random Forest",
     "θ": theta_rf, "SE": se_rf, "t": t_rf, "p": p_rf,
     "Significance": stars(p_rf), "N": len(Y)},
])

out_csv = RESULTS / "study_a_dml_results.csv"
results.to_csv(out_csv, index=False)
print(f"\nSaved → {out_csv}")
print("\n" + results[["Model","Nuisance","θ","SE","t","p","Significance"]].to_string(index=False))

# ── Bias comparison ───────────────────────────────────────────────────────────
bias = theta_ols - theta_lasso
summary = f"""
Project A DML Results
=====================
Sample: {len(Y):,} firm-quarter observations
Z dimension: {Z_full.shape[1]} (controls + text + time FE)

                     θ (ERC)    SE       t       p
OLS (no partial):   {theta_ols:.4f}    {se_ols:.4f}   {t_ols:.2f}   {p_ols:.4f}
DML Lasso:          {theta_lasso:.4f}    {se_lasso:.4f}   {t_lasso:.2f}   {p_lasso:.4f}
DML Ridge:          {theta_ridge:.4f}    {se_ridge:.4f}   {t_ridge:.2f}   {p_ridge:.4f}
DML RF:             {theta_rf:.4f}    {se_rf:.4f}   {t_rf:.2f}   {p_rf:.4f}

Narrative bias in OLS: {bias:+.4f}
  (OLS overstates/understates ERC by this amount due to text confounding)

Interpretation:
  θ_DML is the marginal effect of hard earnings news (SUE) on CAR
  after fully partialling out managerial narrative (text features Z).
  Significance of θ_DML confirms the hard news channel is independent
  of narrative framing — the basis for Project B.
"""
print(summary)
(RESULTS / "study_a_dml_summary.txt").write_text(summary)

# Save residuals for Study B
residuals_df = sample[["gvkey","permno","transcriptid","call_date","fpedats"]].copy()
residuals_df["Y_resid"] = Yr_lasso
residuals_df["D_resid"] = Dr_lasso
residuals_df.to_parquet(BASE / "study_a_residuals.parquet", index=False)
print(f"\nSaved residuals → data/study_a_residuals.parquet  (for Study B HTE)")
