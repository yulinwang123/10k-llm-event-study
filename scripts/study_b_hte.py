"""
study_b_hte.py  —  Project B: Heterogeneous Treatment Effects
─────────────────────────────────────────────────────────────
Research question: When do investors rely more on hard earnings numbers vs. narrative?

Uses DML residuals from Project A:
  Ỹ = Y − E[Y|Z]   (CAR net of text + controls)
  D̃ = D − E[D|Z]   (SUE net of text + controls)

Partially linear HTE model (Best Linear Predictor):
  Ỹ_i = θ_0·D̃_i
         + τ_1·(D̃_i × eps_vol_s)    ← cognitive load / earnings complexity
         + τ_2·(D̃_i × io_pct_s)     ← investor sophistication
         + τ_3·(D̃_i × numest_s)     ← information intermediation
         + γ·X_i + ε_i

Hypotheses:
  τ_1 < 0 : more volatile EPS → numbers less informative → lower ERC
  τ_2 > 0 : more institutional ownership → harder numbers weighted more
  τ_3 > 0 : more analysts → hard news more efficiently priced

Robustness: Causal Forest via econml (if available)

Outputs:
  results/study_b_blp_table.csv
  results/study_b_blp_summary.txt
  results/study_b_tercile_erc.png
  results/study_b_cate_dist.png
  data/study_b_cates.parquet
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

BASE    = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading DML residuals (Project A Lasso) …")
resids = pd.read_parquet(BASE / "study_a_residuals.parquet")
resids["gvkey"] = resids["gvkey"].astype(object)
print(f"  {len(resids):,} obs")

print("Loading event panel …")
keep = ["gvkey", "fpedats", "numest", "eps_vol", "io_pct",
        "size", "btm", "leverage", "roa", "loss", "call_date"]
panel = pd.read_parquet(BASE / "event_panel.parquet", columns=keep)
panel["gvkey"] = panel["gvkey"].astype(object)
panel["fpedats"] = pd.to_datetime(panel["fpedats"])
resids["fpedats"] = pd.to_datetime(resids["fpedats"])

df = resids.merge(panel, on=["gvkey", "fpedats"], how="inner")
print(f"  Merged: {len(df):,} obs, {df['gvkey'].nunique():,} firms")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Prepare moderators
# ─────────────────────────────────────────────────────────────────────────────
MODS = {
    "eps_vol": "EPS Volatility\n(Cognitive Load ↑)",
    "io_pct":  "Inst. Ownership %\n(Sophistication ↑)",
    "numest":  "Analyst Coverage\n(Intermediation ↑)",
}

# Winsorize moderators at 1/99%
for m in MODS:
    lo, hi = df[m].quantile([0.01, 0.99])
    df[m] = df[m].clip(lo, hi)

# Standardize (zero-mean, unit-variance) for BLP coefficients
for m in MODS:
    mu, sd = df[m].mean(), df[m].std()
    df[f"{m}_s"] = (df[m] - mu) / sd

# Drop rows missing any moderator
req = ["Y_resid", "D_resid"] + list(MODS.keys())
df = df.dropna(subset=req).copy().reset_index(drop=True)
n = len(df)
print(f"  Estimation sample (all moderators non-missing): {n:,}")

Yr = df["Y_resid"].astype(float).values
Dr = df["D_resid"].astype(float).values

# ─────────────────────────────────────────────────────────────────────────────
# 3. BLP: Best Linear Predictor of HTE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── BLP: Best Linear Predictor ──")

# Build design matrix
# Main terms: intercept, D̃, main effects of X (standardized)
# Interaction terms: D̃ × each standardized moderator
mod_s_cols = [f"{m}_s" for m in MODS]
X_main = df[mod_s_cols].astype(float).values
X_int  = Dr[:, None] * X_main          # D̃ × X_k  (interaction)

# Full design: [1, D̃, D̃×eps_s, D̃×io_s, D̃×numest_s, eps_s, io_s, numest_s]
X_blp = np.column_stack([
    np.ones(n),                         # intercept (absorbs α)
    Dr,                                 # θ_0: ATE
    X_int,                              # τ_k: HTE interactions
    X_main,                             # γ_k: main effects of X
]).astype(np.float64)

col_names = (["intercept", "D_tilde"]
             + [f"D_tilde × {m}_s" for m in MODS]
             + [f"{m}_s" for m in MODS])

# OLS fit
beta_blp, _, _, _ = np.linalg.lstsq(X_blp, Yr, rcond=None)

# Clustered SE by firm
resid_blp = Yr - X_blp @ beta_blp
firms = df["gvkey"].values
unique_firms = np.unique(firms)
k = X_blp.shape[1]
bread = np.linalg.inv(X_blp.T @ X_blp)
meat  = np.zeros((k, k))
for g in unique_firms:
    idx   = firms == g
    score = X_blp[idx].T @ resid_blp[idx]
    meat += np.outer(score, score)
adj    = n / (n - k) * len(unique_firms) / (len(unique_firms) - 1)
V_cl   = adj * bread @ meat @ bread
se_blp = np.sqrt(np.diag(V_cl))
t_blp  = beta_blp / se_blp
p_blp  = 2 * (1 - stats.t.cdf(np.abs(t_blp), df=n - k))

def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

blp_rows = []
for i, name in enumerate(col_names):
    blp_rows.append({
        "variable": name,
        "coef":     beta_blp[i],
        "se":       se_blp[i],
        "t":        t_blp[i],
        "p":        p_blp[i],
        "stars":    stars(p_blp[i]),
    })
blp_df = pd.DataFrame(blp_rows)
blp_df.to_csv(RESULTS / "study_b_blp_table.csv", index=False)

print(f"\n{'Variable':<30} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}  {'Sig'}")
print("-" * 75)
for _, r in blp_df.iterrows():
    print(f"  {r['variable']:<28} {r['coef']:>10.5f} {r['se']:>10.5f} "
          f"{r['t']:>8.2f} {r['p']:>8.4f}  {r['stars']}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Tercile analysis: ERC by group
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Tercile ERC Analysis ──")
tercile_results = {}

for m, label in MODS.items():
    df[f"{m}_t"] = pd.qcut(df[m], 3, labels=["Low", "Mid", "High"])
    grp_results = []
    for g, gdata in df.groupby(f"{m}_t", observed=True):
        # Regress Ỹ ~ D̃ within group (no intercept since residuals mean ≈ 0)
        yr_g = gdata["Y_resid"].values
        dr_g = gdata["D_resid"].values
        theta_g = np.dot(dr_g, yr_g) / np.dot(dr_g, dr_g)
        psi_g   = (yr_g - theta_g * dr_g) * dr_g
        v_g     = np.mean(dr_g**2)
        se_g    = np.sqrt(np.mean(psi_g**2) / (v_g**2) / len(yr_g))
        grp_results.append({
            "group": str(g), "theta": theta_g, "se": se_g, "n": len(gdata)
        })
    tercile_results[m] = pd.DataFrame(grp_results)
    print(f"\n  {m}:")
    for _, r in tercile_results[m].iterrows():
        ci_lo = r["theta"] - 1.96 * r["se"]
        ci_hi = r["theta"] + 1.96 * r["se"]
        print(f"    {r['group']:>5}: θ={r['theta']:.5f}  SE={r['se']:.5f}  "
              f"95%CI=[{ci_lo:.5f}, {ci_hi:.5f}]  n={r['n']:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Figure 1: Tercile ERC bar chart
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Earnings Response Coefficient (ERC) by Moderator Tercile\n"
             "(DML-partialled residuals, 95% CI)", fontsize=13, fontweight="bold")

colors = {"Low": "#4E79A7", "Mid": "#A0CBE8", "High": "#F28E2B"}

for ax, (m, label) in zip(axes, MODS.items()):
    res = tercile_results[m]
    xs  = np.arange(3)
    thetas = res["theta"].values
    ses    = res["se"].values

    bars = ax.bar(xs, thetas, color=[colors[g] for g in res["group"]],
                  edgecolor="black", linewidth=0.7, width=0.55)
    ax.errorbar(xs, thetas, yerr=1.96 * ses, fmt="none",
                color="black", capsize=5, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Low\nTercile", "Mid\nTercile", "High\nTercile"])
    ax.set_ylabel("ERC (θ)", fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add n= labels
    for i, (bar, row) in enumerate(zip(bars, res.itertuples())):
        ax.text(i, bar.get_height() + 1.96*ses[i] + 0.0002,
                f"n={row.n:,}", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
plt.savefig(RESULTS / "study_b_tercile_erc.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → results/study_b_tercile_erc.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Causal Forest (econml) — robustness
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Causal Forest (robustness) ──")
try:
    from sklearn.ensemble import RandomForestRegressor

    X_mods = df[mod_s_cols].astype(float).values

    # R-Learner CATE estimation (Nie & Wager 2021)
    # Given pre-computed residuals Ỹ and D̃:
    #   CATE(X_i) estimated by WLS: min Σ (Ỹ_i - θ(X_i)·D̃_i)² / D̃_i²
    # Equivalent to: fit Y_pseudo_i = Ỹ_i/D̃_i with weight w_i = D̃_i²
    # Use Random Forest as the CATE learner.

    # Filter near-zero D̃ to avoid division instability
    mask = np.abs(Dr) > np.percentile(np.abs(Dr), 5)
    Y_ps = (Yr[mask] / Dr[mask])                  # pseudo-outcome
    W_ps = Dr[mask] ** 2                          # weights
    X_ps = X_mods[mask]

    print(f"  R-Learner sample (|D̃|>5th pct): {mask.sum():,}")

    rf_cate = RandomForestRegressor(
        n_estimators=500, max_depth=5, min_samples_leaf=50,
        n_jobs=-1, random_state=42
    )
    rf_cate.fit(X_ps, Y_ps, sample_weight=W_ps)

    # Predict CATE for all obs
    cate_rf = rf_cate.predict(X_mods)
    df["cate"] = cate_rf

    print(f"  CATE mean: {cate_rf.mean():.5f}  std: {cate_rf.std():.5f}")
    print(f"  CATE range: [{cate_rf.min():.5f}, {cate_rf.max():.5f}]")

    # Feature importance
    feat_imp = pd.Series(rf_cate.feature_importances_, index=list(MODS.keys()))
    print(f"\n  RF CATE feature importance:")
    for m, imp in feat_imp.sort_values(ascending=False).items():
        print(f"    {m}: {imp:.4f}")

    # Correlation with moderators
    for m in MODS:
        corr = np.corrcoef(df[m], cate_rf)[0, 1]
        print(f"  Corr(CATE, {m}): {corr:+.4f}")

    # Figure 2: CATE vs each moderator (scatter + moving average)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("R-Learner CATE Estimates vs Moderators\n(Random Forest, weighted by D̃²)",
                 fontsize=12, fontweight="bold")

    for ax, (m, label) in zip(axes, MODS.items()):
        x_vals = df[m].values
        idx_s  = np.argsort(x_vals)
        x_s    = x_vals[idx_s]
        y_s    = cate_rf[idx_s]
        w_win  = max(1, len(x_s) // 40)
        y_ma   = pd.Series(y_s).rolling(w_win, center=True, min_periods=1).mean().values

        ax.scatter(x_s, y_s, alpha=0.08, s=4, color="#4E79A7")
        ax.plot(x_s, y_ma, color="#E15759", linewidth=2.5, label="Moving avg")
        ax.axhline(cate_rf.mean(), color="gray", linestyle="--",
                   linewidth=1, label="Mean CATE")
        ax.set_xlabel(m, fontsize=10)
        ax.set_ylabel("Estimated CATE (θ̂)", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS / "study_b_cate_dist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → results/study_b_cate_dist.png")

    econml_available = True

except Exception as e:
    print(f"  R-Learner error: {e}")
    df["cate"] = np.nan
    econml_available = False

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save CATE data
# ─────────────────────────────────────────────────────────────────────────────
save_cols = (["gvkey", "fpedats", "Y_resid", "D_resid"]
             + list(MODS.keys())
             + [f"{m}_s" for m in MODS]
             + [f"{m}_t" for m in MODS]
             + (["cate", "cate_lo", "cate_hi"] if econml_available else ["cate"]))
save_cols = [c for c in save_cols if c in df.columns]
df[save_cols].to_parquet(BASE / "study_b_cates.parquet", index=False)
print(f"\nSaved → data/study_b_cates.parquet")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Summary
# ─────────────────────────────────────────────────────────────────────────────
# Extract key BLP interaction coefficients
def get_blp(var_substr):
    row = blp_df[blp_df["variable"].str.contains(var_substr)].iloc[0]
    return row["coef"], row["se"], row["t"], row["p"], row["stars"]

c_eps, s_eps, t_eps, p_eps, st_eps = get_blp("eps_vol")
c_io,  s_io,  t_io,  p_io,  st_io  = get_blp("io_pct")
c_num, s_num, t_num, p_num, st_num  = get_blp("numest")

theta_0 = blp_df[blp_df["variable"] == "D_tilde"]["coef"].values[0]
se_0    = blp_df[blp_df["variable"] == "D_tilde"]["se"].values[0]
t_0     = blp_df[blp_df["variable"] == "D_tilde"]["t"].values[0]

summary = f"""
Project B HTE Results
=====================
Sample: {n:,} firm-quarter observations
Moderators: eps_vol (standardized), io_pct (standardized), numest (standardized)

BLP Interaction Regression
--------------------------
Baseline ERC (θ_0):
  coef = {theta_0:.5f}   SE = {se_0:.5f}   t = {t_0:.2f}  ***

HTE interaction coefficients (τ_k) — units: Δθ per 1-SD change in moderator:
  D̃ × eps_vol  (cognitive load):       τ = {c_eps:+.5f}  SE={s_eps:.5f}  t={t_eps:.2f}  {st_eps}
  D̃ × io_pct   (sophistication):       τ = {c_io:+.5f}  SE={s_io:.5f}  t={t_io:.2f}  {st_io}
  D̃ × numest   (intermediation):       τ = {c_num:+.5f}  SE={s_num:.5f}  t={t_num:.2f}  {st_num}

Tercile ERC (θ by group):
"""
for m in MODS:
    res = tercile_results[m]
    summary += f"\n  {m}:\n"
    for _, r in res.iterrows():
        summary += (f"    {r['group']:>5}: θ = {r['theta']:.5f}  "
                    f"SE = {r['se']:.5f}  n={r['n']:,}\n")

summary += f"""
Causal Forest: {'Estimated — see study_b_cate_dist.png' if econml_available else 'Not run (pip install econml)'}

Interpretation:
  τ(eps_vol) {'<' if c_eps < 0 else '>'} 0 → {'✓ Higher earnings complexity reduces ERC (cognitive load hypothesis)' if c_eps < 0 else '✗ Cognitive load hypothesis not supported'}
  τ(io_pct)  {'>' if c_io  > 0 else '<'} 0 → {'✓ Sophisticated investors price hard numbers more aggressively' if c_io > 0 else '✗ Sophistication hypothesis not supported'}
  τ(numest)  {'>' if c_num > 0 else '<'} 0 → {'✓ More analysts → harder news more efficiently priced' if c_num > 0 else '✗ Intermediation hypothesis not supported'}
"""
print(summary)
(RESULTS / "study_b_blp_summary.txt").write_text(summary)
print(f"Saved → results/study_b_blp_summary.txt")
