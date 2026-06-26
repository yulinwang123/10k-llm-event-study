"""
study_a_ols.py  —  Project A: OLS Baseline
────────────────────────────────────────────
Research question: After controlling for managerial narrative text,
does earnings surprise (SUE) still have a marginal effect on CAR?

Four specifications (increasingly stringent controls):
  (1)  CAR ~ SUE                           [raw ERC]
  (2)  CAR ~ SUE + firm controls           [standard ERC]
  (3)  CAR ~ SUE + firm controls + text    [text-augmented ERC]
  (4)  CAR ~ SUE + firm controls + text + firm FE + time FE

Output:
  results/study_a_ols_table.csv   — coefficient table
  results/study_a_ols_stats.txt   — key stats summary
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

BASE    = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

# ── Load panel ────────────────────────────────────────────────────────────────
print("Loading event panel …")
panel = pd.read_parquet(BASE / "event_panel.parquet")
print(f"  Raw panel: {len(panel):,} obs, {panel['gvkey'].nunique():,} firms")

# Merge text features if not already in panel
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
    print(f"  Text features merged: {panel['prep_lm_tone'].notna().sum():,} obs")

# ── Variable definitions ──────────────────────────────────────────────────────
Y = "car_m1_p1_win"   # CAR[-1,+1] winsorized
D = "sue_win"         # SUE winsorized

FIRM_CONTROLS = ["size", "btm", "leverage", "roa", "loss", "eps_vol"]
TEXT_VARS     = [
    "prep_lm_tone", "prep_lm_uncertainty", "prep_lm_negative",
    "prep_fog_index", "prep_word_count",
    "qa_lm_tone",   "qa_lm_uncertainty",
    "qa_fog_index",  "qa_word_count",
]

# Convert string columns to object dtype so patsy can handle them
for col in ["gvkey", "ticker", "companyname"]:
    if col in panel.columns:
        panel[col] = panel[col].astype(object)

# Time FE: year-quarter
panel["ym"]       = pd.to_datetime(panel["call_date"]).dt.to_period("Q").astype(str)
panel["log_prep_wc"] = np.log1p(panel.get("prep_word_count", 0))
panel["log_qa_wc"]   = np.log1p(panel.get("qa_word_count",   0))

# Replace raw word counts with log in TEXT_VARS
TEXT_VARS_FINAL = [v for v in TEXT_VARS
                   if v not in ("prep_word_count","qa_word_count")] \
                + ["log_prep_wc", "log_qa_wc"]

# ── Helper: run OLS and extract results ───────────────────────────────────────
def run_ols(data, formula, cluster_var="gvkey"):
    """Run OLS with clustered standard errors."""
    m = smf.ols(formula, data=data.dropna(subset=data.columns.tolist())).fit(
        cov_type="cluster",
        cov_kwds={"groups": data.dropna(subset=data.columns.tolist())[cluster_var]}
    )
    return m

def extract(m, var):
    """Extract coef, se, t-stat, p-val for one variable."""
    if var not in m.params:
        return (np.nan,)*4
    return (m.params[var], m.bse[var], m.tvalues[var], m.pvalues[var])

# ── Drop to common sample across all specs ────────────────────────────────────
req_vars = [Y, D] + FIRM_CONTROLS + TEXT_VARS_FINAL
sample = panel.dropna(subset=[Y, D]).copy()
print(f"\nEstimation sample (Y and D non-missing): {len(sample):,}")

# ── Specification 1: Raw ERC ──────────────────────────────────────────────────
print("\nSpec 1: CAR ~ SUE")
m1 = smf.ols(f"{Y} ~ {D}", data=sample).fit(
    cov_type="cluster", cov_kwds={"groups": sample["gvkey"]}
)
print(f"  SUE coef = {m1.params[D]:.4f}  t = {m1.tvalues[D]:.2f}  R2 = {m1.rsquared:.4f}")

# ── Specification 2: + Firm controls ─────────────────────────────────────────
s2 = sample.dropna(subset=FIRM_CONTROLS)
ctrl_str = " + ".join(FIRM_CONTROLS)
print(f"\nSpec 2: CAR ~ SUE + controls  (n={len(s2):,})")
m2 = smf.ols(f"{Y} ~ {D} + {ctrl_str}", data=s2).fit(
    cov_type="cluster", cov_kwds={"groups": s2["gvkey"]}
)
print(f"  SUE coef = {m2.params[D]:.4f}  t = {m2.tvalues[D]:.2f}  R2 = {m2.rsquared:.4f}")

# ── Specification 3: + Text ───────────────────────────────────────────────────
text_present = [v for v in TEXT_VARS_FINAL if v in sample.columns]
s3 = sample.dropna(subset=FIRM_CONTROLS + text_present)
txt_str = " + ".join(text_present)
print(f"\nSpec 3: CAR ~ SUE + controls + text  (n={len(s3):,})")
m3 = smf.ols(f"{Y} ~ {D} + {ctrl_str} + {txt_str}", data=s3).fit(
    cov_type="cluster", cov_kwds={"groups": s3["gvkey"]}
)
print(f"  SUE coef = {m3.params[D]:.4f}  t = {m3.tvalues[D]:.2f}  R2 = {m3.rsquared:.4f}")

# ── Specification 4: + Firm FE + Time FE ─────────────────────────────────────
s4 = s3.copy()
print(f"\nSpec 4: CAR ~ SUE + controls + text + Firm FE + Time FE  (n={len(s4):,})")
m4 = smf.ols(
    f"{Y} ~ {D} + {ctrl_str} + {txt_str} + C(ym) + C(gvkey)",
    data=s4
).fit(cov_type="cluster", cov_kwds={"groups": s4["gvkey"]})
print(f"  SUE coef = {m4.params[D]:.4f}  t = {m4.tvalues[D]:.2f}  R2 = {m4.rsquared:.4f}")

# ── Build coefficient table ───────────────────────────────────────────────────
rows = []
key_vars = [D] + FIRM_CONTROLS + text_present

for var in key_vars:
    row = {"variable": var}
    for label, m in [("(1) Raw", m1), ("(2) +Controls", m2),
                     ("(3) +Text", m3), ("(4) +FE", m4)]:
        coef, se, t, p = extract(m, var)
        stars = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
        row[label] = f"{coef:.4f}{stars}" if not np.isnan(coef) else ""
        row[f"{label}_se"] = f"({se:.4f})" if not np.isnan(se) else ""
    rows.append(row)

# Bottom rows
meta = [
    {"variable": "N",
     "(1) Raw": len(sample), "(2) +Controls": len(s2),
     "(3) +Text": len(s3), "(4) +FE": len(s4)},
    {"variable": "R²",
     "(1) Raw": f"{m1.rsquared:.4f}", "(2) +Controls": f"{m2.rsquared:.4f}",
     "(3) +Text": f"{m3.rsquared:.4f}", "(4) +FE": f"{m4.rsquared:.4f}"},
    {"variable": "Firm FE",  "(1) Raw": "No",  "(2) +Controls": "No",
     "(3) +Text": "No", "(4) +FE": "Yes"},
    {"variable": "Time FE",  "(1) Raw": "No",  "(2) +Controls": "No",
     "(3) +Text": "No", "(4) +FE": "Yes"},
    {"variable": "Clustered SE (firm)", "(1) Raw": "Yes", "(2) +Controls": "Yes",
     "(3) +Text": "Yes", "(4) +FE": "Yes"},
]

table = pd.DataFrame(rows + meta)
out = RESULTS / "study_a_ols_table.csv"
table.to_csv(out, index=False)
print(f"\nSaved → {out}")

# ── Summary text ─────────────────────────────────────────────────────────────
summary = f"""
Project A OLS Results
=====================
Sample: {len(s3):,} firm-quarter observations

Key finding (SUE coefficient = ERC):
  Spec 1 (raw):         {m1.params[D]:.4f}  (t={m1.tvalues[D]:.2f})
  Spec 2 (+controls):   {m2.params[D]:.4f}  (t={m2.tvalues[D]:.2f})
  Spec 3 (+text):       {m3.params[D]:.4f}  (t={m3.tvalues[D]:.2f})
  Spec 4 (+FE):         {m4.params[D]:.4f}  (t={m4.tvalues[D]:.2f})

Text channel (prepared remarks tone in Spec 3):
  prep_lm_tone coef:    {extract(m3, 'prep_lm_tone')[0]:.4f}
  (t={extract(m3, 'prep_lm_tone')[2]:.2f})
"""
print(summary)
(RESULTS / "study_a_ols_stats.txt").write_text(summary)
