"""
merge_study_panel.py
─────────────────────
Merge all sources into the final study panel for Study A + B.

What's already in analysis_panel (DON'T re-merge):
    lm_*, fb_*, embed_*  — text measures
    log_assets, log_mktcap, bm_ratio, roa, leverage  — firm controls
    car_1_1, car_3_3  — CAR around date_filed (keep for comparison)
    rdq  — earnings announcement date
    fyear as STRING type ('2011'...'2020')

What we add here:
    CAR_short, CAR_long  — CAR around rdq (from car_rdq.parquet)
    SUE, actual, medest  — earnings surprise (from ibes_sue.parquet)
    management_optimism, guidance_specificity,
    uncertainty_hedging, risk_framing  — LLM scores (from llm_out/)

Output:
    data/study_panel.parquet
    data/study_panel.csv   (no heavy columns, for quick inspection)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE    = Path(__file__).parent.parent / "data"
LLM_DIR = BASE / "llm_out"

# ── 1. Base panel (already has lm/finbert/embed/controls) ────────────────────
print("Loading analysis_panel …")
panel = pd.read_parquet(BASE / "analysis_panel.parquet")
# Normalise fyear to int throughout
panel["fyear"] = panel["fyear"].astype(int)
print(f"  {len(panel):,} rows, {panel['permno'].nunique():,} firms, "
      f"fyear {panel['fyear'].min()}–{panel['fyear'].max()}")

# ── 2. CAR around rdq ─────────────────────────────────────────────────────────
print("Merging CAR around rdq …")
car_rdq = pd.read_parquet(BASE / "car_rdq.parquet",
                           columns=["gvkey","fyear","CAR_short","CAR_long",
                                    "n_short","n_long"])
car_rdq["fyear"] = car_rdq["fyear"].astype(int)
panel = panel.merge(car_rdq, on=["gvkey","fyear"], how="left")
print(f"  CAR_short non-null: {panel['CAR_short'].notna().sum():,} "
      f"({panel['CAR_short'].notna().mean():.1%})")

# ── 3. I/B/E/S SUE ───────────────────────────────────────────────────────────
print("Merging I/B/E/S SUE …")
sue = pd.read_parquet(BASE / "ibes_sue.parquet",
                      columns=["gvkey","fyear","anndats_ibes",
                                "actual","medest","numest","SUE"])
sue["fyear"] = sue["fyear"].astype(int)
panel = panel.merge(sue, on=["gvkey","fyear"], how="left")
print(f"  SUE non-null: {panel['SUE'].notna().sum():,} "
      f"({panel['SUE'].notna().mean():.1%})")

# ── 4. LLM dimension scores ───────────────────────────────────────────────────
print("Loading LLM scores …")
llm_records = []
for fp in sorted(LLM_DIR.glob("results_*.jsonl")):
    with open(fp) as f:
        for line in f:
            r = json.loads(line)
            scores = r.get("scores", {})
            if isinstance(scores, str):
                try:
                    scores = json.loads(scores)
                except Exception:
                    scores = {}
            llm_records.append({
                "gvkey":                 r["gvkey"],
                "fyear":                 int(r["fyear"]),
                "management_optimism":   scores.get("management_optimism"),
                "guidance_specificity":  scores.get("guidance_specificity"),
                "uncertainty_hedging":   scores.get("uncertainty_hedging"),
                "risk_framing":          scores.get("risk_framing"),
            })

llm_df = pd.DataFrame(llm_records)
llm_df["fyear"] = llm_df["fyear"].astype(int)
panel  = panel.merge(llm_df, on=["gvkey","fyear"], how="left")
llm_cover = panel["management_optimism"].notna().mean()
print(f"  LLM scores non-null: {panel['management_optimism'].notna().sum():,} "
      f"({llm_cover:.1%})")

# ── 5. Composite variables ────────────────────────────────────────────────────
# LLM composite (z-score average of four dimensions)
llm_cols = ["management_optimism","guidance_specificity",
            "uncertainty_hedging","risk_framing"]
for col in llm_cols:
    mu, sd = panel[col].mean(), panel[col].std()
    panel[col + "_z"] = (panel[col] - mu) / sd
panel["llm_composite"] = panel[[c + "_z" for c in llm_cols]].mean(axis=1)

# ── 6. Summary ────────────────────────────────────────────────────────────────
study_a = panel.dropna(subset=["SUE", "CAR_short"])
print(f"\nStudy A sample (SUE + CAR_short non-null): {len(study_a):,}")

print("\n── Key variable summary ─────────────────────────────────────")
key = ["SUE","CAR_short","CAR_long","car_1_1",
       "lm_tone","fb_net","embed_novelty","llm_composite"]
print(panel[[v for v in key if v in panel.columns]].describe().round(4))

# ── 7. Save ───────────────────────────────────────────────────────────────────
panel.to_parquet(BASE / "study_panel.parquet", index=False)
print(f"\nSaved → data/study_panel.parquet  ({len(panel):,} rows)")

skip_csv = ["embed_cos_sim", "lm_negative", "lm_positive",
            "lm_litigious", "lm_strong_modal", "lm_weak_modal"]
csv_cols = [c for c in panel.columns if c not in skip_csv]
panel[csv_cols].to_csv(BASE / "study_panel.csv", index=False)
print(f"Saved → data/study_panel.csv      ({len(panel):,} rows)")
print("\nReady for DML estimation.")
