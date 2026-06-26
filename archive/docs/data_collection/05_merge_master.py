"""
05_merge_master.py
───────────────────
Merge all data sources into a single analysis-ready panel.

Sources:
    data/ibes_quarterly.parquet  (01_pull_ibes.py)
    data/ibes_crsp_link.parquet  (02_ibes_crsp_link.py)
    data/car_panel.parquet       (03_pull_crsp_car.py)
    data/transcripts.parquet     (04_pull_transcripts.py)

Output: data/master_panel.parquet + data/master_panel.csv

Final schema:
    permno          CRSP PERMNO
    gvkey           Compustat GVKEY
    ticker          IBES ticker
    anndats         earnings announcement date
    fpedats         fiscal period end date
    actual          actual EPS
    medest          median analyst forecast
    stdev           std-dev of forecasts
    numest          number of estimates
    SUE             standardized unexpected earnings
    CAR_short       CAR [-1, +1]
    CAR_long        CAR [+2, +60]
    prepared_text   earnings call prepared remarks text
    qa_text         earnings call Q&A text
    full_text       full transcript text
    has_transcript  1 if transcript matched, 0 otherwise
"""

import pandas as pd
import numpy as np
from config import (
    IBES_FILE, LINK_FILE, CAR_FILE, TRANSCRIPT_FILE,
    MASTER_FILE, MASTER_CSV
)

print("Loading data …")
ibes        = pd.read_parquet(IBES_FILE)
link        = pd.read_parquet(LINK_FILE)
car         = pd.read_parquet(CAR_FILE)

# Transcripts optional — may not exist if both sources failed
try:
    trans = pd.read_parquet(TRANSCRIPT_FILE)
    has_transcripts = True
    print(f"  Transcripts: {len(trans):,} rows")
except FileNotFoundError:
    trans = pd.DataFrame()
    has_transcripts = False
    print("  Transcripts: not found (will proceed without)")

print(f"  IBES:  {len(ibes):,} rows")
print(f"  Link:  {len(link):,} rows")
print(f"  CAR:   {len(car):,} rows")

# ── Step 1: attach PERMNO & GVKEY to IBES ────────────────────────────────────
master = ibes.merge(
    link[["ticker", "anndats", "permno", "gvkey"]],
    on=["ticker", "anndats"],
    how="inner"
)
master["permno"] = master["permno"].astype(int)
print(f"\nAfter IBES ⟶ PERMNO merge: {len(master):,} rows")

# ── Step 2: attach CAR ────────────────────────────────────────────────────────
master = master.merge(
    car[["permno", "anndats", "CAR_short", "CAR_long",
         "n_obs_CAR_short", "n_obs_CAR_long"]],
    on=["permno", "anndats"],
    how="inner"
)
print(f"After CAR merge:            {len(master):,} rows")

# ── Step 3: attach transcripts (left join — keep all, flag missing) ───────────
if has_transcripts and len(trans) > 0:
    trans["permno"] = trans["permno"].astype(int)
    master = master.merge(
        trans[["permno", "anndats", "prepared_text", "qa_text", "full_text"]],
        on=["permno", "anndats"],
        how="left"
    )
    master["has_transcript"] = master["full_text"].notna().astype(int)
    print(f"After transcript merge:     {len(master):,} rows")
    print(f"  With transcripts: {master['has_transcript'].sum():,} "
          f"({master['has_transcript'].mean():.1%})")
else:
    master["prepared_text"] = np.nan
    master["qa_text"]       = np.nan
    master["full_text"]     = np.nan
    master["has_transcript"] = 0
    print("No transcripts attached.")

# ── Step 4: Additional cleaning ───────────────────────────────────────────────
# Sort by firm and date
master = master.sort_values(["permno", "anndats"]).reset_index(drop=True)

# Drop any duplicates (shouldn't exist but sanity check)
n_before = len(master)
master = master.drop_duplicates(subset=["permno", "anndats"])
if len(master) < n_before:
    print(f"  Dropped {n_before - len(master)} duplicate rows")

# Add calendar variables (useful for FE)
master["year"]    = master["anndats"].dt.year
master["quarter"] = master["anndats"].dt.quarter
master["yearqtr"] = master["year"].astype(str) + "Q" + master["quarter"].astype(str)

# Column order
cols = [
    "permno", "gvkey", "ticker", "anndats", "fpedats", "yearqtr",
    "year", "quarter",
    "actual", "medest", "stdev", "numest", "SUE",
    "CAR_short", "CAR_long",
    "n_obs_CAR_short", "n_obs_CAR_long",
    "has_transcript", "prepared_text", "qa_text", "full_text"
]
# Keep only columns that exist
cols = [c for c in cols if c in master.columns]
master = master[cols]

# ── Step 5: Summary statistics ────────────────────────────────────────────────
print("\n── Master Panel Summary ──────────────────────────────────")
print(f"  Total obs:          {len(master):,}")
print(f"  Unique firms:       {master['permno'].nunique():,}")
print(f"  Date range:         {master['anndats'].min().date()} → {master['anndats'].max().date()}")
print(f"  With transcripts:   {master['has_transcript'].sum():,} ({master['has_transcript'].mean():.1%})")
print()
print(master[["SUE", "CAR_short", "CAR_long"]].describe().round(4))

# ── Save ──────────────────────────────────────────────────────────────────────
master.to_parquet(MASTER_FILE, index=False)
print(f"\nSaved parquet → {MASTER_FILE}")

# CSV without long text columns (for easy inspection in Excel)
csv_cols = [c for c in cols if c not in ["prepared_text", "qa_text", "full_text"]]
master[csv_cols].to_csv(MASTER_CSV, index=False)
print(f"Saved CSV     → {MASTER_CSV}")
print("\nDone. Run DML estimation next.")
