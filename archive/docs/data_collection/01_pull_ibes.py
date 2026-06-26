"""
01_pull_ibes.py
───────────────
Pull I/B/E/S quarterly earnings surprise data from WRDS.

Output: data/ibes_quarterly.parquet
Columns:
    ticker      IBES ticker
    cusip       8-char CUSIP (used to link to CRSP)
    anndats     actual announcement date
    fpedats     fiscal period end date
    actual      actual EPS
    medest      median analyst forecast (most recent before announcement)
    meanest     mean analyst forecast
    stdev       std-dev of forecasts
    numest      number of estimates
    SUE         standardized unexpected earnings = (actual - medest) / stdev
    SUE_price   price-scaled surprise (fallback when stdev == 0)
"""

import wrds
import pandas as pd
import numpy as np
from config import WRDS_USERNAME, START_DATE, END_DATE, IBES_FILE

# ── Connect ───────────────────────────────────────────────────────────────────
print("Connecting to WRDS …")
db = wrds.Connection(wrds_username=WRDS_USERNAME)

# ── Pull I/B/E/S summary (quarterly, US firms only) ───────────────────────────
# fpi = '6'  →  current fiscal quarter forecast
# usfirm = 1 →  US firms only
# curcode = 'USD' → USD-denominated forecasts
print("Pulling I/B/E/S statsum_us …")
query = f"""
    SELECT
        ticker,
        cusip,
        anndats,
        fpedats,
        fpi,
        actual,
        medest,
        meanest,
        stdev,
        numest,
        curcode
    FROM ibes.statsum_us
    WHERE fpi     = '6'
      AND usfirm  = 1
      AND curcode = 'USD'
      AND actual  IS NOT NULL
      AND medest  IS NOT NULL
      AND anndats BETWEEN '{START_DATE}' AND '{END_DATE}'
"""
ibes = db.raw_sql(query, date_cols=["anndats", "fpedats"])
print(f"  Raw rows: {len(ibes):,}")

# ── Clean & derive SUE ────────────────────────────────────────────────────────
# 1. Drop rows with no CUSIP (can't link to CRSP)
ibes = ibes.dropna(subset=["cusip"])
ibes["cusip"] = ibes["cusip"].str.strip().str.upper()

# 2. Standardized Unexpected Earnings (primary)
#    SUE = (actual - median_forecast) / stdev_of_forecasts
#    Floor stdev at a small positive value to avoid division by zero;
#    if stdev is genuinely 0 or missing, flag and use price-scaled version later
ibes["surprise"] = ibes["actual"] - ibes["medest"]

MIN_STDEV = 0.001   # floors effectively-zero stdev
ibes["stdev_clean"] = ibes["stdev"].where(ibes["stdev"] > MIN_STDEV, np.nan)
ibes["SUE"] = ibes["surprise"] / ibes["stdev_clean"]

# 3. Winsorize SUE at 1st / 99th percentile (standard in finance)
q01 = ibes["SUE"].quantile(0.01)
q99 = ibes["SUE"].quantile(0.99)
ibes["SUE"] = ibes["SUE"].clip(q01, q99)

# 4. Drop extreme outliers in actual / medest (data errors)
for col in ["actual", "medest"]:
    lo = ibes[col].quantile(0.001)
    hi = ibes[col].quantile(0.999)
    ibes = ibes[ibes[col].between(lo, hi)]

# 5. Require at least 2 analyst estimates (single-analyst consensus is noisy)
ibes = ibes[ibes["numest"] >= 2]

# 6. Keep one record per (ticker, fpedats) — take the row closest to anndats
#    (should already be unique in statsum, but be safe)
ibes = (
    ibes
    .sort_values("anndats")
    .drop_duplicates(subset=["ticker", "fpedats"], keep="last")
)

print(f"  After cleaning: {len(ibes):,} firm-quarter observations")
print(f"  SUE stats:\n{ibes['SUE'].describe().round(3)}")
print(f"  Date range: {ibes['anndats'].min()} → {ibes['anndats'].max()}")

# ── Save ──────────────────────────────────────────────────────────────────────
ibes.to_parquet(IBES_FILE, index=False)
print(f"Saved → {IBES_FILE}")

db.close()
