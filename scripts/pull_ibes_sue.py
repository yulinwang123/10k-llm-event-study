"""
pull_ibes_sue.py
─────────────────
Pull quarterly earnings surprise (SUE) from IBES.

Logic:
  - Table: ibes.statsum_epsus (NOT statsum_us)
  - Quarterly: fpi = '6'
  - Announcement date: anndats_act (NOT anndats)
  - For each (ticker, fpedats), take the consensus snapshot closest to
    but BEFORE anndats_act to avoid look-ahead bias.
  - SUE = (actual - medest) / stdev

Output: data/ibes_sue.parquet
Columns:
  ticker, cusip, fpedats, anndats_act, statpers,
  actual, medest, stdev, numest, sue, sue_win
"""

import wrds
import pandas as pd
from pathlib import Path

BASE      = Path(__file__).parent.parent / "data"
WRDS_USER = os.getenv("WRDS_USER", "your_wrds_username")
START     = "2004-01-01"
END       = "2024-12-31"

db = wrds.Connection(wrds_username=WRDS_USER)

# ── Pull pre-announcement consensus via window function ───────────────────────
# For each (ticker, fpedats):
#   - keep rows where statpers < anndats_act  (no look-ahead)
#   - take row with MAX(statpers)  → last consensus before announcement
#   - require numest >= 2 and stdev > 0
print("Pulling IBES quarterly pre-announcement consensus …")

ibes = db.raw_sql(f"""
    WITH ranked AS (
        SELECT
            ticker,
            cusip,
            fpedats,
            statpers,
            anndats_act,
            actual,
            medest,
            stdev,
            numest,
            ROW_NUMBER() OVER (
                PARTITION BY ticker, fpedats
                ORDER BY statpers DESC
            ) AS rn
        FROM ibes.statsum_epsus
        WHERE usfirm = 1
          AND fpi = '6'
          AND anndats_act IS NOT NULL
          AND actual      IS NOT NULL
          AND stdev > 0
          AND numest >= 2
          AND statpers < anndats_act
          AND anndats_act BETWEEN '{START}' AND '{END}'
    )
    SELECT ticker, cusip, fpedats, statpers, anndats_act,
           actual, medest, stdev, numest
    FROM ranked
    WHERE rn = 1
""", date_cols=["fpedats", "statpers", "anndats_act"])

print(f"  Raw rows: {len(ibes):,}")
print(f"  Unique (ticker, fpedats): {ibes.groupby(['ticker','fpedats']).ngroups:,}")
print(f"  Announcement date range: {ibes['anndats_act'].min()} → {ibes['anndats_act'].max()}")

# ── Compute SUE ───────────────────────────────────────────────────────────────
ibes["sue"] = (ibes["actual"] - ibes["medest"]) / ibes["stdev"]

# Winsorize at 1st/99th percentile
p01 = ibes["sue"].quantile(0.01)
p99 = ibes["sue"].quantile(0.99)
ibes["sue_win"] = ibes["sue"].clip(p01, p99)

print(f"\n  SUE (raw) mean / std / median:")
print(f"  {ibes['sue'].agg(['mean','std','median']).round(3).to_string()}")
print(f"\n  SUE (winsorized) mean / std:")
print(f"  {ibes['sue_win'].agg(['mean','std']).round(3).to_string()}")

# Safety dedup
ibes = ibes.drop_duplicates(subset=["ticker", "fpedats"])

# ── Save ──────────────────────────────────────────────────────────────────────
out = BASE / "ibes_sue.parquet"
ibes.to_parquet(out, index=False)
print(f"\nSaved → {out}  ({len(ibes):,} rows)")

db.close()
