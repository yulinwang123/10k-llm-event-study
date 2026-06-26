"""
fix_io_pct.py
──────────────
Fix institutional ownership (io_pct) using CUSIP-based linkage.

Problem: tr_13f.s34 does not have a 'permno' column; it has 'cusip'.
Fix:     Join s34.cusip → crsp.stocknames.ncusip → permno.
         Then compute io_pct = inst_shares / (shrout * 1000).

Output: data/io_pct.parquet
Columns: permno, rdate, year_q, io_pct
"""

import wrds
import numpy as np
import pandas as pd
from pathlib import Path

BASE      = Path(__file__).parent.parent / "data"
WRDS_USER = "yulinwang"
START     = "2009-01-01"   # one year before sample to ensure coverage

db = wrds.Connection(wrds_username=WRDS_USER)

# ── Step 1: Get 13F holdings aggregated by (cusip, rdate) ────────────────────
print("Step 1: pulling 13F aggregate holdings by cusip + rdate …")
io_cusip = db.raw_sql(f"""
    SELECT rdate,
           SUBSTRING(cusip, 1, 8) AS cusip8,
           SUM(shares)            AS inst_shares
    FROM tr_13f.s34
    WHERE rdate >= '{START}'
      AND cusip IS NOT NULL
      AND shares > 0
    GROUP BY rdate, SUBSTRING(cusip, 1, 8)
""", date_cols=["rdate"])

print(f"  Rows: {len(io_cusip):,}  |  unique cusip8: {io_cusip['cusip8'].nunique():,}")

# ── Step 2: Link cusip → permno via crsp.stocknames ──────────────────────────
print("\nStep 2: linking cusip8 → permno via crsp.stocknames …")
names = db.raw_sql("""
    SELECT permno,
           SUBSTRING(ncusip, 1, 8) AS cusip8,
           namedt,
           COALESCE(nameenddt, '2099-12-31'::date) AS nameendt
    FROM crsp.stocknames
    WHERE ncusip IS NOT NULL
      AND ncusip != ''
""", date_cols=["namedt", "nameendt"])

print(f"  stocknames rows: {len(names):,}")

# Merge on cusip8, then filter by date range
merged = io_cusip.merge(names, on="cusip8", how="inner")
date_ok = (merged["rdate"] >= merged["namedt"]) & (merged["rdate"] <= merged["nameendt"])
merged  = merged[date_ok]

# If multiple permno match same cusip8 on same date (rare), keep most recent namedt
merged = (merged.sort_values("namedt", ascending=False)
                .drop_duplicates(subset=["rdate", "cusip8"]))

# Aggregate to permno level
io_perm = (merged.groupby(["permno", "rdate"])["inst_shares"]
                 .sum().reset_index())
print(f"  Matched rows: {len(io_perm):,}  |  unique permno: {io_perm['permno'].nunique():,}")

# ── Step 3: Get shares outstanding from CRSP (quarter-end) ───────────────────
print("\nStep 3: pulling CRSP shares outstanding (quarter-end) …")
permnos = io_perm["permno"].unique().tolist()
min_date = io_perm["rdate"].min() - pd.Timedelta(days=30)
max_date = io_perm["rdate"].max() + pd.Timedelta(days=30)

BATCH = 1000
shrout_chunks = []
for i in range(0, len(permnos), BATCH):
    batch = permnos[i:i+BATCH]
    ids_str = ",".join(str(p) for p in batch)
    chunk = db.raw_sql(f"""
        SELECT permno, date, shrout
        FROM crsp.dsf
        WHERE permno IN ({ids_str})
          AND date BETWEEN '{min_date.date()}' AND '{max_date.date()}'
          AND shrout IS NOT NULL AND shrout > 0
    """, date_cols=["date"])
    shrout_chunks.append(chunk)
    if i % (BATCH * 10) == 0:
        print(f"  shrout batch {i//BATCH+1}/{len(permnos)//BATCH+1}")

shrout = pd.concat(shrout_chunks, ignore_index=True)
shrout["year_q"] = shrout["date"].dt.to_period("Q")

# Keep last trading day of each quarter per permno
shrout_qtr = (
    shrout.sort_values("date")
          .groupby(["permno", "year_q"])
          .last()
          .reset_index()
          [["permno", "year_q", "shrout"]]
)
print(f"  shrout quarterly obs: {len(shrout_qtr):,}")

# ── Step 4: Compute io_pct ────────────────────────────────────────────────────
print("\nStep 4: computing io_pct …")
io_perm["year_q"] = io_perm["rdate"].dt.to_period("Q")
io_perm = io_perm.merge(shrout_qtr, on=["permno", "year_q"], how="left")

# shrout in crsp.dsf is in thousands
io_perm["io_pct"] = (io_perm["inst_shares"] / (io_perm["shrout"] * 1000) * 100).clip(0, 150)
io_perm = io_perm.dropna(subset=["io_pct"])

print(f"  io_pct records: {len(io_perm):,}")
print(f"  io_pct stats:\n{io_perm['io_pct'].describe().round(2).to_string()}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = BASE / "io_pct.parquet"
io_perm[["permno", "rdate", "year_q", "io_pct"]].to_parquet(out, index=False)
print(f"\nSaved → {out}  ({len(io_perm):,} rows)")

db.close()
