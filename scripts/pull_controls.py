"""
pull_controls.py
─────────────────
Pull firm-level control variables and HTE moderators.

Sources:
  A. Compustat fundq  → firm controls + complexity variables
  B. tr_13f.s34       → institutional ownership (HTE: sophistication)
  C. ibes_sue.parquet → analyst coverage (HTE: information intermediation)

Firm controls (matched to fiscal quarter fpedats):
  size        = log(total assets)
  btm         = book equity / market equity
  leverage    = total debt / total assets
  roa         = net income / total assets (lagged)
  eps_vol     = std(actual EPS over prior 8 quarters)
  special_items = abs(special items / total assets)  (complexity proxy)
  loss        = I(net income < 0)

HTE moderators:
  io_pct      = % shares held by institutions (13F)  → sophistication
  numest      = analyst count from IBES              → intermediation
  eps_vol     = EPS volatility                       → cognitive load / complexity

Output: data/firm_controls.parquet
"""

import wrds
import numpy as np
import pandas as pd
from pathlib import Path

BASE      = Path(__file__).parent.parent / "data"
WRDS_USER = "yulinwang"

db = wrds.Connection(wrds_username=WRDS_USER)

# ── A. Compustat fundq ────────────────────────────────────────────────────────
print("Step A: pulling Compustat fundq …")
fundq = db.raw_sql("""
    SELECT gvkey, datadate, fyearq, fqtr, rdq,
           atq   AS total_assets,
           ceqq  AS book_equity,
           dlcq  AS debt_current,
           dlttq AS debt_lt,
           ibq   AS net_income,
           spiq  AS special_items,
           revtq AS revenue
    FROM comp.fundq
    WHERE indfmt = 'INDL'
      AND datafmt = 'STD'
      AND popsrc = 'D'
      AND consol = 'C'
      AND datadate >= '2004-01-01'
      AND atq > 0
""", date_cols=["datadate", "rdq"])

print(f"  fundq rows: {len(fundq):,}")
print(f"  Unique gvkeys: {fundq['gvkey'].nunique():,}")

# ── Compute quarterly firm controls ──────────────────────────────────────────
# Size (log total assets) — mktcap will be joined from CRSP later
fundq["size"] = np.log(fundq["total_assets"].clip(lower=0.001))
fundq["mktcap"] = np.nan   # placeholder; filled from CRSP in merge_event_panel

# Book-to-market (asset-based proxy until CRSP mktcap joined)
fundq["btm"] = (fundq["book_equity"] / fundq["total_assets"]).clip(-10, 10)

# Leverage
fundq["total_debt"] = fundq["debt_current"].fillna(0) + fundq["debt_lt"].fillna(0)
fundq["leverage"]   = fundq["total_debt"] / fundq["total_assets"]
fundq["leverage"]   = fundq["leverage"].clip(0, 5)

# ROA
fundq["roa"] = fundq["net_income"] / fundq["total_assets"]
fundq["roa"] = fundq["roa"].clip(-2, 2)

# Loss indicator
fundq["loss"] = (fundq["net_income"].fillna(0) < 0).astype(int)

# Special items (absolute, scaled by assets) → earnings complexity proxy
fundq["special_items_scaled"] = (
    fundq["special_items"].fillna(0).abs() / fundq["total_assets"]
).clip(0, 1)

# ── EPS volatility (trailing 8 quarters) ─────────────────────────────────────
# Need actual EPS from IBES; load it here for computing eps_vol
print("  Computing EPS volatility (trailing 8 quarters) …")
ibes = pd.read_parquet(BASE / "ibes_sue.parquet",
                       columns=["ticker", "fpedats", "actual"])

# Link gvkey → ticker via transcript_events
tevents = pd.read_parquet(BASE / "transcript_events.parquet",
                          columns=["gvkey", "ciq_ticker"])
tevents = tevents.rename(columns={"ciq_ticker": "ticker"}).drop_duplicates("gvkey")

ibes_gvkey = ibes.merge(tevents, on="ticker", how="left").dropna(subset=["gvkey"])
ibes_gvkey = ibes_gvkey.sort_values(["gvkey", "fpedats"])

# Rolling 8-quarter std of actual EPS
eps_vol = (
    ibes_gvkey.groupby("gvkey")["actual"]
    .transform(lambda x: x.rolling(8, min_periods=4).std())
)
ibes_gvkey["eps_vol"] = eps_vol
ibes_gvkey = ibes_gvkey[["gvkey", "fpedats", "eps_vol"]].dropna()

# Merge eps_vol into fundq on gvkey + datadate ≈ fpedats
fundq["datadate_ym"] = fundq["datadate"].dt.to_period("M")
ibes_gvkey["fpedats_ym"] = ibes_gvkey["fpedats"].dt.to_period("M")
eps_vol_merge = ibes_gvkey.rename(columns={"fpedats_ym": "datadate_ym"})

fundq = fundq.merge(eps_vol_merge[["gvkey", "datadate_ym", "eps_vol"]],
                    on=["gvkey", "datadate_ym"], how="left")
fundq = fundq.drop(columns=["datadate_ym"])

print(f"  EPS vol attached to {fundq['eps_vol'].notna().sum():,} / {len(fundq):,} obs")

# ── B. 13F Institutional Ownership ───────────────────────────────────────────
print("\nStep B: pulling 13F institutional ownership …")
try:
    # tr_13f.s34 is Thomson/Refinitiv 13F holdings
    # rdate = report date (quarter-end), instown = # institutions
    # Use wrds_13f.inst_holding_summary if available, otherwise compute from s34
    io = db.raw_sql("""
        SELECT rdate, permno,
               SUM(shares) AS inst_shares
        FROM tr_13f.s34
        WHERE rdate >= '2004-01-01'
        GROUP BY rdate, permno
    """, date_cols=["rdate"])

    print(f"  13F rows: {len(io):,}")

    # Get shares outstanding from CRSP to compute io_pct
    shrout = db.raw_sql("""
        SELECT permno, date, shrout
        FROM crsp.dsf
        WHERE date >= '2004-01-01'
          AND shrout IS NOT NULL
          AND shrout > 0
    """, date_cols=["date"])

    # Match 13F rdate to nearest CRSP shrout at quarter-end
    shrout["year_q"] = shrout["date"].dt.to_period("Q")
    # Keep only last trading day of each quarter for each permno
    shrout_qtr = (
        shrout.groupby(["permno", "year_q"])
        .apply(lambda g: g.sort_values("date").iloc[-1])
        .reset_index(drop=True)
        [["permno", "date", "shrout"]]
    )
    io["year_q"] = io["rdate"].dt.to_period("Q")
    io = io.merge(shrout_qtr[["permno", "year_q", "shrout"]], on=["permno", "year_q"], how="left")
    io["io_pct"] = (io["inst_shares"] * 100 / (io["shrout"] * 1000)).clip(0, 150)
    io = io[["permno", "rdate", "year_q", "io_pct"]].dropna(subset=["io_pct"])
    print(f"  IO records with pct: {len(io):,}")

    io.to_parquet(BASE / "io_pct.parquet", index=False)
    print(f"  Saved io_pct.parquet")

except Exception as e:
    print(f"  13F pull failed: {e}")
    print("  Skipping IO — will use NaN for io_pct in panel")
    io = None

# ── Save fundq controls ───────────────────────────────────────────────────────
control_cols = [
    "gvkey", "datadate", "fyearq", "fqtr", "rdq",
    "total_assets", "size", "btm", "leverage", "roa", "loss",
    "special_items_scaled", "eps_vol", "mktcap"
]
controls = fundq[control_cols].copy()

out = BASE / "firm_controls.parquet"
controls.to_parquet(out, index=False)
print(f"\nSaved → {out}  ({len(controls):,} rows)")

db.close()
