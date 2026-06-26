"""
compute_car_around_rdq.py
──────────────────────────
Recompute CAR around earnings announcement date (rdq) using
already-downloaded crsp_daily.parquet and crsp_market.parquet.

NO WRDS connection needed — runs entirely on local data.

Input:  data/analysis_panel.parquet  (has permno + rdq)
        data/crsp_daily.parquet      (daily stock returns, 2009-2021)
        data/crsp_market.parquet     (vwretd market returns)

Output: data/car_rdq.parquet
Columns:
    gvkey, permno, fyear, rdq
    CAR_short   : compounded market-adjusted return [-1, +1] around rdq
    CAR_long    : compounded market-adjusted return [+2, +60] around rdq
    n_short     : trading days in short window
    n_long      : trading days in long window
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent / "data"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data …")
panel  = pd.read_parquet(BASE / "analysis_panel.parquet",
                         columns=["gvkey", "permno", "fyear", "rdq"])
crsp   = pd.read_parquet(BASE / "crsp_daily.parquet")
market = pd.read_parquet(BASE / "crsp_market.parquet")

panel  = panel.dropna(subset=["permno", "rdq"]).copy()
panel["permno"] = panel["permno"].astype(int)
crsp["permno"]  = crsp["permno"].astype(int)

print(f"  Events to process : {len(panel):,}")
print(f"  CRSP daily rows   : {len(crsp):,}")

# ── Build trading-day calendar ────────────────────────────────────────────────
market = market.sort_values("date").reset_index(drop=True)
trading_days = pd.DatetimeIndex(market["date"].values)   # for searchsorted compatibility
mkt_ret_map  = market.set_index("date")["mkt_ret"]

def trading_day_offset(base_date, offset):
    """Return the trading date `offset` days from base_date."""
    idx = np.searchsorted(trading_days, base_date)
    target = idx + offset
    if target < 0 or target >= len(trading_days):
        return None
    return trading_days[target]

# ── Attach market return to CRSP ──────────────────────────────────────────────
crsp = crsp.merge(mkt_ret_map.reset_index(), on="date", how="left")
crsp["ar"] = crsp["ret"] - crsp["mkt_ret"]           # abnormal return

# Index by (permno, date) for fast slicing
crsp_idx = crsp.set_index(["permno", "date"]).sort_index()

# ── Compute CARs ──────────────────────────────────────────────────────────────
print("Computing CARs around rdq …")

SHORT_LO, SHORT_HI = -1,  1
LONG_LO,  LONG_HI  =  2, 60

records = []
for _, row in panel.iterrows():
    permno   = int(row["permno"])
    rdq      = row["rdq"]

    rec = {"gvkey": row["gvkey"], "permno": permno,
           "fyear": row["fyear"], "rdq": rdq}

    for label, lo, hi in [("CAR_short", SHORT_LO, SHORT_HI),
                           ("CAR_long",  LONG_LO,  LONG_HI)]:
        w_start = trading_day_offset(rdq, lo)
        w_end   = trading_day_offset(rdq, hi)
        if w_start is None or w_end is None:
            rec[label]   = np.nan
            rec[f"n_{label.split('_')[1]}"] = 0
            continue
        try:
            sub = crsp_idx.loc[(permno, slice(w_start, w_end)), "ar"]
            if len(sub) == 0:
                rec[label]   = np.nan
                rec[f"n_{label.split('_')[1]}"] = 0
            else:
                rec[label]   = float((1 + sub).prod() - 1)
                rec[f"n_{label.split('_')[1]}"] = len(sub)
        except KeyError:
            rec[label]   = np.nan
            rec[f"n_{label.split('_')[1]}"] = 0

    records.append(rec)

car = pd.DataFrame(records)

# ── Quality filter ────────────────────────────────────────────────────────────
car = car[car["n_short"] >= 2]
car = car[car["n_long"]  >= 10]

# Winsorize at 1/99%
for col in ["CAR_short", "CAR_long"]:
    lo_q, hi_q = car[col].quantile([0.01, 0.99])
    car[col] = car[col].clip(lo_q, hi_q)

print(f"\nFinal rows: {len(car):,}")
print(car[["CAR_short", "CAR_long"]].describe().round(4))

# ── Save ──────────────────────────────────────────────────────────────────────
out = BASE / "car_rdq.parquet"
car.to_parquet(out, index=False)
print(f"\nSaved → {out}")
