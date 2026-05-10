"""
compute_car_filing.py
─────────────────────
Computes CAR[-1,+1] and CAR[-3,+3] around the 10-K *filing date* (date_filed)
as a robustness check against the baseline which uses earnings announcement date (rdq).

Uses existing local CRSP data — no WRDS re-query needed.

Output:
    data/car_filing_date.parquet   — gvkey, fyear, car_filed_1_1, car_filed_3_3
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

DATA = "data"

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading panel...")
panel = pd.read_parquet(f"{DATA}/analysis_panel.parquet",
                        columns=["gvkey", "permno", "fyear", "date_filed", "rdq"])
panel["date_filed"] = pd.to_datetime(panel["date_filed"])
panel["permno"] = panel["permno"].astype(int)
panel = panel.dropna(subset=["date_filed", "permno"])
print(f"  {len(panel):,} firm-years with date_filed")

print("Loading CRSP daily returns...")
crsp = pd.read_parquet(f"{DATA}/crsp_daily.parquet", columns=["permno", "date", "ret"])
crsp["date"] = pd.to_datetime(crsp["date"])
crsp["permno"] = crsp["permno"].astype(int)
crsp = crsp.dropna(subset=["ret"])

print("Loading market returns...")
mkt = pd.read_parquet(f"{DATA}/crsp_market.parquet", columns=["date", "mkt_ret"])
mkt["date"] = pd.to_datetime(mkt["date"])

# Merge market return into daily
crsp = crsp.merge(mkt, on="date", how="left")
crsp["abret"] = crsp["ret"] - crsp["mkt_ret"]   # market-adjusted abnormal return

# Build a trading-day calendar for fast lookup
crsp_sorted = crsp.sort_values(["permno", "date"])

# ── Event window CAR computation ──────────────────────────────────────────────
def compute_car(permno, event_date, crsp_df, window):
    """
    Return CAR over trading days [−window, +window] around event_date.
    Finds actual trading days (not calendar days) relative to event_date.
    """
    sub = crsp_df[crsp_df["permno"] == permno].copy()
    if sub.empty:
        return np.nan

    # Find the index of the trading day on or after event_date
    sub = sub.reset_index(drop=True)
    idx = sub["date"].searchsorted(event_date)   # first date >= event_date

    # Clamp to valid range
    start = idx - window
    end   = idx + window + 1   # inclusive

    if start < 0 or end > len(sub):
        return np.nan

    return sub.iloc[start:end]["abret"].sum()


print("\nComputing CARs around date_filed ...")
results = []
permnos = set(crsp["permno"].unique())

# Group CRSP by permno for fast access
crsp_by_permno = {p: g.sort_values("date").reset_index(drop=True)
                  for p, g in crsp.groupby("permno")}

for _, row in tqdm(panel.iterrows(), total=len(panel)):
    permno     = int(row["permno"])
    event_date = row["date_filed"]

    if permno not in crsp_by_permno:
        results.append({"gvkey": row["gvkey"], "fyear": row["fyear"],
                        "car_filed_1_1": np.nan, "car_filed_3_3": np.nan})
        continue

    sub = crsp_by_permno[permno]
    idx = sub["date"].searchsorted(event_date)

    car_1_1 = np.nan
    car_3_3 = np.nan

    if 1 <= idx <= len(sub) - 2:
        car_1_1 = sub.iloc[idx-1 : idx+2]["abret"].sum()
    if 3 <= idx <= len(sub) - 4:
        car_3_3 = sub.iloc[idx-3 : idx+4]["abret"].sum()

    results.append({
        "gvkey":        row["gvkey"],
        "fyear":        row["fyear"],
        "car_filed_1_1": car_1_1,
        "car_filed_3_3": car_3_3,
    })

out = pd.DataFrame(results)
out["fyear"] = out["fyear"].astype(int)

# Summary
n_ok = out["car_filed_1_1"].notna().sum()
print(f"\nCAR computed for {n_ok:,} / {len(out):,} firm-years")
print(out[["car_filed_1_1", "car_filed_3_3"]].describe().round(4))

out.to_parquet(f"{DATA}/car_filing_date.parquet", index=False)
print(f"\n✓ Saved → {DATA}/car_filing_date.parquet")
