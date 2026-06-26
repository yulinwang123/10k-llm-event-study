"""
03_pull_crsp_car.py
────────────────────
Pull CRSP daily returns for matched PERMNOs and compute:
    CAR_short : market-adjusted CAR over [-1, +1] around announcement date
    CAR_long  : market-adjusted CAR over [+2, +60] (post-announcement drift)

Strategy (memory-efficient):
    1. Identify all (permno, anndats) pairs from the link table.
    2. For each permno pull returns only within [anndats - buffer, anndats + buffer].
    3. Compute CARs using the CRSP value-weighted market return (vwretd) from dsi.

Output: data/car_panel.parquet
Columns:
    permno, anndats, CAR_short, CAR_long, n_obs_short, n_obs_long
"""

import wrds
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import (
    WRDS_USERNAME, LINK_FILE, CAR_FILE,
    CAR_SHORT_WINDOW, CAR_LONG_WINDOW, CRSP_BUFFER_DAYS, START_DATE, END_DATE
)

print("Connecting to WRDS …")
db = wrds.Connection(wrds_username=WRDS_USERNAME)

# ── Load link table ───────────────────────────────────────────────────────────
link = pd.read_parquet(LINK_FILE)
link = link.dropna(subset=["permno"])
link["permno"] = link["permno"].astype(int)
events = link[["permno", "anndats"]].drop_duplicates()
print(f"Events to process: {len(events):,} (unique permno-anndats pairs)")

# ── Pull CRSP market index returns ───────────────────────────────────────────
print("Pulling crsp.dsi (market index returns) …")
dsi = db.raw_sql(f"""
    SELECT date, vwretd
    FROM crsp.dsi
    WHERE date BETWEEN '{START_DATE}'
      AND date_add('{END_DATE}', INTERVAL {CRSP_BUFFER_DAYS} DAY)
""", date_cols=["date"])
dsi = dsi.set_index("date")["vwretd"].rename("mkt_ret")
print(f"  Market return rows: {len(dsi):,}")

# ── Build trading-day calendar ─────────────────────────────────────────────
# Used to map event-relative offsets to actual calendar dates
trading_days = pd.Series(dsi.index.sort_values(), name="date")
td_to_idx = {d: i for i, d in enumerate(trading_days)}

def get_event_window(ann_date, lo, hi):
    """Return (start_date, end_date) for event window [lo, hi] trading days
    around ann_date. Returns (None, None) if ann_date not a trading day."""
    # Snap to nearest trading day on or after ann_date
    idx_list = trading_days.searchsorted(ann_date)
    if idx_list >= len(trading_days):
        return None, None
    base_idx = idx_list
    start_idx = base_idx + lo
    end_idx   = base_idx + hi
    if start_idx < 0 or end_idx >= len(trading_days):
        return None, None
    return trading_days.iloc[start_idx], trading_days.iloc[end_idx]

# ── Pull CRSP daily returns in batches ───────────────────────────────────────
# Batch by date range to avoid pulling all of dsf (huge)
permno_list = events["permno"].unique().tolist()

# Pull in annual chunks to keep query size manageable
years = range(
    pd.Timestamp(START_DATE).year,
    pd.Timestamp(END_DATE).year + 1
)

print("Pulling crsp.dsf in annual chunks …")
dsf_chunks = []
for yr in years:
    yr_start = f"{yr}-01-01"
    yr_end   = f"{yr}-12-31"
    # Chunk permno list to avoid SQL IN clause overflow (>10k values)
    permno_chunks = [permno_list[i:i+5000] for i in range(0, len(permno_list), 5000)]
    for pc in permno_chunks:
        permnos_str = ",".join(str(p) for p in pc)
        chunk = db.raw_sql(f"""
            SELECT permno, date, ret
            FROM crsp.dsf
            WHERE permno IN ({permnos_str})
              AND date BETWEEN '{yr_start}' AND '{yr_end}'
              AND ret IS NOT NULL
        """, date_cols=["date"])
        dsf_chunks.append(chunk)
    print(f"  {yr} done")

dsf = pd.concat(dsf_chunks, ignore_index=True)
dsf = dsf.sort_values(["permno", "date"])
print(f"  Total CRSP daily rows pulled: {len(dsf):,}")

# ── Add market return ──────────────────────────────────────────────────────────
dsf = dsf.merge(dsi.reset_index(), on="date", how="left")
dsf["ar"] = dsf["ret"] - dsf["mkt_ret"]   # abnormal return (market-adjusted)

# Index by (permno, date) for fast lookup
dsf_idx = dsf.set_index(["permno", "date"])

# ── Compute CARs for each event ───────────────────────────────────────────────
print("Computing CARs …")
records = []

for _, row in tqdm(events.iterrows(), total=len(events)):
    permno   = int(row["permno"])
    ann_date = row["anndats"]

    result = {"permno": permno, "anndats": ann_date}

    for label, (lo, hi) in [("CAR_short", CAR_SHORT_WINDOW),
                              ("CAR_long",  CAR_LONG_WINDOW)]:
        w_start, w_end = get_event_window(ann_date, lo, hi)
        if w_start is None:
            result[label]            = np.nan
            result[f"n_obs_{label}"] = 0
            continue

        try:
            sub = dsf_idx.loc[(permno, slice(w_start, w_end)), "ar"]
            if len(sub) == 0:
                result[label]            = np.nan
                result[f"n_obs_{label}"] = 0
            else:
                # Compound abnormal returns: CAR = ∏(1+AR_t) - 1
                result[label]            = (1 + sub).prod() - 1
                result[f"n_obs_{label}"] = len(sub)
        except KeyError:
            result[label]            = np.nan
            result[f"n_obs_{label}"] = 0

    records.append(result)

car = pd.DataFrame(records)

# ── Quality filters ────────────────────────────────────────────────────────────
# Require at least 2 trading days of data in each window
short_days = CAR_SHORT_WINDOW[1] - CAR_SHORT_WINDOW[0] + 1
long_days  = CAR_LONG_WINDOW[1]  - CAR_LONG_WINDOW[0]  + 1

car = car[
    (car["n_obs_CAR_short"] >= max(2, short_days - 1)) &
    (car["n_obs_CAR_long"]  >= max(10, long_days  - 5))
]

# Winsorize CARs at 1/99%
for col in ["CAR_short", "CAR_long"]:
    lo, hi = car[col].quantile([0.01, 0.99])
    car[col] = car[col].clip(lo, hi)

print(f"Final CAR rows: {len(car):,}")
print(car[["CAR_short", "CAR_long"]].describe().round(4))

# ── Save ──────────────────────────────────────────────────────────────────────
car.to_parquet(CAR_FILE, index=False)
print(f"Saved → {CAR_FILE}")

db.close()
