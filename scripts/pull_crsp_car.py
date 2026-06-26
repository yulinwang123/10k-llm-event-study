"""
pull_crsp_car.py
─────────────────
Compute CAR around earnings announcement dates.

Pipeline:
  1. Read ibes_sue.parquet (has ticker + anndats_act)
  2. Link IBES ticker → CRSP permno via iclink (IBES-CRSP link table)
     Fallback: via CCM (gvkey from wrds_gvkey → permno from ccmxpf_lnkhist)
  3. Pull daily returns for those permnos (batch by permno)
  4. Compute:
       CAR[-1,+1]  = announcement-window return (price reaction)
       CAR[+2,+60] = post-announcement drift (reversal / underreaction test)

Output: data/crsp_car.parquet
Columns:
  ticker, fpedats, anndats_act, permno,
  car_m1_p1, car_p2_p60, ret_0 (day-of return)
"""

import wrds
import numpy as np
import pandas as pd
from pathlib import Path

BASE      = Path(__file__).parent.parent / "data"
WRDS_USER = os.getenv("WRDS_USER", "your_wrds_username")

db = wrds.Connection(wrds_username=WRDS_USER)

# ── Step 1: Load IBES SUE (need ticker + anndats_act) ────────────────────────
print("Step 1: loading IBES SUE …")
ibes = pd.read_parquet(BASE / "ibes_sue.parquet",
                       columns=["ticker", "cusip", "fpedats", "anndats_act"])
print(f"  {len(ibes):,} firm-quarters")

# ── Step 2: Link IBES ticker → CRSP permno via WRDS iclink ───────────────────
# iclink is the standard IBES-CRSP linking table
print("\nStep 2: linking IBES ticker → CRSP permno via iclink …")
try:
    iclink = db.raw_sql("""
        SELECT ticker, permno, sdate, edate, score
        FROM crsp.iclink
        WHERE score <= 1      -- best match quality (0 = perfect, 1 = good)
    """)
    print(f"  iclink rows (score≤1): {len(iclink):,}")
    use_iclink = True
except Exception as e:
    print(f"  iclink unavailable ({e}), will use CCM fallback")
    use_iclink = False

if use_iclink:
    # For each IBES observation, find the permno valid on anndats_act
    ibes_link = ibes.merge(iclink, on="ticker", how="left")
    # Keep valid date ranges (sdate ≤ anndats_act ≤ edate; null edate = current)
    mask = (
        (ibes_link["sdate"].isna() | (ibes_link["anndats_act"] >= ibes_link["sdate"])) &
        (ibes_link["edate"].isna() | (ibes_link["anndats_act"] <= ibes_link["edate"]))
    )
    ibes_link = ibes_link[mask].copy()
    # If multiple matches, pick lowest score (best quality)
    ibes_link = (
        ibes_link.sort_values("score")
        .drop_duplicates(subset=["ticker", "fpedats"])
        .drop(columns=["sdate", "edate", "score"])
    )
    match_rate = ibes_link["permno"].notna().mean()
    print(f"  Match rate (iclink): {match_rate:.1%}")
else:
    # Fallback: load transcript_events which has gvkey, then CCM → permno
    print("  Loading CCM gvkey→permno link …")
    ccm = db.raw_sql("""
        SELECT gvkey, lpermno AS permno,
               linkdt, COALESCE(linkenddt, '2099-12-31'::date) AS linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
    """, date_cols=["linkdt", "linkenddt"])
    # Need ticker_events to get gvkey for each ticker
    # This path requires transcript_events.parquet to be present
    tevents = pd.read_parquet(BASE / "transcript_events.parquet",
                              columns=["ciq_ticker", "gvkey", "call_date"])
    tevents = tevents.rename(columns={"ciq_ticker": "ticker"})
    tevents_gvkey = tevents.drop_duplicates("ticker")[["ticker", "gvkey"]]
    ibes_link = ibes.merge(tevents_gvkey, on="ticker", how="left")
    ibes_link = ibes_link.merge(ccm, on="gvkey", how="left")
    mask = (
        (ibes_link["anndats_act"] >= ibes_link["linkdt"]) &
        (ibes_link["anndats_act"] <= ibes_link["linkenddt"])
    )
    ibes_link = (
        ibes_link[mask]
        .drop_duplicates(subset=["ticker", "fpedats"])
        .drop(columns=["linkdt", "linkenddt"], errors="ignore")
    )
    match_rate = ibes_link["permno"].notna().mean()
    print(f"  Match rate (CCM fallback): {match_rate:.1%}")

ibes_matched = ibes_link.dropna(subset=["permno"]).copy()
ibes_matched["permno"] = ibes_matched["permno"].astype(int)
print(f"  Matched firm-quarters: {len(ibes_matched):,}")

# ── Step 3: Pull CRSP daily returns (batched by permno) ──────────────────────
print("\nStep 3: pulling CRSP daily returns …")
permnos = ibes_matched["permno"].unique().tolist()
min_date = ibes_matched["anndats_act"].min() - pd.Timedelta(days=90)
max_date = ibes_matched["anndats_act"].max() + pd.Timedelta(days=90)

BATCH = 500
crsp_chunks = []
for i in range(0, len(permnos), BATCH):
    batch = permnos[i:i+BATCH]
    ids_str = ",".join(str(p) for p in batch)
    chunk = db.raw_sql(f"""
        SELECT d.permno, d.date, d.ret, d.shrout, d.prc,
               m.vwretd AS mkt_ret
        FROM crsp.dsf d
        LEFT JOIN crsp.dsi m ON d.date = m.date
        WHERE d.permno IN ({ids_str})
          AND d.date BETWEEN '{min_date.date()}' AND '{max_date.date()}'
          AND d.ret IS NOT NULL
    """, date_cols=["date"])
    crsp_chunks.append(chunk)
    if (i // BATCH) % 5 == 0:
        print(f"  Batch {i//BATCH+1}/{(len(permnos)//BATCH)+1}")

crsp = pd.concat(crsp_chunks, ignore_index=True)
crsp = crsp.sort_values(["permno", "date"])
print(f"  CRSP daily rows pulled: {len(crsp):,}")

# ── Step 4: Compute CAR for each event ───────────────────────────────────────
print("\nStep 4: computing CARs …")

def get_trading_dates(dates_series):
    """Return sorted unique trading date index."""
    return sorted(dates_series.unique())

def compute_car(event_date, permno, crsp_sub, window_start, window_end):
    """
    Compute CAR from window_start to window_end (relative to event_date=0).
    Uses market-adjusted returns: AR_t = ret_t - mkt_ret_t
    """
    # Get trading dates around event
    sub = crsp_sub[crsp_sub["permno"] == permno].sort_values("date")
    if sub.empty:
        return np.nan

    dates = sub["date"].values
    idx = np.searchsorted(dates, np.datetime64(event_date))
    if idx >= len(dates):
        return np.nan

    # Collect returns for the window
    start_idx = idx + window_start
    end_idx   = idx + window_end + 1

    if start_idx < 0:
        start_idx = 0
    if end_idx > len(dates):
        return np.nan

    window_rows = sub.iloc[start_idx:end_idx]
    if len(window_rows) < abs(window_end - window_start):  # too few trading days
        return np.nan

    ar = window_rows["ret"] - window_rows["mkt_ret"].fillna(0)
    car = (1 + ar).prod() - 1
    return car

# Build per-permno CRSP index for fast lookup
crsp_by_permno = {p: g.reset_index(drop=True) for p, g in crsp.groupby("permno")}

results = []
for _, row in ibes_matched.iterrows():
    permno    = int(row["permno"])
    event_dt  = row["anndats_act"]

    if permno not in crsp_by_permno:
        continue

    sub = crsp_by_permno[permno]
    dates_np = pd.DatetimeIndex(sub["date"].values)
    idx = dates_np.searchsorted(event_dt)

    def car_window(w_start, w_end):
        s = idx + w_start
        e = idx + w_end + 1
        if s < 0 or e > len(sub):
            return np.nan
        rows = sub.iloc[s:e]
        required_days = w_end - w_start + 1
        if len(rows) < required_days - 1:   # allow 1 missing day
            return np.nan
        ar = rows["ret"].values - rows["mkt_ret"].fillna(0).values
        return float((1 + ar).prod() - 1)

    # Day-0 return (earnings call day itself)
    ret_0 = float(sub.iloc[idx]["ret"]) if idx < len(sub) else np.nan

    results.append({
        "ticker":      row["ticker"],
        "fpedats":     row["fpedats"],
        "anndats_act": event_dt,
        "permno":      permno,
        "ret_0":       ret_0,
        "car_m1_p1":   car_window(-1,  1),
        "car_p2_p60":  car_window( 2, 60),
    })

car_df = pd.DataFrame(results)
print(f"  Events with CAR[-1,+1]: {car_df['car_m1_p1'].notna().sum():,}")
print(f"  Events with CAR[+2,+60]: {car_df['car_p2_p60'].notna().sum():,}")
print(f"\n  CAR[-1,+1] stats:")
print(f"  {car_df['car_m1_p1'].describe().round(4).to_string()}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = BASE / "crsp_car.parquet"
car_df.to_parquet(out, index=False)
print(f"\nSaved → {out}  ({len(car_df):,} rows)")

db.close()
