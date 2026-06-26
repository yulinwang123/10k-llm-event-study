"""
merge_event_panel.py
─────────────────────
Merge all data sources into the event-level analysis panel.

Inputs (must exist in data/):
  transcript_events.parquet  → CIQ: gvkey, call_date, transcriptid, text word-counts
  transcripts_raw.parquet    → text variables (prepared_text, qa_text)
  ibes_sue.parquet           → ticker, fpedats, anndats_act, sue, sue_win, numest
  crsp_car.parquet           → ticker, fpedats, anndats_act, permno, car_m1_p1, car_p2_p60
  firm_controls.parquet      → gvkey, datadate, size, btm, leverage, roa, eps_vol, ...
  io_pct.parquet             → permno, rdate, io_pct

Merge logic:
  1. Start from CIQ transcript events (one row per earnings call)
  2. Match to IBES on (ciq_ticker = ticker) + (fpedats ≈ call_date ± 45 days)
     Precise: take the IBES fiscal quarter whose anndats_act is within ±7 days of call_date
  3. Join CRSP CAR on (ticker, fpedats)
  4. Join firm controls on (gvkey, datadate = fpedats ± 15 days)
  5. Join 13F on (permno, quarter)

Output: data/event_panel.parquet
Key columns:
  gvkey, permno, ticker, transcriptid, call_date,
  fpedats, anndats_act,
  sue, sue_win, numest,          ← treatment + instrument
  car_m1_p1, car_p2_p60,        ← outcomes
  prepared_wordcount, qa_wordcount,
  size, btm, leverage, roa, loss, eps_vol, special_items_scaled,
  io_pct                         ← HTE moderators
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent / "data"

# ── Load inputs ───────────────────────────────────────────────────────────────
print("Loading inputs …")

events = pd.read_parquet(BASE / "transcript_events.parquet")
# Drop text from events (too large); use word-count columns only
transcripts = pd.read_parquet(BASE / "transcripts_raw.parquet",
                              columns=["transcriptid", "prepared_wordcount", "qa_wordcount"])
ibes  = pd.read_parquet(BASE / "ibes_sue.parquet")
cars  = pd.read_parquet(BASE / "crsp_car.parquet")
ctrl  = pd.read_parquet(BASE / "firm_controls.parquet")

# IO pct (optional)
io_path = BASE / "io_pct.parquet"
if io_path.exists():
    io = pd.read_parquet(io_path)
else:
    print("  WARNING: io_pct.parquet not found — io_pct will be NaN")
    io = None

print(f"  Events:       {len(events):,}")
print(f"  Transcripts:  {len(transcripts):,}")
print(f"  IBES SUE:     {len(ibes):,}")
print(f"  CRSP CAR:     {len(cars):,}")
print(f"  Firm controls:{len(ctrl):,}")

# ── Step 1: Attach word counts to events ─────────────────────────────────────
print("\nStep 1: attaching text word counts …")
events = events.merge(transcripts, on="transcriptid", how="left")

# ── Step 2: Match CIQ events to IBES on ticker + call date ───────────────────
# Strategy: sort-merge join
#   For each CIQ event (ticker, call_date), find the IBES row where
#   |anndats_act - call_date| ≤ 7 days AND fpedats is the most recent before call_date
print("\nStep 2: matching CIQ events to IBES …")

# Rename CIQ ticker column if needed
if "ciq_ticker" in events.columns:
    events = events.rename(columns={"ciq_ticker": "ticker"})

events["call_date"] = pd.to_datetime(events["call_date"])
ibes["anndats_act"] = pd.to_datetime(ibes["anndats_act"])
ibes["fpedats"]     = pd.to_datetime(ibes["fpedats"])

# Drop rows with null keys, then dedup
events = events.dropna(subset=["ticker", "call_date"])
events = events.drop_duplicates(subset=["ticker", "call_date"])

ibes = ibes.dropna(subset=["ticker", "anndats_act"])
ibes = ibes.drop_duplicates(subset=["ticker", "fpedats"])

# Match on ticker, then filter by |call_date - anndats_act| <= 7 days
# Approach: merge on ticker (inner), compute abs date diff, keep closest IBES row per event
ibes_cols = ["ticker", "anndats_act", "fpedats", "actual", "medest",
             "stdev", "numest", "sue", "sue_win"]
tmp = events.merge(ibes[ibes_cols], on="ticker", how="left")
tmp["date_diff"] = (tmp["call_date"] - tmp["anndats_act"]).abs()
tmp = tmp[tmp["date_diff"] <= pd.Timedelta("7 days")]
# Keep closest IBES match per event
tmp = tmp.sort_values("date_diff").drop_duplicates(subset=["ticker", "call_date"])
merged = events.merge(
    tmp[["ticker", "call_date"] + [c for c in ibes_cols if c != "ticker"]],
    on=["ticker", "call_date"],
    how="left"
)

n_matched = merged["sue"].notna().sum()
print(f"  CIQ events:         {len(events):,}")
print(f"  Matched to IBES:    {n_matched:,} ({n_matched/len(events):.1%})")

# ── Step 3: Attach CRSP CAR ──────────────────────────────────────────────────
print("\nStep 3: attaching CRSP CAR …")
cars["fpedats"]     = pd.to_datetime(cars["fpedats"])
cars["anndats_act"] = pd.to_datetime(cars["anndats_act"])

panel = merged.merge(
    cars[["ticker", "fpedats", "permno", "ret_0", "car_m1_p1", "car_p2_p60"]],
    on=["ticker", "fpedats"],
    how="left"
)
car_match = panel["car_m1_p1"].notna().sum()
print(f"  Events with CAR[-1,+1]:  {car_match:,} ({car_match/len(panel):.1%})")

# ── Step 4: Attach firm controls ─────────────────────────────────────────────
print("\nStep 4: attaching firm controls …")
ctrl["datadate"] = pd.to_datetime(ctrl["datadate"])

# Match on gvkey + datadate closest to fpedats (±45 days)
ctrl_cols = ["gvkey", "datadate", "fyearq", "fqtr",
             "size", "btm", "leverage", "roa", "loss",
             "eps_vol", "special_items_scaled", "mktcap"]
ctrl_clean = ctrl.dropna(subset=["gvkey", "datadate"])[ctrl_cols]

# Merge on gvkey, then filter by |fpedats - datadate| <= 45 days, keep closest
panel_clean = panel.dropna(subset=["gvkey", "fpedats"])
tmp_ctrl = panel_clean[["gvkey", "fpedats"]].merge(ctrl_clean, on="gvkey", how="left")
tmp_ctrl["date_diff"] = (tmp_ctrl["fpedats"] - tmp_ctrl["datadate"]).abs()
tmp_ctrl = tmp_ctrl[tmp_ctrl["date_diff"] <= pd.Timedelta("45 days")]
tmp_ctrl = tmp_ctrl.sort_values("date_diff").drop_duplicates(subset=["gvkey", "fpedats"])
tmp_ctrl = tmp_ctrl.drop(columns=["date_diff"])

panel_ctrl = panel.merge(
    tmp_ctrl,
    on=["gvkey", "fpedats"],
    how="left"
)
ctrl_match = panel_ctrl["size"].notna().sum()
print(f"  Events with controls: {ctrl_match:,} ({ctrl_match/len(panel_ctrl):.1%})")

# ── Step 5: Attach institutional ownership ────────────────────────────────────
if io is not None:
    print("\nStep 5: attaching 13F institutional ownership …")
    io["rdate"]  = pd.to_datetime(io["rdate"])
    io["year_q"] = io["rdate"].dt.to_period("Q")
    panel_ctrl["year_q"] = panel_ctrl["anndats_act"].dt.to_period("Q")

    panel_ctrl = panel_ctrl.merge(
        io[["permno", "year_q", "io_pct"]],
        on=["permno", "year_q"],
        how="left"
    )
    panel_ctrl = panel_ctrl.drop(columns=["year_q"])
    io_match = panel_ctrl["io_pct"].notna().sum()
    print(f"  Events with IO pct: {io_match:,}")
else:
    panel_ctrl["io_pct"] = np.nan

# ── Step 6: Clean & finalize ──────────────────────────────────────────────────
print("\nStep 6: cleaning panel …")

# Keep only events with at least SUE and CAR[-1,+1]
panel_final = panel_ctrl.dropna(subset=["sue", "car_m1_p1"]).copy()

# Winsorize CAR at 1%/99%
for col in ["car_m1_p1", "car_p2_p60", "ret_0"]:
    if col in panel_final.columns:
        p1 = panel_final[col].quantile(0.01)
        p99 = panel_final[col].quantile(0.99)
        panel_final[col + "_win"] = panel_final[col].clip(p1, p99)

# Key variable summary
keep_cols = [
    "gvkey", "permno", "ticker", "transcriptid",
    "companyname", "call_date", "keydevid",
    "fpedats", "anndats_act",
    # Treatment
    "sue", "sue_win", "actual", "medest", "stdev", "numest",
    # Outcomes
    "ret_0", "ret_0_win",
    "car_m1_p1", "car_m1_p1_win",
    "car_p2_p60", "car_p2_p60_win",
    # Text
    "prepared_wordcount", "qa_wordcount",
    # Controls
    "size", "btm", "leverage", "roa", "loss",
    "eps_vol", "special_items_scaled", "mktcap",
    # HTE
    "io_pct",
]
keep_cols = [c for c in keep_cols if c in panel_final.columns]
panel_final = panel_final[keep_cols]

print(f"\nFinal panel:")
print(f"  Rows (firm-quarters):   {len(panel_final):,}")
print(f"  Unique firms (gvkey):   {panel_final['gvkey'].nunique():,}")
print(f"  Date range:             {panel_final['call_date'].min().date()} → {panel_final['call_date'].max().date()}")
print(f"  SUE mean / std:         {panel_final['sue'].mean():.3f} / {panel_final['sue'].std():.3f}")
print(f"  CAR[-1,+1] mean / std:  {panel_final['car_m1_p1'].mean():.4f} / {panel_final['car_m1_p1'].std():.4f}")
print(f"  IO pct available:       {panel_final['io_pct'].notna().sum():,}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = BASE / "event_panel.parquet"
panel_final.to_parquet(out, index=False)
print(f"\nSaved → {out}")

# Quick column overview
print("\nColumn summary:")
print(panel_final.describe(percentiles=[.1,.25,.5,.75,.9]).round(4).to_string())
