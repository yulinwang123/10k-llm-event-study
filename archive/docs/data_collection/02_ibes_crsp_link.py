"""
02_ibes_crsp_link.py
─────────────────────
Build an IBES ticker → CRSP PERMNO linkage table via 8-character CUSIP.

Why CUSIP and not pre-built tables:
    The WRDS pre-built ibes_crsp_link table has known coverage gaps.
    CUSIP matching via crsp.stocknames is the most reliable academic method
    (used by, e.g., Loughran & McDonald 2011).

Output: data/ibes_crsp_link.parquet
Columns:
    ticker    IBES ticker
    cusip     8-char CUSIP
    permno    CRSP PERMNO
    gvkey     Compustat GVKEY (via CCM linkage)
"""

import wrds
import pandas as pd
from config import WRDS_USERNAME, IBES_FILE, LINK_FILE

print("Connecting to WRDS …")
db = wrds.Connection(wrds_username=WRDS_USERNAME)

# ── Load IBES data (produced in step 01) ─────────────────────────────────────
ibes = pd.read_parquet(IBES_FILE, columns=["ticker", "cusip", "anndats"])
ibes_cusips = ibes["cusip"].dropna().unique().tolist()
print(f"Unique IBES CUSIPs to match: {len(ibes_cusips):,}")

# ── Pull CRSP stocknames (PERMNO ↔ CUSIP ↔ date range) ───────────────────────
# ncusip = 8-char historical CUSIP in CRSP
print("Pulling crsp.stocknames …")
stocknames = db.raw_sql("""
    SELECT permno, ncusip, namedt, nameendt, ticker AS crsp_ticker
    FROM crsp.stocknames
    WHERE ncusip IS NOT NULL
""", date_cols=["namedt", "nameendt"])

stocknames["ncusip"] = stocknames["ncusip"].str.strip().str.upper()
stocknames = stocknames[stocknames["ncusip"].isin(ibes_cusips)]
print(f"  Matched stocknames rows: {len(stocknames):,}")

# ── Merge IBES → CRSP via CUSIP, respecting date ranges ──────────────────────
# For each (ticker, anndats) in IBES, find the PERMNO active on that date.
ibes_unique = ibes[["ticker", "cusip", "anndats"]].drop_duplicates()

link = ibes_unique.merge(stocknames, left_on="cusip", right_on="ncusip", how="inner")

# Keep only rows where announcement date falls within stocknames validity window
link = link[
    (link["anndats"] >= link["namedt"]) &
    (link["anndats"] <= link["nameendt"].fillna(pd.Timestamp("2099-12-31")))
]

# If multiple PERMNOs match (rare), keep the one with the longest name window
link["window_len"] = (link["nameendt"].fillna(pd.Timestamp("2099-12-31")) - link["namedt"]).dt.days
link = (
    link
    .sort_values("window_len", ascending=False)
    .drop_duplicates(subset=["ticker", "anndats"], keep="first")
)

link = link[["ticker", "cusip", "permno", "anndats"]].copy()
print(f"  IBES→CRSP matched rows: {len(link):,}")

# ── Add Compustat GVKEY via CCM ───────────────────────────────────────────────
# crsp.ccmxpf_lnkhist: PERMNO ↔ GVKEY with date range
print("Pulling crsp.ccmxpf_lnkhist for GVKEY …")
ccm = db.raw_sql("""
    SELECT lpermno AS permno, gvkey, linkdt, linkenddt, linktype, linkprim
    FROM crsp.ccmxpf_lnkhist
    WHERE linktype IN ('LU', 'LC')
      AND linkprim IN ('P', 'C')
""", date_cols=["linkdt", "linkenddt"])

link = link.merge(ccm, on="permno", how="left")
link = link[
    (link["anndats"] >= link["linkdt"].fillna(pd.Timestamp("1900-01-01"))) &
    (link["anndats"] <= link["linkenddt"].fillna(pd.Timestamp("2099-12-31")))
]

# Keep one gvkey per (ticker, anndats)
link = link.drop_duplicates(subset=["ticker", "anndats"], keep="first")
link = link[["ticker", "cusip", "permno", "gvkey", "anndats"]].copy()

match_rate = link["permno"].notna().mean()
print(f"  PERMNO match rate: {match_rate:.1%}")
print(f"  GVKEY  match rate: {link['gvkey'].notna().mean():.1%}")
print(f"  Final link rows:   {len(link):,}")

# ── Save ──────────────────────────────────────────────────────────────────────
link.to_parquet(LINK_FILE, index=False)
print(f"Saved → {LINK_FILE}")

db.close()
