"""
check_ciq_transcripts.py
─────────────────────────
Diagnostic: check CIQ transcript coverage on WRDS before committing
to a sample period.

Answers:
  1. Are CIQ transcript tables accessible under your subscription?
  2. What is the earliest transcript date?
  3. How many earnings call events per year?
  4. What fraction of our existing firms (from master_panel) are covered?

Run this FIRST before pulling any transcript data.
"""

import wrds
import pandas as pd
from pathlib import Path

BASE      = Path(__file__).parent.parent / "data"
WRDS_USER = "yulinwang"

db = wrds.Connection(wrds_username=WRDS_USER)

# ── 1. Check which CIQ schemas/tables exist ───────────────────────────────────
print("=" * 60)
print("1. Checking available CIQ schemas …")
try:
    schemas = db.raw_sql("""
        SELECT DISTINCT table_schema
        FROM information_schema.tables
        WHERE table_schema ILIKE '%ciq%'
           OR table_schema ILIKE '%transcript%'
        ORDER BY table_schema
    """)
    print(schemas.to_string(index=False))
except Exception as e:
    print(f"  Could not query schemas: {e}")

# ── 2. Check CIQ transcript tables ───────────────────────────────────────────
print()
print("2. Checking CIQ transcript tables …")
candidate_tables = [
    ("ciq", "wrds_transcript_detail"),
    ("ciq", "wrds_transcript_person"),
    ("ciq", "wrds_keydev"),
    ("ciq", "wrds_company"),
]
accessible = []
for schema, table in candidate_tables:
    try:
        db.raw_sql(f"SELECT * FROM {schema}.{table} LIMIT 1")
        print(f"  ✅ {schema}.{table}")
        accessible.append((schema, table))
    except Exception as e:
        print(f"  ❌ {schema}.{table} — {e}")

# ── 3. If accessible, check date range and coverage ──────────────────────────
if ("ciq", "wrds_keydev") in accessible:
    print()
    print("3. Earnings call event coverage (keydevtypeid=48) …")
    coverage = db.raw_sql("""
        SELECT
            EXTRACT(year FROM mostimportantdateutc) AS yr,
            COUNT(*) AS n_events
        FROM ciq.wrds_keydev
        WHERE keydevtypeid = 48
        GROUP BY yr
        ORDER BY yr
    """)
    print(coverage.to_string(index=False))
    print(f"\n  Total events: {coverage['n_events'].sum():,.0f}")
    print(f"  Earliest year: {coverage['yr'].min():.0f}")
    print(f"  Latest year:   {coverage['yr'].max():.0f}")

    # ── 4. Check overlap with our firm universe ───────────────────────────────
    print()
    print("4. Overlap with our master_panel firms …")
    master = pd.read_parquet(BASE / "master_panel.parquet",
                             columns=["gvkey", "ticker", "fyear"])
    our_tickers = master["ticker"].dropna().str.upper().unique().tolist()

    ciq_co = db.raw_sql("""
        SELECT companyid, tickersymbol
        FROM ciq.wrds_company
        WHERE tickersymbol IS NOT NULL
    """)
    ciq_tickers = ciq_co["tickersymbol"].str.upper().unique().tolist()

    overlap = set(our_tickers) & set(ciq_tickers)
    print(f"  Our firm universe:    {len(our_tickers):,} tickers")
    print(f"  CIQ company table:    {len(ciq_tickers):,} tickers")
    print(f"  Overlap:              {len(overlap):,} ({len(overlap)/len(our_tickers):.1%} of our universe)")

    # ── 5. Sample transcript text to check quality ────────────────────────────
    if ("ciq", "wrds_transcript_detail") in accessible:
        print()
        print("5. Sample transcript text (first 3 rows) …")
        sample = db.raw_sql("""
            SELECT d.keydevid, d.componenttypeid,
                   d.speakertypeid, LEFT(d.componenttext, 200) AS text_preview
            FROM ciq.wrds_transcript_detail d
            JOIN ciq.wrds_keydev k ON d.keydevid = k.keydevid
            WHERE k.keydevtypeid = 48
            LIMIT 3
        """)
        for _, row in sample.iterrows():
            print(f"\n  keydevid={row['keydevid']} "
                  f"type={row['componenttypeid']} "
                  f"speaker={row['speakertypeid']}")
            print(f"  {row['text_preview']}")

else:
    print()
    print("CIQ transcript tables not accessible.")
    print("Options:")
    print("  A) Request CIQ transcript access from your institution's WRDS admin")
    print("  B) Use Refinitiv Eikon (separate subscription)")
    print("  C) Scrape SeekingAlpha for pilot sample (200 events)")

db.close()
print()
print("=" * 60)
print("Done. Use the earliest year above to set IBES/CRSP START_DATE.")
