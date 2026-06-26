"""
pull_transcripts.py
────────────────────
Pull earnings call transcripts from CIQ via WRDS.

Pipeline:
  wrds_transcript_detail  → earnings call events (keydeveventtypeid=48)
  ciqtranscriptcomponent  → full text by transcriptid
  wrds_gvkey              → companyid → GVKEY

Sample period: 2010-2023
  - CIQ transcript coverage is sparse before 2010
  - 2024 data incomplete (fiscal year not yet closed for many firms)

Output: data/transcripts_raw.parquet
Columns:
  gvkey, companyid, companyname, keydevid, transcriptid,
  call_date, prepared_text, qa_text, full_text,
  prepared_wordcount, qa_wordcount
"""

import wrds
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE      = Path(__file__).parent.parent / "data"
WRDS_USER = os.getenv("WRDS_USER", "your_wrds_username")
START     = "2010-01-01"   # CIQ coverage sparse before 2010
END       = "2023-12-31"   # full fiscal year coverage

BATCH   = 500   # transcripts per query (smaller = more parallel-friendly)
WORKERS = 8     # parallel WRDS connections

db = wrds.Connection(wrds_username=WRDS_USER)

# ── Step 1: Earnings call events with company + date ─────────────────────────
# keydeveventtypeid = 48 → Earnings Calls
# Use transcriptpresentationtypeid = 5 (Final) to avoid duplicates
# (preliminary/spellchecked versions have same keydevid but different transcriptid)
print("Step 1: pulling earnings call events …")
events = db.raw_sql(f"""
    SELECT DISTINCT
        d.companyid,
        d.companyname,
        d.keydevid,
        d.transcriptid,
        d.mostimportantdateutc::date AS call_date,
        d.headline
    FROM ciq.wrds_transcript_detail d
    WHERE d.keydeveventtypeid = 48
      AND d.mostimportantdateutc::date BETWEEN '{START}' AND '{END}'
      AND d.companyid IS NOT NULL
      AND d.transcriptpresentationtypeid = 5
""", date_cols=["call_date"])

print(f"  Earnings call events (Final transcripts): {len(events):,}")
print(f"  Date range: {events['call_date'].min()} → {events['call_date'].max()}")
print(f"  Unique companies: {events['companyid'].nunique():,}")

# ── Step 2: Link companyid → GVKEY + IBES ticker ─────────────────────────────
print("\nStep 2: linking companyid → GVKEY + ticker …")
gvkey_map = db.raw_sql("""
    SELECT DISTINCT ON (companyid) companyid, gvkey
    FROM ciq.wrds_gvkey
    WHERE primaryflag = 1
    ORDER BY companyid, enddate DESC NULLS FIRST
""")
ticker_map = db.raw_sql("""
    SELECT DISTINCT ON (companyid) companyid, ticker AS ciq_ticker
    FROM ciq.wrds_ticker
    WHERE primaryflag = 1
    ORDER BY companyid, enddate DESC NULLS FIRST
""")
events = events.merge(gvkey_map, on="companyid", how="left")
events = events.merge(ticker_map, on="companyid", how="left")
match_rate = events["gvkey"].notna().mean()
print(f"  GVKEY match rate:  {match_rate:.1%} ({events['gvkey'].notna().sum():,} / {len(events):,})")
print(f"  Ticker match rate: {events['ciq_ticker'].notna().mean():.1%}")

# Keep only events with GVKEY (Compustat coverage)
events_linked = events.dropna(subset=["gvkey"])
print(f"  Events with GVKEY: {len(events_linked):,}")

# Save event list
events_linked.to_parquet(BASE / "transcript_events.parquet", index=False)
print(f"  Saved transcript_events.parquet")

# ── Step 3: Pull transcript text ──────────────────────────────────────────────
# transcriptcomponenttypeid:
#   1 = Operator/Presentation header (skip)
#   2 = Presenter speech  → PREPARED REMARKS (management)
#   3 = Q&A Analyst       → Q&A
#   4 = Q&A Management    → Q&A
#   5, 6, 7 = other system messages (skip)

print("\nStep 3: pulling transcript text by transcriptid batches …")
transcript_ids = events_linked["transcriptid"].dropna().astype(int).unique().tolist()
BATCH      = 500    # transcripts per SQL query
FLUSH_EVERY = 100   # aggregate + write to disk every N batches (~50K transcripts)
n_batches  = (len(transcript_ids) + BATCH - 1) // BATCH
tmp_dir    = BASE / "transcripts_tmp"
tmp_dir.mkdir(exist_ok=True)
print(f"  Total transcripts: {len(transcript_ids):,}  |  batches: {n_batches}  |  flush every {FLUSH_EVERY}")

def aggregate_and_save(raw_chunks, chunk_idx):
    """Aggregate a list of raw component DataFrames and save one parquet chunk."""
    if not raw_chunks:
        return
    raw = pd.concat(raw_chunks, ignore_index=True)
    raw = raw.sort_values(["transcriptid", "componentorder"])

    prepared = (
        raw[raw["transcriptcomponenttypeid"] == 2]
        .groupby("transcriptid")["componenttext"]
        .apply(lambda x: " ".join(x.dropna()))
        .rename("prepared_text").reset_index()
    )
    qa = (
        raw[raw["transcriptcomponenttypeid"].isin([3, 4])]
        .groupby("transcriptid")["componenttext"]
        .apply(lambda x: " ".join(x.dropna()))
        .rename("qa_text").reset_index()
    )
    out = prepared.merge(qa, on="transcriptid", how="outer")
    out.to_parquet(tmp_dir / f"chunk_{chunk_idx:04d}.parquet", index=False)
    return len(out)

chunks = []
chunk_idx = 0
for i in range(0, len(transcript_ids), BATCH):
    batch = transcript_ids[i:i+BATCH]
    ids_str = ",".join(str(t) for t in batch)
    try:
        chunk = db.raw_sql(f"""
            SELECT transcriptid, transcriptcomponenttypeid,
                   componentorder, componenttext
            FROM ciq.ciqtranscriptcomponent
            WHERE transcriptid IN ({ids_str})
              AND transcriptcomponenttypeid IN (2, 3, 4)
        """)
        chunks.append(chunk)
    except Exception as e:
        print(f"  Batch {i//BATCH+1} error: {e}")

    batch_num = i // BATCH + 1
    if batch_num % FLUSH_EVERY == 0 or batch_num == n_batches:
        n_saved = aggregate_and_save(chunks, chunk_idx)
        pct = batch_num / n_batches * 100
        print(f"  {batch_num}/{n_batches} ({pct:.0f}%) — chunk {chunk_idx} saved ({n_saved:,} transcripts)")
        chunks = []   # free memory
        chunk_idx += 1

# ── Step 4: Combine chunks — keep word counts only (text stays in chunk files) ─
# Full text is in data/transcripts_tmp/chunk_XXXX.parquet for compute_text_features.py
# Here we only extract word counts to avoid OOM.
print("\nStep 4: extracting word counts from chunks (text stays on disk) …")
all_chunk_files = sorted(tmp_dir.glob("chunk_*.parquet"))
print(f"  {len(all_chunk_files)} chunk files found")

wc_parts = []
for f in all_chunk_files:
    chunk = pd.read_parquet(f)   # one chunk at a time (~300MB)
    chunk["prepared_wordcount"] = (
        chunk["prepared_text"].str.split().str.len().fillna(0).astype(int)
    )
    chunk["qa_wordcount"] = (
        chunk["qa_text"].str.split().str.len().fillna(0).astype(int)
    )
    # Keep only IDs + word counts — drop raw text immediately
    wc_parts.append(chunk[["transcriptid", "prepared_wordcount", "qa_wordcount"]])
    del chunk   # free memory

wc = pd.concat(wc_parts, ignore_index=True)
print(f"  Word counts extracted: {len(wc):,} transcripts")

# Merge back to events
result = events_linked.merge(wc, on="transcriptid", how="left")
result = result[result["prepared_wordcount"].fillna(0) + result["qa_wordcount"].fillna(0) > 50]

print(f"  Final transcripts: {len(result):,}")
print(f"  Median prepared words: {result['prepared_wordcount'].median():.0f}")
print(f"  Median Q&A words:      {result['qa_wordcount'].median():.0f}")
print(f"  Date range: {result['call_date'].min()} → {result['call_date'].max()}")

# ── Save ──────────────────────────────────────────────────────────────────────
# transcripts_raw.parquet = event metadata + word counts (no full text)
# Full text lives in data/transcripts_tmp/ for compute_text_features.py
out = BASE / "transcripts_raw.parquet"
result.to_parquet(out, index=False)
print(f"\nSaved → {out}  (word counts only; full text in transcripts_tmp/)")

db.close()
