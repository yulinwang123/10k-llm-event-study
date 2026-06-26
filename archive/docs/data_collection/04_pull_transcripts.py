"""
04_pull_transcripts.py
───────────────────────
Pull earnings call transcripts.

PRIMARY:   WRDS S&P Capital IQ transcript tables (ciq schema)
FALLBACK:  SeekingAlpha public transcript pages (web scrape)

The script first checks whether ciq transcript tables are accessible on your
WRDS subscription. If yes, it pulls via SQL. If not, it falls back to
SeekingAlpha scraping for a smaller pilot sample.

Output: data/transcripts.parquet
Columns:
    permno        CRSP PERMNO
    anndats       announcement date (matched from link table)
    ticker        IBES/company ticker
    prepared_text full text of prepared remarks section
    qa_text       full text of Q&A section
    full_text     prepared_text + qa_text concatenated
"""

import wrds
import pandas as pd
import numpy as np
import time
import re
import os
import requests
from bs4 import BeautifulSoup
from config import WRDS_USERNAME, LINK_FILE, TRANSCRIPT_FILE

# ── Load link table to know which (ticker, anndats) we need ──────────────────
link = pd.read_parquet(LINK_FILE)
link = link.dropna(subset=["permno"]).copy()
link["permno"] = link["permno"].astype(int)
needed = link[["permno", "ticker", "anndats", "gvkey"]].drop_duplicates()
print(f"Need transcripts for {len(needed):,} events")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD A: WRDS Capital IQ Transcripts
# ══════════════════════════════════════════════════════════════════════════════

def pull_via_wrds_ciq(db, needed):
    """
    Pull transcripts from WRDS Capital IQ.

    CIQ transcript tables on WRDS (if your institution subscribes):
        ciq.wrds_transcript_person   - speaker metadata
        ciq.wrds_transcript_detail   - transcript text by component/speaker
    Linked via keydevid (the earnings call event ID).

    Company linkage: ciq.wrds_company (companyid ↔ ticker/CUSIP)
    """
    print("Checking CIQ transcript table availability …")
    try:
        # Quick schema check
        test = db.raw_sql("SELECT * FROM ciq.wrds_transcript_detail LIMIT 1")
        print("  CIQ transcript tables accessible.")
    except Exception as e:
        print(f"  CIQ tables NOT accessible: {e}")
        return None

    # Pull company-to-ticker mapping from CIQ
    print("  Pulling CIQ company map …")
    ciq_co = db.raw_sql("""
        SELECT companyid, tickersymbol AS ciq_ticker, primaryexchangeid
        FROM ciq.wrds_company
        WHERE tickersymbol IS NOT NULL
    """)

    # Match CIQ company to our ticker list
    our_tickers = needed["ticker"].str.upper().unique().tolist()
    ciq_co = ciq_co[ciq_co["ciq_ticker"].str.upper().isin(our_tickers)]
    ciq_co_ids = ciq_co["companyid"].tolist()
    if not ciq_co_ids:
        print("  No CIQ company matches found.")
        return None

    print(f"  Matched {len(ciq_co_ids)} CIQ companies")

    # Pull earnings call events (keydevtypeid=48 = Earnings Call)
    print("  Pulling CIQ earnings call events …")
    ids_str = ",".join(str(i) for i in ciq_co_ids[:5000])
    events_ciq = db.raw_sql(f"""
        SELECT keydevid, companyid,
               mostimportantdateutc::date AS event_date,
               headline
        FROM ciq.wrds_keydev
        WHERE companyid IN ({ids_str})
          AND keydevtypeid = 48
          AND mostimportantdateutc::date BETWEEN '2010-01-01' AND '2023-12-31'
    """, date_cols=["event_date"])
    print(f"  Earnings call events: {len(events_ciq):,}")

    # Pull transcript text
    print("  Pulling transcript text (this may take a while) …")
    keydev_ids = events_ciq["keydevid"].tolist()

    text_chunks = []
    batch_size = 500
    for i in range(0, len(keydev_ids), batch_size):
        batch = keydev_ids[i:i+batch_size]
        ids_str = ",".join(str(k) for k in batch)
        chunk = db.raw_sql(f"""
            SELECT keydevid, componentorder, componenttypeid,
                   speakertypeid, transcriptpersonname, componenttext
            FROM ciq.wrds_transcript_detail
            WHERE keydevid IN ({ids_str})
        """)
        text_chunks.append(chunk)
        if (i // batch_size) % 10 == 0:
            print(f"    Batch {i//batch_size + 1}/{(len(keydev_ids)//batch_size)+1} done")

    text = pd.concat(text_chunks, ignore_index=True)
    print(f"  Total transcript components: {len(text):,}")

    # ── Parse into prepared remarks vs Q&A ───────────────────────────────────
    # componenttypeid: 1=Presentation/Prepared, 2=Q&A, 3=Other
    # speakertypeid:   1=Operator/Moderator, 2=Company, 3=Analyst
    prepared = (
        text[text["componenttypeid"] == 1]
        .groupby("keydevid")["componenttext"]
        .apply(lambda x: " ".join(x.dropna()))
        .rename("prepared_text")
    )
    qa = (
        text[text["componenttypeid"] == 2]
        .groupby("keydevid")["componenttext"]
        .apply(lambda x: " ".join(x.dropna()))
        .rename("qa_text")
    )

    transcripts = (
        events_ciq[["keydevid", "companyid", "event_date"]]
        .merge(prepared, on="keydevid", how="left")
        .merge(qa, on="keydevid", how="left")
        .merge(ciq_co[["companyid", "ciq_ticker"]], on="companyid", how="left")
    )
    transcripts["full_text"] = (
        transcripts["prepared_text"].fillna("") + " " +
        transcripts["qa_text"].fillna("")
    ).str.strip()

    # Link to PERMNO via ticker + event_date
    transcripts = transcripts.rename(columns={
        "ciq_ticker": "ticker",
        "event_date": "anndats"
    })
    needed_upper = needed.copy()
    needed_upper["ticker_upper"] = needed_upper["ticker"].str.upper()
    transcripts["ticker_upper"] = transcripts["ticker"].str.upper()

    result = transcripts.merge(
        needed_upper[["permno", "ticker_upper", "anndats"]],
        on=["ticker_upper", "anndats"],
        how="inner"
    )
    result = result[["permno", "anndats", "ticker",
                      "prepared_text", "qa_text", "full_text"]].copy()
    print(f"  Matched transcripts: {len(result):,}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# METHOD B: SeekingAlpha scrape (pilot / fallback)
# ══════════════════════════════════════════════════════════════════════════════

def scrape_seekingalpha(needed, max_events=200, delay=2.0):
    """
    Scrape earnings call transcripts from SeekingAlpha.
    Only used as a fallback for pilot analysis (max_events).
    Full production pipeline should use CIQ or purchase a transcript feed.
    """
    print(f"Falling back to SeekingAlpha scrape (pilot, max {max_events} events) …")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # SeekingAlpha search URL for earnings call transcripts
    BASE_SEARCH = "https://seekingalpha.com/symbol/{ticker}/earnings/transcripts"

    sample = needed.head(max_events).copy()
    records = []

    for _, row in sample.iterrows():
        ticker  = row["ticker"]
        permno  = row["permno"]
        anndate = row["anndats"]

        url = BASE_SEARCH.format(ticker=ticker)
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            # Find transcript links near the announcement date
            # SeekingAlpha lists transcripts with dates — find the closest one
            links = soup.find_all("a", href=re.compile(r"/article/\d+.*transcript"))
            if not links:
                continue

            # Grab the most recent transcript (first link)
            transcript_url = "https://seekingalpha.com" + links[0]["href"]
            time.sleep(delay)

            tresp = requests.get(transcript_url, headers=headers, timeout=15)
            if tresp.status_code != 200:
                continue

            tsoup = BeautifulSoup(tresp.text, "html.parser")

            # Extract full article text
            article = tsoup.find("div", {"data-test-id": "article-content"})
            if article is None:
                article = tsoup.find("article")
            if article is None:
                continue

            full_text = article.get_text(separator=" ", strip=True)

            # Rough split into prepared remarks and Q&A
            qa_split = re.split(r"(?i)question.and.answer|Q&A Session", full_text, maxsplit=1)
            prepared_text = qa_split[0].strip()
            qa_text       = qa_split[1].strip() if len(qa_split) > 1 else ""

            records.append({
                "permno":        permno,
                "anndats":       anndate,
                "ticker":        ticker,
                "prepared_text": prepared_text,
                "qa_text":       qa_text,
                "full_text":     full_text,
                "source":        "seekingalpha"
            })
            print(f"  Scraped {ticker} ({anndate.date() if hasattr(anndate,'date') else anndate})")

        except Exception as e:
            print(f"  Error scraping {ticker}: {e}")
            continue

        time.sleep(delay)

    print(f"  Successfully scraped: {len(records)} transcripts")
    return pd.DataFrame(records) if records else None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

print("Connecting to WRDS …")
db = wrds.Connection(wrds_username=WRDS_USERNAME)

transcripts = pull_via_wrds_ciq(db, needed)

if transcripts is None or len(transcripts) == 0:
    print("\nCIQ pull failed or returned nothing. Switching to SeekingAlpha …")
    transcripts = scrape_seekingalpha(needed, max_events=200)

db.close()

if transcripts is not None and len(transcripts) > 0:
    # Basic text cleaning
    for col in ["prepared_text", "qa_text", "full_text"]:
        if col in transcripts.columns:
            transcripts[col] = (
                transcripts[col]
                .fillna("")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    # Filter out near-empty transcripts
    transcripts = transcripts[transcripts["full_text"].str.len() > 500]

    print(f"\nFinal transcript rows: {len(transcripts):,}")
    print(f"Median full_text length: {transcripts['full_text'].str.len().median():.0f} chars")

    transcripts.to_parquet(TRANSCRIPT_FILE, index=False)
    print(f"Saved → {TRANSCRIPT_FILE}")
else:
    print("No transcripts obtained. Check WRDS subscription or scraper.")
