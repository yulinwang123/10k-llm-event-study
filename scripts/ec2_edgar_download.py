"""
ec2_edgar_download.py
─────────────────────
Multi-threaded SEC EDGAR 10-K downloader for AWS EC2.
Downloads MD&A sections for S&P 1500 × FY2010–2020 and writes to S3.

Usage:
    # Test mode (5 companies, 2 years) — run locally first to verify
    python ec2_edgar_download.py --test --bucket your-bucket-name

    # Full run on EC2
    python ec2_edgar_download.py --bucket your-bucket-name --workers 16

Requirements:
    pip install boto3 requests beautifulsoup4 lxml tqdm pandas pyarrow
"""

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────
HEADERS = {"User-Agent": "YulinWang yulinwang@uchicago.edu"}  # SEC requires this
BASE_SUBMISSIONS = "https://data.sec.gov/submissions"
BASE_ARCHIVE     = "https://www.sec.gov/Archives/edgar/data"
MAX_RATE         = 8          # requests per second (SEC limit is 10, stay under)
RATE_LOCK        = threading.Lock()
_last_request_time = [0.0]   # mutable for thread-safe rate limiting

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",   required=True,  help="S3 bucket name")
    p.add_argument("--prefix",   default="10k-project", help="S3 key prefix")
    p.add_argument("--workers",  type=int, default=8, help="Thread pool size")
    p.add_argument("--years",    default="2010-2020", help="Year range, e.g. 2010-2020")
    p.add_argument("--test",     action="store_true",
                   help="Test mode: 5 companies, 2 years (2018-2019)")
    p.add_argument("--universe", default=None,
                   help="Path to sp1500_universe.csv (downloaded by local_wrds_pull.py)")
    return p.parse_args()


# ── Rate-limited request ───────────────────────────────────────────────────────
def rate_limited_get(url: str, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        with RATE_LOCK:
            elapsed = time.time() - _last_request_time[0]
            wait    = (1.0 / MAX_RATE) - elapsed
            if wait > 0:
                time.sleep(wait)
            _last_request_time[0] = time.time()
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


# ── EDGAR filing metadata ──────────────────────────────────────────────────────
def _parse_filing_block(block: dict, cik: str, start_year: int, end_year: int) -> list[dict]:
    """Parse one filing block (recent or historical) into a list of 10-K dicts."""
    forms   = block.get("form", [])
    dates   = block.get("filingDate", [])
    accnos  = block.get("accessionNumber", [])
    docs    = block.get("primaryDocument", [])
    reports = block.get("reportDate", [])  # fiscal year end date (more accurate than filingDate)

    results = []
    for form, date_filed, acc, doc, report_date in zip(forms, dates, accnos, docs, reports
                                                        if reports else [""] * len(forms)):
        if form not in ("10-K", "10-K405"):
            continue
        # Use reportDate year (fiscal year) if available, else fall back to filingDate year
        if report_date and len(report_date) >= 4:
            year = int(report_date[:4])
        else:
            year = int(date_filed[:4])
        if not (start_year <= year <= end_year):
            continue
        results.append({
            "cik":          cik,
            "accession_no": acc,
            "date_filed":   date_filed,
            "primary_doc":  doc,
            "year":         year,
        })
    return results


def get_10k_filings(cik: str, start_year: int, end_year: int) -> list[dict]:
    """Fetch 10-K filing metadata from EDGAR Submissions API.
    Reads both 'recent' and historical 'files' archives to avoid missing older filings.
    """
    cik_padded = cik.zfill(10)
    url = f"{BASE_SUBMISSIONS}/CIK{cik_padded}.json"
    r   = rate_limited_get(url)
    if r is None:
        return []
    try:
        data = r.json()
    except Exception:
        return []

    seen_accnos = set()
    filings = []

    # 1. Recent filings
    recent = data.get("filings", {}).get("recent", {})
    for f in _parse_filing_block(recent, cik, start_year, end_year):
        if f["accession_no"] not in seen_accnos:
            seen_accnos.add(f["accession_no"])
            filings.append(f)

    # 2. Historical filing archives (older filings stored in separate JSON files)
    for hist_file in data.get("filings", {}).get("files", []):
        hist_url = f"https://data.sec.gov/submissions/{hist_file['name']}"
        r2 = rate_limited_get(hist_url)
        if r2 is None:
            continue
        try:
            hist_block = r2.json()
        except Exception:
            continue
        for f in _parse_filing_block(hist_block, cik, start_year, end_year):
            if f["accession_no"] not in seen_accnos:
                seen_accnos.add(f["accession_no"])
                filings.append(f)

    return filings


# ── MD&A extraction (same as pilot — tested & working) ────────────────────────
def clean_html_to_text(raw: str) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:", re.I)):
        if tag.name and tag.name.lower() in ("ix:header", "ix:hidden", "ix:references"):
            tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    for src, dst in [(" ", " "), ("’", "'"), ("‘", "'"),
                     ("“", '"'), ("”", '"')]:
        text = text.replace(src, dst)
    text = re.sub(r"[\t\r\n\f\v]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text


ITEM7_END_PATS = [
    r"item\s+7a[.]?\s",
    r"item\s+7a\b",
    r"item\s+8[.]?\s",
    r"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market",
]


def find_mda_start(tl: str, n: int) -> int | None:
    min_pos = max(200, int(n * 0.01))
    pat = r"item\s+7[.]?\s*management[\s']{0,3}s?\s+discussion"
    cands = [m for m in re.finditer(pat, tl) if m.start() > min_pos]
    if cands:
        return cands[-1].start()
    all_i7 = list(re.finditer(r"item\s+7\b", tl))
    for m in reversed(all_i7):
        if m.start() <= min_pos:
            continue
        if re.search(r"management", tl[m.start(): m.start() + 120]):
            return m.start()
    max_fb = int(n * 0.85)
    cands3 = [m for m in all_i7 if min_pos < m.start() < max_fb]
    return cands3[-1].start() if cands3 else None


def extract_mda(raw: str, max_chars: int = 120_000) -> str:
    is_html = bool(re.search(r"<html|<htm|<body", raw[:500], re.IGNORECASE))
    text    = clean_html_to_text(raw) if is_html else raw
    tl, n   = text.lower(), len(text)
    start   = find_mda_start(tl, n)
    if start is None:
        return ""
    sf  = start + 200
    end = None
    for pat in ITEM7_END_PATS:
        m = re.search(pat, tl[sf:])
        if m and m.start() > 300:
            end = sf + m.start()
            break
    if end is None or (end - start) < 300:
        end = start + max_chars
    return text[start:end].strip()[:max_chars]


# ── Download one filing ────────────────────────────────────────────────────────
def download_one(filing: dict) -> dict:
    """Download MD&A for one filing. Returns status dict."""
    cik    = filing["cik"]
    acc    = filing["accession_no"]
    doc    = filing.get("primary_doc", "")
    ticker = filing.get("ticker", cik)
    year   = filing["year"]

    acc_nodash = acc.replace("-", "")
    base_url   = f"{BASE_ARCHIVE}/{cik.lstrip('0')}/{acc_nodash}"
    url        = f"{base_url}/{doc}" if doc else f"{base_url}/{acc_nodash}-index.htm"

    r = rate_limited_get(url)
    if r is None:
        # Try index page to find primary document
        idx_url = f"{base_url}/{acc_nodash}-index.htm"
        r2 = rate_limited_get(idx_url)
        if r2 is None:
            return {**filing, "status": "failed", "mda_len": 0, "mda_text": ""}
        # Find 10-K document link
        soup = BeautifulSoup(r2.text, "lxml")
        link = soup.find("a", href=re.compile(r"\.htm", re.I))
        if link:
            r = rate_limited_get(f"{base_url}/{link['href']}")

    if r is None:
        return {**filing, "status": "failed", "mda_len": 0, "mda_text": ""}

    mda = extract_mda(r.text)
    return {
        **filing,
        "status":   "ok" if mda else "empty",
        "mda_len":  len(mda),
        "mda_text": mda,
    }


# ── S3 helpers ─────────────────────────────────────────────────────────────────
def s3_key_for_filing(prefix: str, ticker: str, year: int, acc: str) -> str:
    acc_clean = acc.replace("-", "")
    return f"{prefix}/filings/{ticker}/{year}/{acc_clean}.txt"


def upload_text_to_s3(s3, bucket: str, key: str, text: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))


def upload_parquet_to_s3(s3, bucket: str, key: str, df: pd.DataFrame) -> None:
    import io
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())


def key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False
    except Exception:
        return False


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Parse year range
    if args.test:
        start_year, end_year = 2018, 2019
    else:
        y = args.years.split("-")
        start_year, end_year = int(y[0]), int(y[1])

    s3     = boto3.client("s3")
    prefix = args.prefix

    # ── Load universe ──────────────────────────────────────────────────────────
    # Always fetch SEC ticker→CIK mapping (needed for EDGAR API)
    print("Fetching SEC ticker→CIK mapping...")
    r = rate_limited_get("https://www.sec.gov/files/company_tickers.json")
    sec_tickers = pd.DataFrame.from_dict(r.json(), orient="index")
    sec_tickers.columns = ["cik_int", "ticker", "company_name"]
    sec_tickers["ticker"] = sec_tickers["ticker"].str.upper()
    sec_tickers["cik"]    = sec_tickers["cik_int"].astype(str).str.zfill(10)
    sec_tickers = sec_tickers.drop_duplicates(subset="ticker")

    if args.universe:
        wrds_universe = pd.read_csv(args.universe)
        wrds_universe["ticker"] = wrds_universe["ticker"].str.upper()
        # Merge WRDS tickers with SEC CIKs
        universe = wrds_universe.merge(
            sec_tickers[["ticker", "cik"]], on="ticker", how="inner"
        )
        print(f"Loaded universe: {len(wrds_universe)} WRDS companies → "
              f"{len(universe)} matched with SEC CIK")
    else:
        universe = sec_tickers.copy()
        print(f"SEC universe: {len(universe)} companies")

    if args.test:
        test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
        universe = universe[universe["ticker"].isin(test_tickers)]
        print(f"TEST MODE: {len(universe)} companies, {start_year}–{end_year}")

    # ── Step 1: Collect filing metadata ───────────────────────────────────────
    print(f"\nStep 1: Fetching 10-K filing metadata ({start_year}–{end_year})...")
    all_filings = []
    for _, row in tqdm(universe.iterrows(), total=len(universe), desc="Metadata"):
        cik     = str(row["cik"])
        ticker  = str(row.get("ticker", cik))
        filings = get_10k_filings(cik, start_year, end_year)
        for f in filings:
            f["ticker"] = ticker
        all_filings.extend(filings)

    print(f"Found {len(all_filings)} 10-K filings")

    # Save metadata to S3
    meta_df = pd.DataFrame(all_filings)
    upload_parquet_to_s3(s3, args.bucket,
                         f"{prefix}/raw/filings_metadata.parquet", meta_df)
    print(f"Metadata saved to s3://{args.bucket}/{prefix}/raw/filings_metadata.parquet")

    # ── Step 2: Multi-threaded MD&A download ──────────────────────────────────
    print(f"\nStep 2: Downloading MD&A ({args.workers} threads)...")

    results = []
    skipped = 0

    def process_one(filing):
        ticker = filing.get("ticker", filing["cik"])
        year   = filing["year"]
        acc    = filing["accession_no"]
        s3_key = s3_key_for_filing(prefix, ticker, year, acc)

        # Skip if already uploaded (resume capability)
        if key_exists(s3, args.bucket, s3_key):
            return {**filing, "status": "cached", "mda_len": -1, "s3_key": s3_key}

        result = download_one(filing)

        if result["status"] == "ok" and result["mda_text"]:
            upload_text_to_s3(s3, args.bucket, s3_key, result["mda_text"])
            result["s3_key"] = s3_key
        else:
            result["s3_key"] = ""

        result.pop("mda_text", None)  # don't keep raw text in memory
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, f): f for f in all_filings}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            try:
                results.append(future.result())
            except Exception as e:
                filing = futures[future]
                results.append({**filing, "status": f"error:{e}", "mda_len": 0})

    # ── Step 3: Save results metadata ─────────────────────────────────────────
    results_df = pd.DataFrame(results)
    upload_parquet_to_s3(s3, args.bucket,
                         f"{prefix}/raw/mda_metadata.parquet", results_df)

    # Summary
    ok      = (results_df["status"] == "ok").sum()
    cached  = (results_df["status"] == "cached").sum()
    failed  = results_df["status"].str.startswith("fail").sum()
    empty   = (results_df["status"] == "empty").sum()

    print(f"\n{'='*50}")
    print(f"Results summary:")
    print(f"  OK (new):   {ok}")
    print(f"  Cached:     {cached}")
    print(f"  Empty MD&A: {empty}")
    print(f"  Failed:     {failed}")
    print(f"  Total:      {len(results_df)}")
    print(f"\nAll outputs at: s3://{args.bucket}/{prefix}/")


if __name__ == "__main__":
    main()
