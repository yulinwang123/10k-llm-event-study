"""
local_wrds_pull.py
──────────────────
Run this on your Mac (UChicago network or VPN) to pull all WRDS data
and upload to S3.

Usage:
    # Test mode (5 tickers, 2 years)
    python local_wrds_pull.py --bucket your-bucket-name --test

    # Full run
    python local_wrds_pull.py --bucket your-bucket-name

Requirements:
    pip install wrds boto3 pandas pyarrow
"""

import argparse
import io
import sys

import boto3
import numpy as np
import pandas as pd
import wrds


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",   required=True)
    p.add_argument("--prefix",   default="10k-project")
    p.add_argument("--wrds-user", default="yulinwang")
    p.add_argument("--test",     action="store_true",
                   help="Test mode: 5 tickers, 2018-2019 only")
    return p.parse_args()


# ── S3 helper ──────────────────────────────────────────────────────────────────
def upload_parquet(s3, bucket: str, key: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"  ✓ Uploaded {len(df):,} rows → s3://{bucket}/{key}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    s3   = boto3.client("s3")

    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=args.wrds_user)

    # ── 1. S&P 1500 universe ──────────────────────────────────────────────────
    print("\n[1/5] Pulling S&P 1500 universe from Compustat idxcst_his...")

    if args.test:
        # For test mode use hardcoded pilot tickers
        test_tickers = ("AAPL", "MSFT", "GOOGL", "AMZN", "JPM")
        ticker_str   = "', '".join(test_tickers)
        year_filter  = "AND fyear BETWEEN 2018 AND 2019"
        print(f"  TEST MODE: tickers = {test_tickers}, years = 2018-2019")
    else:
        ticker_str  = None
        year_filter = "AND fyear BETWEEN 2010 AND 2020"

    if args.test:
        sp1500 = db.raw_sql(f"""
            SELECT DISTINCT gvkey, tic AS ticker, conm AS company_name
            FROM comp.funda
            WHERE tic IN ('{ticker_str}')
              AND indfmt = 'INDL' AND datafmt = 'STD'
              AND popsrc = 'D'   AND consol  = 'C'
        """)
    else:
        # Pull S&P 1500 constituents using comp.idxcst_his
        # gvkeyx = '031855' is S&P Composite 1500 (confirmed: 1500 distinct members)
        sp1500 = db.raw_sql("""
            SELECT DISTINCT f.gvkey, f.tic AS ticker, f.conm AS company_name
            FROM comp.funda f
            WHERE EXISTS (
                SELECT 1 FROM comp.idxcst_his s
                WHERE s.gvkey  = f.gvkey
                  AND s.gvkeyx = '031855'
                  AND (s.thru IS NULL OR s.thru >= '2010-01-01')
                  AND s.from   <= '2020-12-31'
            )
            AND indfmt = 'INDL' AND datafmt = 'STD'
            AND popsrc = 'D'   AND consol  = 'C'
            AND fyear BETWEEN 2010 AND 2020
        """)

    print(f"  {len(sp1500)} unique companies in universe")
    upload_parquet(s3, args.bucket, f"{args.prefix}/raw/sp1500_universe.parquet", sp1500)

    # ── 2. Compustat annual fundamentals ─────────────────────────────────────
    print("\n[2/5] Pulling Compustat fundamentals + earnings announcement dates...")

    gvkey_list = "', '".join(sp1500["gvkey"].unique().tolist())

    compustat = db.raw_sql(f"""
        SELECT
            gvkey,
            tic         AS ticker,
            datadate,
            fyear,
            sich        AS sic,
            cik,
            at          AS total_assets,
            prcc_f      AS price,
            csho        AS shares_out,
            ceq         AS book_equity,
            ni          AS net_income,
            dltt        AS lt_debt
        FROM comp.funda
        WHERE gvkey IN ('{gvkey_list}')
          AND indfmt  = 'INDL'
          AND datafmt = 'STD'
          AND popsrc  = 'D'
          AND consol  = 'C'
          {year_filter}
          AND at      > 0
    """)

    # Compute control variables
    compustat["log_assets"]  = np.log(compustat["total_assets"])
    compustat["mktcap"]      = compustat["price"] * compustat["shares_out"]
    compustat["log_mktcap"]  = np.log(compustat["mktcap"].clip(lower=0.001))
    compustat["bm_ratio"]    = compustat["book_equity"] / compustat["mktcap"].clip(lower=0.001)
    compustat["roa"]         = compustat["net_income"]  / compustat["total_assets"].clip(lower=0.001)
    compustat["leverage"]    = compustat["lt_debt"]     / compustat["total_assets"].clip(lower=0.001)

    print(f"  {len(compustat):,} firm-year observations")

    # ── 2b. Pull rdq (earnings announcement date) from comp.fundq Q4 ─────────
    # rdq lives in the quarterly table; Q4 announcement = annual earnings date
    if args.test:
        rdq_year_filter = "AND fyearq BETWEEN 2018 AND 2019"
    else:
        rdq_year_filter = "AND fyearq BETWEEN 2010 AND 2020"

    rdq_df = db.raw_sql(f"""
        SELECT gvkey, fyearq AS fyear, rdq
        FROM comp.fundq
        WHERE gvkey IN ('{gvkey_list}')
          AND fqtr    = 4
          AND indfmt  = 'INDL'
          AND datafmt = 'STD'
          AND popsrc  = 'D'
          AND consol  = 'C'
          {rdq_year_filter}
          AND rdq IS NOT NULL
    """)
    rdq_df["rdq"]   = pd.to_datetime(rdq_df["rdq"], errors="coerce")
    rdq_df["fyear"] = rdq_df["fyear"].astype(int)
    # Keep one rdq per gvkey-fyear (take the latest if duplicates)
    rdq_df = (rdq_df.sort_values("rdq")
                    .drop_duplicates(subset=["gvkey", "fyear"], keep="last"))

    compustat = compustat.merge(rdq_df[["gvkey", "fyear", "rdq"]],
                                on=["gvkey", "fyear"], how="left")

    print(f"  rdq (earnings date) non-null: {compustat['rdq'].notna().sum():,} "
          f"({compustat['rdq'].notna().mean():.1%})")
    upload_parquet(s3, args.bucket, f"{args.prefix}/raw/compustat.parquet", compustat)

    # ── 3. CRSP-Compustat link table ──────────────────────────────────────────
    print("\n[3/5] Pulling CRSP-Compustat link table...")

    ccm = db.raw_sql(f"""
        SELECT gvkey, lpermno AS permno, linktype, linkprim,
               linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE gvkey IN ('{gvkey_list}')
          AND linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
    """)
    ccm["linkdt"]    = pd.to_datetime(ccm["linkdt"],    errors="coerce")
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"], errors="coerce")
    print(f"  {len(ccm):,} link records")
    upload_parquet(s3, args.bucket, f"{args.prefix}/raw/ccm_link.parquet", ccm)

    # ── 4. CRSP daily returns ─────────────────────────────────────────────────
    print("\n[4/5] Pulling CRSP daily stock returns...")

    # Get permnos from CCM
    permnos = ccm["permno"].dropna().unique().tolist()
    permno_str = ", ".join(str(int(p)) for p in permnos)

    if args.test:
        date_filter = "AND date BETWEEN '2017-01-01' AND '2020-12-31'"
    else:
        date_filter = "AND date BETWEEN '2009-01-01' AND '2021-12-31'"

    crsp = db.raw_sql(f"""
        SELECT permno, date, ret, shrout, prc
        FROM crsp.dsf
        WHERE permno IN ({permno_str})
          {date_filter}
          AND ret IS NOT NULL
    """)
    crsp["date"] = pd.to_datetime(crsp["date"])
    print(f"  {len(crsp):,} daily return observations")
    upload_parquet(s3, args.bucket, f"{args.prefix}/raw/crsp_daily.parquet", crsp)

    # ── 5. CRSP market index (for abnormal return calculation) ────────────────
    print("\n[5/5] Pulling CRSP value-weighted market index...")

    if args.test:
        mkt_filter = "WHERE date BETWEEN '2017-01-01' AND '2020-12-31'"
    else:
        mkt_filter = "WHERE date BETWEEN '2009-01-01' AND '2021-12-31'"

    crsp_mkt = db.raw_sql(f"""
        SELECT date, vwretd AS mkt_ret, ewretd AS ew_ret
        FROM crsp.dsi
        {mkt_filter}
    """)
    crsp_mkt["date"] = pd.to_datetime(crsp_mkt["date"])
    print(f"  {len(crsp_mkt):,} market index days")
    upload_parquet(s3, args.bucket, f"{args.prefix}/raw/crsp_market.parquet", crsp_mkt)

    db.close()

    print(f"\n{'='*50}")
    print("All WRDS data uploaded to S3:")
    print(f"  s3://{args.bucket}/{args.prefix}/raw/sp1500_universe.parquet")
    print(f"  s3://{args.bucket}/{args.prefix}/raw/compustat.parquet")
    print(f"  s3://{args.bucket}/{args.prefix}/raw/ccm_link.parquet")
    print(f"  s3://{args.bucket}/{args.prefix}/raw/crsp_daily.parquet")
    print(f"  s3://{args.bucket}/{args.prefix}/raw/crsp_market.parquet")
    print("\nNext: run ec2_edgar_download.py on EC2")


if __name__ == "__main__":
    main()
