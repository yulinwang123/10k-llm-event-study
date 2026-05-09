"""
merge_panel.py
──────────────
Merge WRDS data + EDGAR MD&A metadata into master panel.
Run this after both local_wrds_pull.py and ec2_edgar_download.py complete.

Usage:
    python merge_panel.py --bucket your-bucket-name --test
    python merge_panel.py --bucket your-bucket-name
"""

import argparse
import io

import boto3
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket",  required=True)
    p.add_argument("--prefix",  default="10k-project")
    p.add_argument("--test",    action="store_true")
    return p.parse_args()


def read_parquet_from_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def upload_parquet(s3, bucket: str, key: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"  ✓ Saved {len(df):,} rows → s3://{bucket}/{key}")


def main():
    args = parse_args()
    s3   = boto3.client("s3")
    px   = args.prefix

    print("Loading data from S3...")
    compustat  = read_parquet_from_s3(s3, args.bucket, f"{px}/raw/compustat.parquet")
    ccm        = read_parquet_from_s3(s3, args.bucket, f"{px}/raw/ccm_link.parquet")
    mda_meta   = read_parquet_from_s3(s3, args.bucket, f"{px}/raw/mda_metadata.parquet")

    print(f"  Compustat:   {len(compustat):,} rows")
    print(f"  CCM link:    {len(ccm):,} rows")
    print(f"  MDA metadata:{len(mda_meta):,} rows")

    # ── 1. Merge Compustat + CCM to get permno ────────────────────────────────
    compustat["datadate"] = pd.to_datetime(compustat["datadate"])

    # Keep most recent valid CCM link per gvkey
    ccm_clean = (
        ccm
        .sort_values(["gvkey", "linkdt"], ascending=[True, False])
        .drop_duplicates(subset=["gvkey"], keep="first")
        [["gvkey", "permno"]]
    )

    panel = compustat.merge(ccm_clean, on="gvkey", how="inner")
    print(f"\nAfter Compustat + CCM merge: {len(panel):,} rows")

    # ── 2. Merge MD&A metadata ────────────────────────────────────────────────
    # MDA metadata has ticker + year; Compustat has ticker + fyear
    mda_ok = mda_meta[mda_meta["status"].isin(["ok", "cached"])].copy()
    mda_ok["year"] = mda_ok["year"].astype(int)

    panel["fyear"] = panel["fyear"].astype(int)

    # Merge on ticker + year
    panel = panel.merge(
        mda_ok[["ticker", "year", "accession_no", "date_filed", "s3_key", "mda_len"]],
        left_on=["ticker", "fyear"],
        right_on=["ticker", "year"],
        how="inner"
    )
    panel = panel.drop(columns=["year"])
    print(f"After MD&A merge: {len(panel):,} rows")

    # ── 3. Select and order final columns ─────────────────────────────────────
    keep = [
        # Identifiers
        "gvkey", "permno", "cik", "ticker",
        "sic", "fyear", "datadate",
        # Event study keys
        "date_filed", "rdq",           # rdq = earnings announcement date
        "accession_no", "s3_key",      # S3 path to MD&A text
        "mda_len",
        # Financial controls
        "log_assets", "log_mktcap", "bm_ratio", "roa", "leverage",
    ]
    # Only keep columns that exist
    keep = [c for c in keep if c in panel.columns]
    panel = panel[keep].drop_duplicates(subset=["gvkey", "fyear"])

    # ── 4. Quality checks ──────────────────────────────────────────────────────
    print(f"\nFinal panel: {len(panel):,} firm-years")
    print(f"  rdq non-null:      {panel['rdq'].notna().sum():,} ({panel['rdq'].notna().mean():.1%})")
    print(f"  mda_len median:    {panel['mda_len'].median():.0f} chars")
    print(f"  Unique firms:      {panel['gvkey'].nunique():,}")
    print(f"  Year range:        {panel['fyear'].min()} – {panel['fyear'].max()}")
    print(f"  Missing controls:  {panel[['log_assets','bm_ratio','roa','leverage']].isna().sum().to_dict()}")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    suffix = "_test" if args.test else ""
    upload_parquet(s3, args.bucket,
                   f"{px}/processed/master_panel{suffix}.parquet", panel)

    print(f"\n✓ Master panel saved.")
    print(f"  Next: run week2_text_measures.ipynb (LM + FinBERT + Llama)")


if __name__ == "__main__":
    main()
