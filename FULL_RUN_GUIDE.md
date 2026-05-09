# Full Data Collection Guide
## S&P 1500 × FY2010–2020 — Step-by-Step Execution

---

## Prerequisites

Every time you start a new session, refresh AWS credentials first:

```bash
cat > ~/.aws/credentials << 'EOF'
[default]
aws_access_key_id=ASIA...
aws_secret_access_key=...
aws_session_token=...
EOF

# Verify
aws sts get-caller-identity
```

---

## Step 1 — WRDS Full Pull (Local Mac, UChicago VPN required)

**Connect VPN first**, then run:

```bash
python "/Users/yulinwang/Desktop/macs final project/scripts/local_wrds_pull.py" \
    --bucket yulinwang-10k-llm
```

**What it pulls:**
- ~2,000–3,000 unique S&P 1500 companies (2010–2020 index members)
- Compustat annual fundamentals (assets, equity, earnings, leverage)
- `rdq` (earnings announcement dates) from `comp.fundq` Q4
- CRSP daily returns (~6–8M rows)
- CRSP value-weighted market index
- CCM gvkey→permno link table

**Expected runtime:** 30–60 minutes  
**Expected output:**
```
s3://yulinwang-10k-llm/10k-project/raw/sp1500_universe.parquet   (~2,000 rows)
s3://yulinwang-10k-llm/10k-project/raw/compustat.parquet         (~18,000 rows)
s3://yulinwang-10k-llm/10k-project/raw/ccm_link.parquet          (~5,000 rows)
s3://yulinwang-10k-llm/10k-project/raw/crsp_daily.parquet        (~6M rows)
s3://yulinwang-10k-llm/10k-project/raw/crsp_market.parquet       (~3,000 rows)
```

---

## Step 2 — Launch EC2 for EDGAR Download

### 2a. Launch Instance (AWS Console)

1. Login → AWS Console → search **EC2** → **Launch Instance**
2. Settings:
   - **Name:** `edgar-download`
   - **AMI:** Ubuntu Server 22.04 LTS
   - **Instance type:** `t3.large` (2 vCPU, 8 GB, ~$0.08/hr)
   - **Key pair:** Create new → name `ec2-key` → download `.pem` file
   - **Security group:** default (SSH port 22 allowed)
3. Click **Launch instance**
4. Wait ~1 min → go to **Instances** → copy **Public IPv4 address**

### 2b. SSH into EC2

```bash
# Fix key permissions (Mac requires this)
chmod 400 ~/Downloads/ec2-key.pem

# SSH in (replace with your actual IP)
ssh -i ~/Downloads/ec2-key.pem ubuntu@<YOUR_EC2_IP>
```

### 2c. Install Dependencies on EC2

```bash
sudo apt update -y && sudo apt install -y python3-pip
pip3 install boto3 pandas pyarrow requests beautifulsoup4 lxml tqdm
```

### 2d. Configure AWS Credentials on EC2

Same credentials as your Mac (copy from AWS Academy → Details → CLI):

```bash
mkdir -p ~/.aws
cat > ~/.aws/credentials << 'EOF'
[default]
aws_access_key_id=ASIA...
aws_secret_access_key=...
aws_session_token=...
EOF

# Verify
aws sts get-caller-identity
```

### 2e. Upload Script to EC2

Run this **on your Mac** (not on EC2):

```bash
scp -i ~/Downloads/ec2-key.pem \
    "/Users/yulinwang/Desktop/macs final project/scripts/ec2_edgar_download.py" \
    ubuntu@<YOUR_EC2_IP>:~/
```

---

## Step 3 — Run EDGAR Full Download on EC2

**On EC2**, after Step 1 (WRDS) has finished:

```bash
# Download WRDS universe from S3
aws s3 cp s3://yulinwang-10k-llm/10k-project/raw/sp1500_universe.parquet /tmp/
python3 -c "
import pandas as pd
pd.read_parquet('/tmp/sp1500_universe.parquet').to_csv('/tmp/sp1500.csv', index=False)
print('Universe:', pd.read_csv('/tmp/sp1500.csv').shape)
"

# Run full download in background (nohup keeps it alive after SSH disconnect)
nohup python3 ec2_edgar_download.py \
    --bucket yulinwang-10k-llm \
    --universe /tmp/sp1500.csv \
    --workers 16 \
    > edgar.log 2>&1 &

echo "Job started, PID: $!"
```

**Monitor progress (from EC2):**
```bash
tail -f edgar.log          # live log
grep "OK\|Failed\|Total" edgar.log | tail -5   # summary
```

**Expected runtime:** 1–2 hours  
**Expected output:**
```
s3://yulinwang-10k-llm/10k-project/raw/mda_metadata.parquet      (~16,000 rows)
s3://yulinwang-10k-llm/10k-project/filings/{ticker}/{year}/*.txt (~14,000 files)
```

> ⚠️ AWS Academy sessions expire after ~4 hours. The job runs on EC2 in the background
> and writes directly to S3, so even if your laptop closes, the job continues.
> However EC2 credentials may expire — if so, re-run with fresh credentials
> and the resume feature (cached S3 check) will skip already-downloaded files.

---

## Step 4 — Merge Master Panel

After Steps 1 and 3 both complete:

```bash
# On your Mac (local)
python "/Users/yulinwang/Desktop/macs final project/scripts/merge_panel.py" \
    --bucket yulinwang-10k-llm
```

**Expected output:**
```
s3://yulinwang-10k-llm/10k-project/processed/master_panel.parquet
```

**Expected shape:** ~14,000–16,000 firm-years (some filings will fail to extract MD&A)

**Verify:**
```bash
python3 -c "
import boto3, io, pandas as pd
s3 = boto3.client('s3')
obj = s3.get_object(Bucket='yulinwang-10k-llm', Key='10k-project/processed/master_panel.parquet')
df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
print(df.shape)
print(df[['ticker','fyear','rdq','mda_len','log_assets']].describe())
print('rdq non-null:', df['rdq'].notna().mean())
"
```

---

## Step 5 — Verify S3 Contents

```bash
# Count all MD&A files
aws s3 ls s3://yulinwang-10k-llm/10k-project/filings/ --recursive | wc -l

# Check file sizes (sample)
aws s3 ls s3://yulinwang-10k-llm/10k-project/filings/ --recursive --human-readable | head -20

# List all processed files
aws s3 ls s3://yulinwang-10k-llm/10k-project/processed/
```

---

## Summary Table

| Step | Where | Command | Time |
|------|-------|---------|------|
| 1. WRDS pull | Mac (VPN) | `local_wrds_pull.py` | 30–60 min |
| 2. Launch EC2 | AWS Console | click-through | 5 min |
| 3. EDGAR download | EC2 (background) | `ec2_edgar_download.py --workers 16` | 1–2 hrs |
| 4. Merge panel | Mac | `merge_panel.py` | 1–2 min |
| 5. Verify | Mac | `aws s3 ls ...` | instant |

**Total wall-clock time:** ~2 hours (Steps 1 and 3 can overlap)

---

## Next Steps After Data Collection

- **Week 2:** Run `week2_text_measures.ipynb` (LM Dict + BERT embedding + FinBERT + Llama batches)
- **Midway3:** Submit Llama SLURM job on real MD&A texts
- **Week 3:** `week3_event_study.ipynb` — compute CAR[-1,+1] and run regressions
