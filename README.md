# Do LLMs Read 10-Ks Better Than Dictionaries?
### Comparing NLP Approaches to Predict Earnings Announcement Returns

**Course:** MACS 30113 — Large-Scale Computing for the Social Sciences  
**Author:** Yulin Wang, University of Chicago

---

## Research Question

10-K filings contain a Management Discussion & Analysis (MD&A) section where executives describe the company's performance, outlook, and risks in plain language. Loughran & McDonald (2011) showed that bag-of-words dictionary methods applied to these texts can predict abnormal stock returns around earnings announcements. But dictionaries are blind to context, negation, and nuance.

This project asks: **do more sophisticated NLP methods — neural sentiment (FinBERT), semantic novelty (Sentence-BERT), and large language model scoring (Llama-3.1-8B) — produce text measures that better predict cumulative abnormal returns (CAR) around earnings announcements?** And once we have all four measures in a horse-race regression, which signal survives, and does any of it hold up under causal identification?

---

## Pipeline Overview

Data flows through three compute environments:

```
  Local Mac (VPN)          AWS EC2 (t3.large)       Local Mac
  local_wrds_pull.py       ec2_edgar_download.py    merge_panel.py
  WRDS API ──────┐         SEC EDGAR ────────┐      reads S3 ──┐
                 │ direct upload             │ direct upload   │ writes back
                 ▼                           ▼                 ▼
        ┌────────────────────────────────────────────────────────────┐
        │               AWS S3  (yulinwang-10k-llm)                  │
        │                                                            │
        │  10k-project/raw/         ← WRDS parquets (5 files)        │
        │  10k-project/filings/     ← MD&A text files (~14,000)      │
        │  10k-project/processed/   ← master_panel.parquet           │
        └──────────────────────────────┬─────────────────────────────┘
                                       │ Midway3 pulls via boto3
                                       ▼
                        ┌──────────────────────────────┐
                        │  Midway3 HPC  (macs30113)    │
                        │  GPU partition (V100 / A100) │
                        │                              │
                        │  Track 3: Llama-3.1-8B       │
                        │  (vLLM, SLURM job array)     │
                        │                              │
                        │  Track 2: FinBERT            │
                        │  Track 4: Sentence-BERT      │
                        │  (PyTorch, SLURM job array)  │
                        └──────────────┬───────────────┘
                                       │ scp results to local
                                       ▼
                        ┌──────────────────────────────┐
                        │  Local Mac — Analysis only   │
                        │                              │
                        │  Track 1: LM Dictionary      │
                        │  compute_car_filing.py       │
                        │  analysis_track124.ipynb     │
                        │  analysis_track1234.ipynb    │
                        └──────────────────────────────┘
```

---

## Data Sources

| Source | Access | Content |
|--------|--------|---------|
| **WRDS / Compustat** | UChicago VPN required | Annual fundamentals (assets, ROA, leverage, book-to-market), Q4 earnings announcement dates (`rdq`) |
| **WRDS / CRSP** | UChicago VPN required | Daily stock returns, value-weighted market index, gvkey→permno link |
| **SEC EDGAR** | Public | Full-text 10-K filings; MD&A extracted via iXBRL-aware parser |

---

## Step-by-Step Execution

### Prerequisites — AWS Credentials

AWS Academy credentials expire every ~4 hours. Refresh before each session:

```bash
cat > ~/.aws/credentials << 'EOF'
[default]
aws_access_key_id=ASIA...
aws_secret_access_key=...
aws_session_token=...
EOF
aws sts get-caller-identity   # verify
```

---

### Step 1 — WRDS Pull (Local Mac, UChicago VPN required)

Pulls S&P 1500 universe, Compustat fundamentals, CRSP returns, and the CCM gvkey→permno link. Uploads directly to S3.

```bash
# Connect to UChicago VPN first
python scripts/local_wrds_pull.py --bucket yulinwang-10k-llm
```

**Runtime:** 30–60 min  
**Output on S3:**
```
s3://yulinwang-10k-llm/10k-project/raw/
├── sp1500_universe.parquet   # ~2,000 S&P 1500 members (2010–2020)
├── compustat.parquet         # ~18,000 firm-year fundamentals
├── ccm_link.parquet          # gvkey → permno mapping
├── crsp_daily.parquet        # ~6M daily returns
└── crsp_market.parquet       # VW market index
```

---

### Step 2 — EDGAR MD&A Download (AWS EC2)

10-K filings are public. EC2 is used for multi-threaded downloading without consuming local bandwidth.

**Launch EC2 (AWS Console):**
- AMI: Ubuntu Server 22.04 LTS
- Instance type: `t3.large` (2 vCPU, 8 GB, ~$0.08/hr)
- Key pair: `ec2-key` (download `.pem`)

```bash
# Fix key permissions, SSH in
chmod 400 ec2-key.pem
ssh -i ec2-key.pem ubuntu@<EC2_PUBLIC_IP>

# On EC2: install dependencies
sudo apt update -y && sudo apt install -y python3-pip
pip3 install boto3 pandas pyarrow requests beautifulsoup4 lxml tqdm

# Set AWS credentials (same as Mac), then upload and run script
# From Mac:
scp -i ec2-key.pem scripts/ec2_edgar_download.py ubuntu@<EC2_PUBLIC_IP>:~/

# On EC2 — run in background (job persists if SSH disconnects)
nohup python3 ec2_edgar_download.py \
    --bucket yulinwang-10k-llm --workers 16 \
    > edgar.log 2>&1 &

tail -f edgar.log   # monitor
```

**Runtime:** 1–2 hours  
**Output on S3:**
```
s3://yulinwang-10k-llm/10k-project/
├── raw/mda_metadata.parquet          # filing metadata + download status
└── filings/{ticker}/{year}/*.txt     # ~14,000 MD&A text files
```

> The downloader respects SEC's rate limit (≤10 req/sec), retries failures, and resumes interrupted runs by checking S3 before re-downloading.

---

### Step 3 — Build Master Panel (Local Mac)

Joins WRDS fundamentals, CRSP returns, and MD&A metadata into a single firm × year panel. Reads from and writes back to S3.

```bash
python scripts/merge_panel.py --bucket yulinwang-10k-llm
```

**Output:**
```
s3://yulinwang-10k-llm/10k-project/processed/master_panel.parquet
```

Key columns: `gvkey`, `permno`, `ticker`, `fyear`, `rdq`, `date_filed`, `s3_key` (path to MD&A text), `log_assets`, `bm_ratio`, `roa`, `leverage`, `car_1_1`, `car_3_3`.

**Expected shape:** ~11,000–14,000 firm-years after requiring matched MD&A + CRSP coverage.

---

### Step 4 — Download Data to Midway3

SSH into Midway3 and download all necessary data from S3 to scratch. AWS CLI is not available on Midway3 — use the Python boto3 downloader.

```bash
ssh <cnetid>@midway3.rcc.uchicago.edu

SCRATCH="/scratch/midway3/${USER}"
mkdir -p ${SCRATCH}/10k_data/10k-project/{raw,processed,filings,llm_batches,llm_out}

# Download via Python (boto3)
python3 - << 'EOF'
import boto3, os, pathlib
s3 = boto3.client('s3')
bucket = 'yulinwang-10k-llm'
root   = '/scratch/midway3/' + os.environ['USER'] + '/10k_data/10k-project'

for key in ['10k-project/processed/master_panel.parquet',
            '10k-project/raw/crsp_daily.parquet',
            '10k-project/raw/crsp_market.parquet']:
    dest = root + '/' + key.replace('10k-project/', '')
    pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, dest)
    print(f'✓ {key}')

# Download all MD&A text files
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket, Prefix='10k-project/filings/'):
    for obj in page.get('Contents', []):
        dest = root + '/' + obj['Key'].replace('10k-project/', '')
        pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(dest):
            s3.download_file(bucket, obj['Key'], dest)
print('✓ All filings downloaded')
EOF
```

---

### Step 5 — Midway3 Environment Setup (One-Time)

```bash
SCRATCH="/scratch/midway3/${USER}"
export PIP_CACHE_DIR="${SCRATCH}/pip_cache"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p $PIP_CACHE_DIR $TMPDIR

# Create venv under SCRATCH (home quota ~30 GB is too small for vLLM + PyTorch)
python3 -m venv ${SCRATCH}/vllm_env
source ${SCRATCH}/vllm_env/bin/activate

module load cuda/12.3
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install setuptools_scm numpy
pip install vllm==0.11.2 --no-build-isolation
pip install transformers==4.57.6 accelerate huggingface_hub
pip install sentence-transformers pandas pyarrow tqdm boto3
```

> **GPU compatibility note:** V100 nodes (compute capability 7.0) do not support bfloat16. The scripts use `dtype="half"` (float16) which works on both V100 and A100. Do **not** add `--constraint=a100` unless you want to wait in a longer queue.

**Download Llama model weights (~16 GB):**
```bash
source ${SCRATCH}/vllm_env/bin/activate
huggingface-cli login   # paste HF token
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir ${SCRATCH}/models/Meta-Llama-3.1-8B-Instruct \
    --local-dir-use-symlinks False
```

---

### Step 6 — Track 3: Llama Inference (Midway3, SLURM)

Prepare batch JSONL files (shard size 1000, for job array limit of 12):

```bash
# On Midway3
source ${SCRATCH}/vllm_env/bin/activate
python scripts/prepare_llama_batches.py \
    --data-root ${SCRATCH}/10k_data/10k-project \
    --shard-size 1000

# Check output
ls ${SCRATCH}/10k_data/10k-project/llm_batches/ | wc -l   # should be ~12
```

Submit the job array:
```bash
sbatch --array=0-11%3 submit_llama.sh
squeue -u $USER   # monitor
```

**Output:** `${SCRATCH}/10k_data/10k-project/llm_out/results_000.jsonl` through `results_011.jsonl`  
Each line: `{"gvkey": ..., "fyear": ..., "scores": {"management_optimism": 7, "guidance_specificity": 4, "uncertainty_hedging": 5, "risk_framing": 3}}`

Download results to local:
```bash
# From Mac
scp -r <cnetid>@midway3.rcc.uchicago.edu:${SCRATCH}/10k_data/10k-project/llm_out/ \
    "data/llm_out/"
```

---

### Step 7 — Track 2 & 4: FinBERT + Sentence-BERT (Midway3, SLURM)

```bash
# On Midway3
sbatch scripts/submit_finbert.sh
squeue -u $USER
```

This runs `scripts/midway3_finbert_embed.py`, which computes:
- **Track 2:** FinBERT sentence-level sentiment → `fb_net` = P(positive) − P(negative)
- **Track 4:** Sentence-BERT year-over-year cosine similarity → `embed_novelty` = 1 − cosine similarity

Download results to local:
```bash
# From Mac
scp <cnetid>@midway3.rcc.uchicago.edu:${SCRATCH}/10k_data/10k-project/processed/finbert_scores.parquet data/
scp <cnetid>@midway3.rcc.uchicago.edu:${SCRATCH}/10k_data/10k-project/processed/embed_similarity.parquet data/
```

---

### Step 8 — CAR Computation (Local Mac)

Compute cumulative abnormal returns (market-adjusted) around **both** event dates:

```bash
# CAR around rdq (earnings announcement date) — computed inside analysis_track124.ipynb

# CAR around date_filed (10-K filing date) — robustness check
python scripts/compute_car_filing.py
# Output: data/car_filing_date.parquet
```

Uses `data/crsp_daily.parquet` and `data/crsp_market.parquet` already downloaded locally.

---

### Step 9 — Analysis Notebooks (Local Mac)

Run in order:

```bash
# Step 9a: Track 1 (LM Dictionary) + Track 2 (FinBERT) + Track 4 (SBERT)
# Computes CAR[−1,+1] and CAR[−3,+3] around rdq
# Output: data/analysis_panel.parquet
jupyter notebook analysis_track124.ipynb

# Step 9b: Add Track 3 (Llama), horse-race, causal identification, robustness
jupyter notebook analysis_track1234.ipynb
```

---

## Compute Environment Summary

| Task | Where | Why |
|------|-------|-----|
| WRDS data pull | Local Mac (UChicago VPN) | License restricts to whitelisted IPs |
| SEC EDGAR download | AWS EC2 (`t3.large`) | Multi-threaded, public data, persistent background job |
| Intermediate storage | AWS S3 (`yulinwang-10k-llm`) | Shared across Mac / EC2 / Midway3 |
| Llama-3.1-8B inference | Midway3 GPU (`macs30113`) | 80 GB VRAM, vLLM batch throughput |
| FinBERT + Sentence-BERT | Midway3 GPU (`macs30113`) | PyTorch GPU acceleration |
| CAR computation & regression | Local Mac | Interactive, all data fits in memory |

---

## Analysis Design

### Four NLP Tracks

| Track | Method | Key Variable | What It Measures |
|-------|--------|--------------|-----------------|
| T1 | Loughran-McDonald Dictionary | `lm_tone` = (pos−neg)/total | Bag-of-words surface sentiment |
| T2 | FinBERT (`ProsusAI/finbert`) | `fb_net` = P(pos)−P(neg) | Contextual neural sentiment |
| T3 | Llama-3.1-8B-Instruct (vLLM) | `llm_optimism`, `llm_specificity`, `llm_hedging`, `llm_risk` | LLM multidimensional scoring |
| T4 | Sentence-BERT (`all-mpnet-base-v2`) | `embed_novelty` = 1 − cos_sim(t, t−1) | Year-over-year semantic novelty |

### Regression Strategy

All regressions use OLS with industry (2-digit SIC) + year fixed effects and firm-clustered standard errors. Variables are z-score standardized for cross-track comparability. Sample: S&P 1500, FY2010–2020, N ≈ 9,735 firm-years.

**M1–M4:** Each track estimated separately (standalone R²)  
**MC (Horse-Race):** All 8 NLP variables compete simultaneously — tests which signal survives head-to-head  
**MD:** Robustness with CAR[−3,+3]

### Causal Identification

OLS with FE does not rule out time-varying firm-level confounders (e.g., persistently optimistic management at persistently good companies). Two strategies:

**First Difference (FD):** Regress CAR on year-over-year *changes* in NLP scores. Removes all time-invariant firm heterogeneity.

**IV / 2SLS:** Instrument each firm's LLM optimism with the leave-one-out mean of peers in the same 2-digit SIC × year cell. First-stage F = 116 (strong instrument). Tests whether industry-driven variation in optimism causally predicts CAR.

### Robustness Checks

**CAR[−3,+3]:** Wider event window (Model MD)

**date_filed event window:** Recompute CAR around the 10-K filing date instead of the earnings announcement date. Tests whether text carries *incremental* information beyond what markets price on announcement day.

---

## Repository Structure

```
├── scripts/
│   ├── local_wrds_pull.py          # Step 1: WRDS → S3
│   ├── ec2_edgar_download.py       # Step 2: SEC EDGAR → S3 (run on EC2)
│   ├── merge_panel.py              # Step 3: S3 → master_panel.parquet
│   ├── prepare_llama_batches.py    # Step 6: build JSONL shards for Llama
│   ├── midway3_finbert_embed.py    # Step 7: FinBERT + SBERT (Midway3)
│   ├── submit_finbert.sh           # SLURM submission for FinBERT/SBERT
│   └── compute_car_filing.py       # Step 8: CAR around date_filed
│
├── week2_llama_inference.py        # Step 6: vLLM inference script (Midway3)
├── submit_llama.sh                 # SLURM job array for Llama
├── setup_midway3.sh                # One-time Midway3 environment setup
│
├── analysis_track124.ipynb         # Step 9a: Track 1/2/4 + CAR computation
├── analysis_track1234.ipynb        # Step 9b: All tracks + causal ID + robustness
│
├── data/                           # Local data (gitignored)
│   ├── master_panel.parquet
│   ├── crsp_daily.parquet
│   ├── crsp_market.parquet
│   ├── lm_scores.parquet
│   ├── finbert_scores.parquet
│   ├── embed_similarity.parquet
│   ├── analysis_panel.parquet
│   ├── analysis_panel_1234.parquet
│   ├── car_filing_date.parquet
│   ├── regression_results_1234.csv
│   └── llm_out/                    # results_000.jsonl … results_011.jsonl
│
├── LM_MasterDictionary.csv         # Loughran-McDonald word list
└── proposal_10k_llm_eventstudy.tex # Research proposal
```

---

## Dependencies

**Local Mac / EC2:**
```bash
pip install wrds boto3 pandas pyarrow requests beautifulsoup4 lxml tqdm scipy statsmodels seaborn
```

**Midway3 (GPU environment under `$SCRATCH`):**
```bash
module load cuda/12.3
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install setuptools_scm numpy
pip install vllm==0.11.2 --no-build-isolation
pip install transformers==4.57.6 accelerate huggingface_hub
pip install sentence-transformers pandas pyarrow tqdm boto3
```

---

## References

- Loughran, T. & McDonald, B. (2011). When is a liability not a liability? *Journal of Finance*, 66(1), 35–65.
- Yang, Y. et al. (2020). FinBERT: A Pretrained Language Model for Financial Communications. *arXiv:2006.08097*.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
- Dubey, A. et al. (2024). The Llama 3 Herd of Models. *arXiv:2407.21783*.
- Ball, R. & Brown, P. (1968). An Empirical Evaluation of Accounting Income Numbers. *Journal of Accounting Research*, 6(2), 159–178.
