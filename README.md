# Do LLMs Read 10-Ks Better Than Dictionaries?
### Comparing NLP Approaches to Predict Earnings Announcement Returns

**Course:** MACS 30113 — Large-Scale Computing for the Social Sciences  
**Author:** Yulin Wang, University of Chicago

---

## Research Question

10-K filings contain a Management Discussion & Analysis (MD&A) section where executives describe the company's performance, outlook, and risks in plain language. Loughran & McDonald (2011) showed that bag-of-words dictionary methods applied to these texts can predict abnormal stock returns around earnings announcements. But dictionaries are blind to context, negation, and nuance.

This project asks: **do more sophisticated NLP methods — neural sentiment (FinBERT), semantic novelty (Sentence-BERT), and large language model scoring (Llama-3.1-8B) — produce text measures that better predict cumulative abnormal returns (CAR) around the 10-K filing date, and therefore provide more actionable signals for investors processing annual filings?** Once we have all four measures in a horse-race regression, which signal survives head-to-head competition, and does any of it hold up under causal identification strategies that rule out confounding?

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

This project uses AWS Academy, which issues temporary session credentials that expire every ~4 hours. Before running any step that touches S3 or EC2, refresh credentials from the AWS Academy portal (Details → CLI) and paste them into `~/.aws/credentials`:

```bash
cat > ~/.aws/credentials << 'EOF'
[default]
aws_access_key_id=ASIA...
aws_secret_access_key=...
aws_session_token=...
EOF
aws sts get-caller-identity   # verify — should return your account ID
```

---

### Step 1 — WRDS Pull (Local Mac, UChicago VPN required)

Queries the Wharton Research Data Services (WRDS) API to pull four datasets that form the financial backbone of the event study: the S&P 1500 index membership list (2010–2020) that defines the firm universe; Compustat annual fundamentals (assets, earnings, leverage, book-to-market) used as regression controls; CRSP daily stock returns and the value-weighted market index needed to compute cumulative abnormal returns; and the CRSP-Compustat Merged (CCM) link table that maps Compustat `gvkey` identifiers to CRSP `permno` identifiers, bridging the two databases at the firm level.

WRDS access is licensed to UChicago and restricted to whitelisted IP addresses, so this step must run on a local machine connected to the UChicago VPN — it cannot be run from EC2 or Midway3. All five output files are uploaded directly to S3, which serves as the shared data store across the three compute environments.

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

Downloads the full text of 10-K annual filings for every firm-year in the S&P 1500 universe from the SEC's EDGAR system. For each filing, the script locates the document on EDGAR's full-text search index, fetches the HTML/iXBRL file, and extracts the Management Discussion & Analysis (MD&A) section — the portion of the 10-K where executives describe business performance, strategy, and risks in natural language. Extracted text is written as plain `.txt` files directly to S3.

SEC EDGAR is public data with no IP restriction, so EC2 is chosen purely for operational reasons: downloading ~14,000 filings with 16 parallel workers takes 1–2 hours and would monopolize local bandwidth. Running the job on EC2 with `nohup` lets it continue in the background even after the SSH session disconnects. The script respects SEC's rate limit (≤10 req/sec), retries transient failures automatically, and resumes interrupted runs by checking S3 before re-downloading each file.

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

# From Mac: upload the download script to EC2
scp -i ec2-key.pem scripts/ec2_edgar_download.py ubuntu@<EC2_PUBLIC_IP>:~/

# On EC2 — run in background (job continues after SSH disconnect)
nohup python3 ec2_edgar_download.py \
    --bucket yulinwang-10k-llm --workers 16 \
    > edgar.log 2>&1 &

tail -f edgar.log   # monitor progress
```

**Runtime:** 1–2 hours  
**Output on S3:**
```
s3://yulinwang-10k-llm/10k-project/
├── raw/mda_metadata.parquet          # filing metadata + download status per firm-year
└── filings/{ticker}/{year}/*.txt     # ~14,000 MD&A plain-text files
```

---

### Step 3 — Build Master Panel (Local Mac)

Assembles the three data sources into a single analysis-ready firm × year panel by joining Compustat fundamentals and CRSP returns (via the CCM gvkey→permno link) with the EDGAR filing metadata, producing one row per firm-year. This is also where the event-study outcome variable is constructed: cumulative abnormal return (CAR), defined as the sum of daily market-adjusted returns over the three-day window [−1, +1] around the 10-K filing date. Market-adjusted return subtracts the CRSP value-weighted market return from each day's firm return, removing common market movements so that any remaining return reflects firm-specific news from the filing. The merged panel is written back to S3 for use by all downstream steps.

```bash
python scripts/merge_panel.py --bucket yulinwang-10k-llm
```

**Output:**
```
s3://yulinwang-10k-llm/10k-project/processed/master_panel.parquet
```

Key columns: `gvkey`, `permno`, `ticker`, `fyear`, `rdq` (earnings announcement date), `date_filed` (10-K filing date), `s3_key` (S3 path to MD&A text), `log_assets`, `bm_ratio`, `roa`, `leverage`, `car_filed_1_1`, `car_filed_3_3`.

**Expected shape:** ~11,000–14,000 firm-years after requiring matched MD&A + CRSP coverage.

---

### Step 4 — Download Data to Midway3

Transfers the master panel and all MD&A text files from S3 to Midway3's scratch storage so the GPU jobs in Steps 6 and 7 can read them locally rather than streaming from S3. Midway3 does not have the AWS CLI installed, so the transfer uses a short `boto3` script instead. Data is placed under `/scratch/midway3/${USER}/` rather than the home directory because the home quota (~30 GB) is too small for the full dataset and the Llama model weights (~16 GB).

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

Creates an isolated Python virtual environment under scratch and installs the full GPU inference stack. This is a one-time setup; subsequent SLURM jobs simply activate the same environment. `vllm==0.11.2` handles batched Llama inference, `transformers==4.57.6` is pinned because vLLM 0.11.2 breaks with newer releases, and `torch` is compiled for CUDA 12.1 to match Midway3's GPU drivers. The Llama model weights (~16 GB) are downloaded from HuggingFace once and stored in scratch alongside the environment.

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

> **GPU compatibility note:** V100 nodes (compute capability 7.0) do not support bfloat16. The scripts use `dtype="half"` (float16) which works on both V100 and A100. Do **not** add `--constraint=a100` to avoid long queue waits — the float16 fix handles compatibility.

> **transformers version pin:** vLLM 0.11.2 requires `transformers==4.57.6`. Installing a newer transformers version causes incompatibility errors at model load time.

**Download Llama model weights (~16 GB, one-time):**
```bash
source ${SCRATCH}/vllm_env/bin/activate
huggingface-cli login   # paste HF token (request access at huggingface.co/meta-llama)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir ${SCRATCH}/models/Meta-Llama-3.1-8B-Instruct \
    --local-dir-use-symlinks False
# Download takes ~15 min on Midway3 fast scratch I/O
```

---

### Step 6 — Track 3: Llama-3.1-8B Inference (Midway3, SLURM)

#### Model and Inference Engine

**Model:** [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)  
An 8-billion parameter instruction-tuned language model from Meta. Chosen for its strong instruction-following on structured JSON output tasks and its ability to run on a single A100/V100 GPU (40 GB VRAM). Access requires approval at HuggingFace.

**Inference engine:** [`vLLM`](https://github.com/vllm-project/vllm) v0.11.2  
vLLM enables high-throughput batched inference via PagedAttention, making it ~10–20× faster than HuggingFace `generate()` for large batches. Configuration used:
```python
LLM(model=model_path,
    dtype="half",              # float16 — required for V100 compatibility
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    max_model_len=4096)
SamplingParams(temperature=0.0,   # greedy decoding — fully deterministic output
               max_tokens=128,
               stop=["\n\n"])
```

#### Scoring Dimensions

Each MD&A excerpt (first 6,000 characters ≈ 1,200 tokens) is scored on four theory-driven dimensions via a zero-shot structured prompt. The model returns a JSON object with integer scores 0–10:

| Dimension | Variable | What it measures |
|-----------|----------|-----------------|
| `management_optimism` | `llm_optimism` | Forward-looking sentiment: 0 = strongly pessimistic, 10 = strongly confident |
| `guidance_specificity` | `llm_specificity` | Concreteness of forward guidance: 0 = only vague qualitative language, 10 = detailed numerical targets and timelines |
| `uncertainty_hedging` | `llm_hedging` | Density of hedging language ("may", "could", "subject to change"): 0 = confident/certain, 10 = constant qualification |
| `risk_framing` | `llm_risk` | How risks are framed relative to opportunities: 0 = risks downplayed/manageable, 10 = risks prominent and threatening |

The key innovation over Track 1 (LM Dictionary) is `guidance_specificity` — whether management provides concrete numerical targets — which has no dictionary equivalent and directly reduces information asymmetry for investors.

#### System Prompt Design

The prompt instructs the model to act as a financial economist, score exactly four dimensions, and return **only** a JSON object with no additional text. This structured output format enables deterministic parsing and near-100% parse success rate across 11,269 filings. The full prompt is in [`week2_llama_inference.py`](week2_llama_inference.py) (`SYSTEM_PROMPT` constant).

#### Batch Preparation and SLURM Submission

Midway3 limits each account to 12 concurrent array jobs. Shards of 1,000 filings each produce 12 batch files for the full 11,269-filing dataset:

```bash
# On Midway3
source ${SCRATCH}/vllm_env/bin/activate

# Step 6a: Build JSONL batch files
python scripts/prepare_llama_batches.py \
    --data-root ${SCRATCH}/10k_data/10k-project \
    --shard-size 1000
ls ${SCRATCH}/10k_data/10k-project/llm_batches/ | wc -l   # should be 12

# Step 6b: Submit job array (max 3 concurrent to respect GPU quota)
sbatch --array=0-11%3 submit_llama.sh
squeue -u $USER   # monitor; each shard takes ~20 min on V100
```

SLURM configuration ([`submit_llama.sh`](submit_llama.sh)):
- 1 GPU, 8 CPUs, 48 GB RAM per array task
- 3-hour time limit per shard
- `VLLM_ATTENTION_BACKEND=FLASHINFER` for memory efficiency

For large re-runs (e.g., if shard size changes), [`submit_llama_full.sh`](submit_llama_full.sh) automates sequential batch submission — it submits 12 jobs, waits for completion, then submits the next 12.

**Output:** `llm_out/results_000.jsonl` through `results_011.jsonl`  
Each line:
```json
{"gvkey": "001722", "fyear": 2015, "ticker": "AAPL",
 "scores": {"management_optimism": 7, "guidance_specificity": 5,
            "uncertainty_hedging": 3, "risk_framing": 2}}
```
Parse success rate: **100%** (11,269 / 11,269) — greedy decoding + structured prompt eliminates malformed outputs.

Download results to local:
```bash
scp -r <cnetid>@midway3.rcc.uchicago.edu:${SCRATCH}/10k_data/10k-project/llm_out/ \
    "data/llm_out/"
```

#### Key Resources

- Model card: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- vLLM documentation: https://docs.vllm.ai
- Llama 3 technical report: Dubey et al. (2024), *arXiv:2407.21783*

---

### Step 7 — Track 2 & 4: FinBERT + Sentence-BERT (Midway3, SLURM)

Runs two neural NLP models over all 11,269 MD&A texts on Midway3 GPU nodes, producing the Track 2 and Track 4 features used in the regressions.

Track 2 uses FinBERT (`ProsusAI/finbert`), a BERT model fine-tuned on financial news and SEC filings. It scores each sentence as positive, negative, or neutral, and these are aggregated to the filing level as `fb_net = P(positive) − P(negative)`. Unlike the LM Dictionary, FinBERT understands context and negation — the phrase "not a liability" scores as positive rather than negative.

Track 4 uses Sentence-BERT (`all-mpnet-base-v2`) to encode each MD&A as a single 768-dimensional semantic vector. Comparing a firm's vector in year *t* to year *t−1* via cosine similarity yields a novelty measure: `embed_novelty = 1 − cosine_similarity`. This captures whether management is saying something substantively new relative to the prior year, independent of whether the tone is optimistic or pessimistic.

```bash
# On Midway3
sbatch scripts/submit_finbert.sh
squeue -u $USER
```

Download results to local:
```bash
# From Mac
scp <cnetid>@midway3.rcc.uchicago.edu:${SCRATCH}/10k_data/10k-project/processed/finbert_scores.parquet data/
scp <cnetid>@midway3.rcc.uchicago.edu:${SCRATCH}/10k_data/10k-project/processed/embed_similarity.parquet data/
```

---

### Step 8 — CAR Computation (Local Mac)

Computes cumulative abnormal return (CAR) around two event dates. The primary event is `date_filed` — the date the 10-K is formally submitted to the SEC and first available to the public. For each firm-year, the script sums market-adjusted daily returns (firm return minus the CRSP value-weighted market return) over the [−1, +1] trading-day window around `date_filed`, producing `car_filed_1_1`. A [−3, +3] window is also computed for robustness.

The earnings announcement date (`rdq`) CAR is computed separately and used only as a robustness check. Because the 10-K text is not yet public at the time of the earnings announcement — the filing typically comes several weeks later — any NLP predictability in the `rdq` window reflects text being correlated with earnings news quality, not a causal effect of the text itself on market prices.

```bash
# CAR around date_filed (primary outcome)
python scripts/compute_car_filing.py
# Output: data/car_filing_date.parquet

# CAR around rdq (robustness check) is computed inside analysis_track124.ipynb
```

---

### Step 9 — Analysis Notebook (Local Mac)

`analysis_full.ipynb` is the single end-to-end analysis notebook. It loads all raw inputs directly (master panel, LM scores, FinBERT scores, Sentence-BERT embeddings, CRSP returns, Llama JSONL output), computes CAR around both `date_filed` (primary) and `rdq` (robustness) inline, merges all four NLP tracks into one panel, and runs the complete econometric analysis.

The notebook is structured in 15 sections: data loading → CAR computation → track merging → z-score standardization → descriptive statistics → OLS regressions (M1–M4 standalone, MC horse-race, MD wider window) → incremental F-test → coefficient plot → CAR-by-quintile → results export → VIF multicollinearity check → first-difference causal identification → exploratory IV/2SLS → causal summary → robustness check with rdq event window.

The primary outcome throughout is `car_filed_1_1` (CAR[−1,+1] around `date_filed`). Models M1–M4 test each track in isolation; Model MC enters all eight NLP variables simultaneously to test which signals survive head-to-head. The first-difference estimator removes all time-invariant firm heterogeneity by regressing Δ(CAR_filed) on Δ(NLP scores). An exploratory IV/2SLS using leave-one-out industry-year peer means is also reported, with an explicit caveat that the exclusion restriction may not hold.

```bash
jupyter notebook analysis_full.ipynb
```

**Rendered notebook (no setup required):** [View on nbviewer](https://nbviewer.org/github/yulinwang123/10k-llm-event-study/blob/main/analysis_full.ipynb)

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

## Verifiable Trail of Work

### Midway3 SLURM Job Records

Slurm job IDs for all runs (pilot test + production), as required for course verification.

| Job ID | Description | Script | Array | Outcome |
|--------|-------------|--------|-------|---------|
| `49440971` | **Pilot test** — small-sample to verify pipeline before full production | `week2_llama_inference.py` | Single job | COMPLETED (exit 0:0) — confirmed vLLM + V100 + float16 config works |
| `49433115` | **Production** — Track 2+4: FinBERT + Sentence-BERT | `scripts/midway3_finbert_embed.py` via `scripts/submit_finbert.sh` | Single job | COMPLETED (exit 0:0) — produced `finbert_scores.parquet` and `embed_similarity.parquet` |
| `49442040` | **Production** — Track 3: Llama-3.1-8B full dataset | `week2_llama_inference.py` via `submit_llama.sh` | `--array=0-11%3` (12 tasks, all COMPLETED exit 0:0) | 11,269 / 11,269 filings scored, 0 parse failures |

**Verification commands (run on Midway3):**
```bash
# Check all three job IDs
sacct -j 49440971,49433115,49442040 \
  --format=JobID,JobName,Partition,State,Elapsed,AllocCPUS,AllocGRES,Start,End

# Confirm Llama output files exist (12 shards)
ls -lh /scratch/midway3/yulinwang/10k_data/10k-project/llm_out/results_*.jsonl
wc -l  /scratch/midway3/yulinwang/10k_data/10k-project/llm_out/results_*.jsonl
```

---

### AWS Usage Verification

#### EC2 — CloudTrail Launch Records

The following command retrieves all EC2 instance launch events tied to this AWS Academy account. The output shows multiple `RunInstances` events attributed to `user5004207=oliviawang011231@gmail.com`, confirming EC2 was used for the EDGAR MD&A download job:

```bash
aws cloudtrail lookup-events \
    --lookup-attributes AttributeKey=EventName,AttributeValue=RunInstances \
    --query 'Events[*].{Time:EventTime,User:Username,Event:EventName}' \
    --output table
```

#### S3 — File Inventory with Timestamps

The S3 bucket `yulinwang-10k-llm` holds all pipeline outputs. The listing below confirms 11,404 MD&A text files plus all raw and processed parquet files, with creation timestamps between 2026-05-06 and 2026-05-07:

```bash
# Full recursive listing (shows timestamps + file sizes)
aws s3 ls s3://yulinwang-10k-llm/10k-project/ --recursive --human-readable

# Total MD&A file count
aws s3 ls s3://yulinwang-10k-llm/10k-project/filings/ --recursive | wc -l
# Expected: ~11,404
```

Key files in S3 with upload timestamps:

| File | Size | Timestamp |
|------|------|-----------|
| `raw/sp1500_universe.parquet` | 33.6 KiB | 2026-05-06 20:08 |
| `raw/compustat.parquet` | 1.2 MiB | 2026-05-06 20:08 |
| `raw/crsp_daily.parquet` | 22.2 MiB | 2026-05-06 20:09 |
| `raw/crsp_market.parquet` | 88.5 KiB | 2026-05-06 20:09 |
| `raw/mda_metadata.parquet` | 465.4 KiB | 2026-05-07 10:15 |
| `processed/master_panel.parquet` | 933.3 KiB | 2026-05-07 15:03 |
| `processed/lm_scores.parquet` | 411.8 KiB | 2026-05-07 20:48 |
| `filings/` (11,404 MD&A text files) | 30–120 KiB each | 2026-05-06 to 2026-05-07 |

**GitHub commit history** documents iterative development across all stages — data collection scripts, Midway3 environment debugging, inference script revisions, and analysis notebooks. View at: `https://github.com/<your-repo>/commits/main`

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

**IV / 2SLS (exploratory):** Instrument each firm's LLM optimism with the leave-one-out mean of peers in the same 2-digit SIC × year cell. First-stage F = 116 (strong instrument on relevance). Reported as an exploratory exercise; the exclusion restriction is not cleanly satisfied because industry-wide shocks that drive peer optimism can also directly affect individual firm CAR through real economic channels, not only through the text channel. Results should be interpreted with this caveat in mind.

### Robustness Checks

**CAR[−3,+3]:** Wider event window (Model MD) — confirms results are not sensitive to window width.

**rdq event window:** Recompute CAR around the earnings announcement date (`rdq`) instead of the 10-K filing date. Because the 10-K text is not yet publicly available at earnings announcement time, any NLP predictability in this window reflects correlation with underlying earnings quality rather than a market reaction to the text itself. Including it as a robustness check tests whether the filing-date results are contaminated by that reverse-causality channel.

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
├── analysis_full.ipynb             # Step 9: All tracks + causal ID + robustness (single notebook)
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
