# Do LLMs Read 10-Ks Better Than Dictionaries?
### A Staggered Event Study of Financial Text Measures and Earnings Announcement Returns

**Course:** MACS 30113 — large-scale computing for the social sciences
**Author:** Yulin Wang, University of Chicago

---

## Research Question

Does replacing bag-of-words dictionary methods with large language model (LLM) inference on 10-K MD&A sections produce text-based sentiment measures that better predict cumulative abnormal returns (CARs) around earnings announcements?

---

## Motivation

Loughran & McDonald (2011) showed that word-count-based negativity in 10-K filings predicts stock returns. Yet dictionaries are blind to context: the same word carries different meaning depending on surrounding sentences, and they cannot detect whether management provides *specific* numerical guidance versus vague qualitative language. LLMs may capture these dimensions.

---

## High-Level Pipeline

```
SEC EDGAR (10-K filings)          WRDS (Compustat + CRSP)
        │                                   │
        ▼                                   ▼
  MD&A Extraction              Financial Controls + Returns
  (iXBRL-aware parser)         (gvkey → permno linkage)
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
              Master Panel (firm × year)
              stored on AWS S3
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    LM Dictionary   FinBERT     Llama-3.1-8B
    (word counts)  (neural      (zero-shot
                   sentiment)    scoring)
          └────────────┼────────────┘
                       ▼
            Staggered Event Study
            CAR[-1,+1] ~ TextMeasure
              + controls + firm FE
```

---

## Data Collection Pipeline

Data collection is split across three compute environments due to access restrictions and computational requirements.

### Step 1 — WRDS Pull (Local Mac, UChicago VPN required)

WRDS data is governed by a license that restricts access to whitelisted IPs. This step must run on a UChicago-networked machine (on campus or via VPN).

```bash
# Test mode: 5 tickers (AAPL, MSFT, GOOGL, AMZN, JPM), FY2018–2019
python scripts/local_wrds_pull.py --bucket <your-s3-bucket> --test

# Full run: S&P 1500, FY2010–2020
python scripts/local_wrds_pull.py --bucket <your-s3-bucket>
```

Uploads to S3:
```
s3://<bucket>/10k-project/raw/
├── sp1500_universe.parquet     # S&P 1500 constituent list
├── compustat.parquet           # Annual fundamentals + rdq (earnings date)
├── ccm_link.parquet            # gvkey → permno mapping (CRSP-Compustat)
├── crsp_daily.parquet          # Daily stock returns
└── crsp_market.parquet         # CRSP value-weighted market index
```

### Step 2 — SEC EDGAR Download (AWS EC2 or local)

10-K filings are public. This step can run anywhere, but EC2 is recommended for the full S&P 1500 run (multi-threaded, ~16,500 filings).

```bash
# Test mode
python scripts/ec2_edgar_download.py --bucket <your-s3-bucket> --test

# Full run on EC2 (16 threads recommended)
python scripts/ec2_edgar_download.py --bucket <your-s3-bucket> --workers 16
```

Uploads to S3:
```
s3://<bucket>/10k-project/
├── raw/mda_metadata.parquet        # Filing metadata + download status
└── filings/{ticker}/{year}/        # Raw MD&A text files (.txt)
```

The downloader respects the SEC's rate limit (≤10 req/sec), handles retries, and resumes interrupted runs by checking S3 before re-downloading.

### Step 3 — Merge Master Panel (local or EC2)

Joins WRDS fundamentals with MD&A metadata into a single firm × year panel.

```bash
python scripts/merge_panel.py --bucket <your-s3-bucket> --test
python scripts/merge_panel.py --bucket <your-s3-bucket>
```

Output:
```
s3://<bucket>/10k-project/processed/master_panel.parquet
```

Key columns: `gvkey`, `permno`, `ticker`, `fyear`, `rdq` (earnings announcement date), `s3_key` (path to MD&A text), `log_assets`, `bm_ratio`, `roa`, `leverage`.

### Step 4 — LLM Inference (Midway3 HPC, A100 GPU)

Llama-3.1-8B-Instruct scores each MD&A on four theory-driven dimensions via
[`week2_llama_inference.py`](week2_llama_inference.py), submitted as a SLURM
job array with [`submit_llama.sh`](submit_llama.sh).

#### 4a. One-Time Environment Setup

SSH into Midway3, then run the setup script — or follow the steps below manually
(the manual path is more robust given Midway3's module naming quirks):

```bash
ssh <cnetid>@midway3.rcc.uchicago.edu
```

**Install Miniconda to home directory** (`module load python` does not expose
`conda` on Midway3 login nodes):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda tos accept --override-channels --channel defaults
conda tos accept --override-channels --channel conda-forge
```

**Create the conda environment in `$SCRATCH`** (home quota is only ~30 GB;
vLLM + PyTorch exceed it):

```bash
SCRATCH="/scratch/midway3/${USER}"
export PIP_CACHE_DIR="${SCRATCH}/pip_cache"
export TMPDIR="${SCRATCH}/tmp"
mkdir -p $PIP_CACHE_DIR $TMPDIR

conda create -y -p ${SCRATCH}/vllm_env python=3.11
conda activate ${SCRATCH}/vllm_env
```

**Load CUDA and install PyTorch + vLLM:**

```bash
module load cuda/12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install setuptools_scm numpy
pip install vllm --no-build-isolation
pip install transformers accelerate huggingface_hub
```

> **Gotcha — V100 vs A100:** If your job lands on a V100 (compute capability 7.0),
> vLLM will crash with `Bfloat16 is only supported on GPUs with compute capability >= 8.0`.
> The fix is already in [`submit_llama.sh`](submit_llama.sh): add
> `--constraint=a100` to the `sbatch` call to target A100 nodes only.

#### 4b. Download Model Weights

Request access to `meta-llama/Meta-Llama-3.1-8B-Instruct` on
[HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct),
then:

```bash
conda activate ${SCRATCH}/vllm_env
huggingface-cli login          # paste your HF token when prompted
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir ${SCRATCH}/models/Meta-Llama-3.1-8B-Instruct \
    --local-dir-use-symlinks False
```

Download is ~16 GB and takes about 15 minutes on Midway3's fast scratch I/O.

#### 4c. Upload Batch Files and Submit

The notebook [`notebooks/week2_text_measures.ipynb`](notebooks/week2_text_measures.ipynb)
writes JSONL shards to `data_pilot/llm_batches/batch_NNN.jsonl`. Upload them
and submit the job array:

```bash
# From your local Mac
scp data_pilot/llm_batches/batch_*.jsonl \
    midway3.rcc.uchicago.edu:${SCRATCH}/llm_batches/

# On Midway3 — check shard count, then update --array in submit_llama.sh
ls ${SCRATCH}/llm_batches/batch_*.jsonl | wc -l

# Submit (pilot: 3 shards; full run: adjust --array=0-249%10)
sbatch --constraint=a100 submit_llama.sh
```

Monitor progress:
```bash
squeue -u $USER
cat logs/llama_<jobid>_0.out    # output for shard 0
```

#### 4d. Output Format

Each shard produces a `.jsonl` results file in `${SCRATCH}/llm_out/`. Each line:

```json
{"ticker": "AAPL", "fyear": 2019, "accession_no": "0000320193-20-000001",
 "management_optimism": 7, "guidance_specificity": 9,
 "uncertainty_hedging": 3, "risk_framing": 2}
```

The notebook merges these back into the master panel via the `accession_no` key.

### Step 5 — Text Measures + Event Study (Jupyter)

```bash
# Track 1 (LM Dictionary) + Track 2 (FinBERT) + merge Llama results
jupyter notebook notebooks/week2_text_measures.ipynb

# Week 3: CAR calculation + regression
jupyter notebook notebooks/week3_event_study.ipynb
```

---

## Text Measures

| Track | Model | Key Metric | Advantage over LM Dict |
|-------|-------|------------|------------------------|
| 1 | Loughran-McDonald (2011) | `lm_net_sent` = (pos−neg)/total | Baseline |
| 2 | FinBERT (`ProsusAI/finbert`) | `fb_net` = P(pos)−P(neg) | Contextual sentence-level |
| 3 | Llama-3.1-8B-Instruct (vLLM) | 4-dim score vector | Guidance specificity, risk framing |

The key differentiator for Track 3 is **`guidance_specificity`** — whether management provides concrete numerical targets — which has no LM Dictionary equivalent and may reduce information asymmetry.

---

## Sample

- **Pilot:** 20 S&P 500 firms × FY2018–2020 = 59 firm-years
- **Full:** S&P 1500 × FY2010–2020 ≈ 16,500 firm-years (in progress)

---

## Repository Structure

```
├── notebooks/
│   ├── week1_pilot_fixed.ipynb       # Pilot data collection (SEC + WRDS)
│   ├── week1_data_collection.ipynb   # Full-scale data collection
│   ├── week2_text_measures.ipynb     # LM Dict + FinBERT + Llama prep
│   └── week3_event_study.ipynb       # CAR calculation + regressions (TODO)
│
├── scripts/
│   ├── local_wrds_pull.py            # Step 1: WRDS → S3 (run on VPN)
│   ├── ec2_edgar_download.py         # Step 2: SEC EDGAR → S3 (EC2 / local)
│   └── merge_panel.py                # Step 3: S3 merge → master panel
│
├── week2_llama_inference.py          # Step 4: vLLM batch inference (Midway3)
├── submit_llama.sh                   # SLURM job array submission
├── setup_midway3.sh                  # One-time Midway3 environment setup
│
├── proposal_10k_llm_eventstudy.tex   # Research proposal (LaTeX)
└── data_pilot/                       # Pilot data (gitignored except structure)
```

Key files with inline links:
[`scripts/local_wrds_pull.py`](scripts/local_wrds_pull.py) ·
[`scripts/ec2_edgar_download.py`](scripts/ec2_edgar_download.py) ·
[`scripts/merge_panel.py`](scripts/merge_panel.py) ·
[`week2_llama_inference.py`](week2_llama_inference.py) ·
[`submit_llama.sh`](submit_llama.sh) ·
[`setup_midway3.sh`](setup_midway3.sh)

---

## Infrastructure

| Task | Where | Why |
|------|-------|-----|
| WRDS data pull | Local Mac (UChicago VPN) | License restricts to whitelisted IPs |
| SEC EDGAR download | AWS EC2 (`t3.large`) or local | Multi-threaded, public data, no IP restriction |
| Data storage | AWS S3 | Shared across Mac / EC2 / Midway3 |
| LLM inference | Midway3 A100 GPU (account: `macs30113`) | 80 GB VRAM, vLLM batch throughput |
| Analysis / notebooks | Local Mac | Interactive exploration |

**Midway3 environment notes:**
- Conda must be installed manually via Miniconda (module load python does not expose `conda`)
- Create the conda env under `$SCRATCH`, not `$HOME` — home quota (30 GB) is too small for vLLM + PyTorch
- Always add `--constraint=a100` to `sbatch`; V100 nodes do not support bfloat16 and will crash
- Set `PIP_CACHE_DIR` and `TMPDIR` to `$SCRATCH` before installing to avoid home quota overflow

---

## Dependencies

**Local / EC2:**
```bash
pip install wrds boto3 pandas pyarrow requests beautifulsoup4 lxml tqdm
```

**Midway3 (GPU environment in `$SCRATCH`):**
```bash
# Must install PyTorch before vLLM to avoid CUDA version conflicts
module load cuda/12.4
conda create -p $SCRATCH/vllm_env python=3.11
conda activate $SCRATCH/vllm_env
export PIP_CACHE_DIR=$SCRATCH/pip_cache
export TMPDIR=$SCRATCH/tmp
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install setuptools_scm numpy
pip install vllm --no-build-isolation
pip install transformers accelerate huggingface_hub
```

---

## References

- Loughran, T. & McDonald, B. (2011). When is a liability not a liability? *Journal of Finance*, 66(1), 35–65.
- Touvron, H. et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. *arXiv*.
- Yang, Y. et al. (2020). FinBERT: A Pretrained Language Model for Financial Communications. *arXiv*.
