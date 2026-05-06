# Do LLMs Read 10-Ks Better Than Dictionaries?
### A Staggered Event Study of Financial Text Measures and Earnings Announcement Returns

**Course:** MACS 30113 — Big Data and Society  
**Author:** Yulin Wang, University of Chicago

---

## Research Question

Does replacing bag-of-words dictionary methods with large language model (LLM) inference on 10-K MD&A sections produce text-based sentiment measures that better predict cumulative abnormal returns (CARs) around earnings announcements?

---

## Motivation

Loughran & McDonald (2011) showed that word-count-based negativity in 10-K filings predicts stock returns. Yet dictionaries are blind to context: the same word carries different meaning depending on surrounding sentences, and they cannot detect whether management provides *specific* numerical guidance versus vague qualitative language. LLMs may capture these dimensions.

---

## Pipeline Overview

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

## Text Measures

| Track | Model | Key Metric | LLM Advantage |
|-------|-------|------------|---------------|
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
│   └── week2_text_measures.ipynb     # LM Dict + FinBERT + Llama prep
│
├── scripts/
│   ├── week2_llama_inference.py      # vLLM batch inference script
│   ├── submit_llama.sh               # SLURM job array (Midway3)
│   └── setup_midway3.sh              # One-time environment setup
│
├── proposal_10k_llm_eventstudy.tex   # Research proposal (LaTeX)
└── data_pilot/                       # Pilot data (gitignored except structure)
```

---

## Reproducing the LLM Inference (Midway3 / HPC)

```bash
# 1. Clone and set up environment on Midway3
git clone https://github.com/<your-username>/10k-llm-event-study
bash scripts/setup_midway3.sh

# 2. Download Llama-3.1-8B-Instruct (requires HuggingFace access)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir $SCRATCH/models/llama3-8b

# 3. Upload batch files and submit job array
scp data_pilot/llm_batches/batch_*.jsonl \
    midway3.rcc.uchicago.edu:$SCRATCH/llm_batches/
sbatch scripts/submit_llama.sh
```

---

## Dependencies

```bash
conda create -n llm_10k python=3.11
conda activate llm_10k
pip install vllm transformers torch accelerate huggingface_hub \
            wrds pandas pyarrow tqdm requests beautifulsoup4 lxml
```

---

## References

- Loughran, T. & McDonald, B. (2011). When is a liability not a liability? *Journal of Finance*, 66(1), 35–65.
- Touvron, H. et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. *arXiv*.
- Yang, Y. et al. (2020). FinBERT: A Pretrained Language Model for Financial Communications. *arXiv*.
