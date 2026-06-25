# Attention Allocation in Earnings Announcements
### Hard News vs. Managerial Narrative: A Double Machine Learning Approach

**Author:** Yulin Wang, University of Chicago  
**Data:** WRDS (CIQ Transcripts + IBES + CRSP + Compustat), 2010–2023

---

## Research Questions

When a firm reports earnings, two signals arrive simultaneously: a hard number (earnings surprise) and a managerial narrative (the earnings call). This project asks:

**Project A (baseline):** After controlling for managerial narrative framing, does earnings surprise (SUE) still have an independent marginal effect on stock returns (CAR)?

**Project B (main paper):** When do investors rely more on hard numbers vs. narrative? We test three moderators — cognitive load (earnings complexity), investor sophistication (institutional ownership), and information intermediation (analyst coverage).

The key methodological contribution is using **Double Machine Learning (DML)** to partial out the text channel from both the treatment (SUE) and the outcome (CAR), recovering a clean causal estimate of the hard-news earnings response coefficient (ERC).

---

## Conceptual Framework

```
Partially Linear Model:
  CAR = SUE · θ + g(Z) + ε        [outcome equation]
  SUE = m(Z) + v                   [first stage]

  Y  = CAR[-1,+1]   around earnings call date
  D  = SUE = (actual EPS − median forecast) / forecast stdev
  Z  = text features (LM tone, uncertainty, fog index)
       + firm controls (size, btm, leverage, roa, eps_vol)
  θ  = marginal ERC after partialling out narrative

Project B adds HTE:
  θ(X) = θ_0 + τ · X   where X ∈ {eps_vol, io_pct, numest}
```

**Intuition:** If management frames bad news with optimistic language, OLS conflates the narrative effect with the number effect. DML separates them. The paper's contribution is not re-estimating ERC — it's asking whether the narrative channel is an *independent* confound, and for whom it matters most.

---

## Data Sources

| Source | Table | Content |
|--------|-------|---------|
| **CIQ (via WRDS)** | `wrds_transcript_detail` | Earnings call events (keydeveventtypeid=48), 2010–2023 |
| **CIQ** | `ciqtranscriptcomponent` | Full transcript text (prepared remarks + Q&A) |
| **CIQ** | `wrds_gvkey`, `wrds_ticker` | companyid → GVKEY / IBES ticker linkage |
| **IBES** | `statsum_epsus` | Quarterly consensus forecasts; `anndats_act` = announcement date |
| **CRSP** | `dsf`, `dsi` | Daily returns; market index for CAR computation |
| **CRSP** | `ccmxpf_lnkhist`, `iclink` | GVKEY → PERMNO linkage |
| **Compustat** | `fundq` | Quarterly firm controls (size, leverage, ROA, special items) |
| **13F** | `tr_13f.s34` | Institutional ownership (HTE moderator) |

**Event definition:** Quarterly earnings call date (`mostimportantdateutc` from CIQ), matched to IBES announcement date (`anndats_act`) within ±7 days.

---

## Pipeline

### Data Collection (WRDS — requires UChicago VPN)

```
Step 1: pull_transcripts.py
  CIQ wrds_transcript_detail (keydeveventtypeid=48)
  + ciqtranscriptcomponent (full text by transcriptid)
  + wrds_gvkey + wrds_ticker
  → data/transcript_events.parquet      [event metadata + word counts]
  → data/transcripts_tmp/chunk_*.parquet [full text, 16 chunks × ~50K transcripts]

Step 2: pull_ibes_sue.py
  ibes.statsum_epsus (fpi='6', quarterly)
  Pre-announcement consensus: statpers < anndats_act, take MAX(statpers)
  SUE = (actual - medest) / stdev
  → data/ibes_sue.parquet

Step 3: pull_crsp_car.py
  crsp.iclink: IBES ticker → PERMNO
  crsp.dsf + crsp.dsi: daily returns + market return
  CAR[-1,+1] and CAR[+2,+60] (market-adjusted compounded)
  → data/crsp_car.parquet

Step 4: pull_controls.py
  comp.fundq: size, btm, leverage, roa, special_items, eps_vol
  tr_13f.s34: institutional ownership pct (io_pct)
  → data/firm_controls.parquet
  → data/io_pct.parquet

Step 5: merge_event_panel.py
  Join all above on (ticker, fpedats) with ±7-day tolerance
  → data/event_panel.parquet
```

### Text Features (local, no WRDS needed)

```
Step 6: compute_text_features.py
  Reads chunk files one at a time (avoids OOM)
  LM (Loughran-McDonald) dictionary: tone, uncertainty, litigious
  Gunning Fog Index: readability / complexity
  Computed separately for prepared_text and qa_text
  → data/text_features.parquet
```

### Estimation

```
Step 7a: study_a_ols.py
  4 specifications (raw ERC → +controls → +text → +firm/time FE)
  Clustered SE by firm
  → results/study_a_ols_table.csv

Step 7b: study_a_dml.py
  DML partialling-out, K=5 cross-fitting
  Nuisance models: Lasso (primary) + Ridge + Random Forest (robustness)
  Influence-function SE
  → results/study_a_dml_results.csv
  → results/study_a_dml_summary.txt
  → data/study_a_residuals.parquet     [Y_resid, D_resid → input for Project B]
```

### Run Order

```bash
cd ~/Desktop/10K

# Step 1–5: WRDS data pull (requires VPN, ~2 hours total)
python -u scripts/pull_transcripts.py
python -u scripts/pull_ibes_sue.py
python -u scripts/pull_crsp_car.py
python -u scripts/pull_controls.py
python -u scripts/merge_event_panel.py

# Step 6: Text features (local, ~30 min)
python -u scripts/compute_text_features.py

# Step 7: Estimation
python -u scripts/study_a_ols.py
python -u scripts/study_a_dml.py
```

Or run everything at once:
```bash
bash scripts/run_pipeline.sh
```

---

## Variable Definitions

| Variable | Role | Source | Notes |
|----------|------|--------|-------|
| `sue_win` | Treatment D | IBES | (actual − medest) / stdev, winsorized 1/99% |
| `car_m1_p1_win` | Outcome Y | CRSP | CAR[−1,+1] market-adjusted, winsorized |
| `car_p2_p60` | Outcome Y2 | CRSP | CAR[+2,+60] post-announcement drift |
| `prep_lm_tone` | Confounder Z | CIQ + LM | (pos−neg)/total in prepared remarks |
| `prep_lm_uncertainty` | Confounder Z | CIQ + LM | uncertainty word ratio |
| `prep_fog_index` | Confounder Z | CIQ | Gunning Fog readability |
| `qa_lm_tone` | Confounder Z | CIQ + LM | tone in Q&A section |
| `size` | Control | Compustat | log(total assets) |
| `btm` | Control | Compustat | book-to-market |
| `leverage` | Control | Compustat | total debt / assets |
| `roa` | Control | Compustat | net income / assets |
| `eps_vol` | HTE moderator (B) | IBES | rolling 8-quarter stdev of actual EPS |
| `io_pct` | HTE moderator (B) | 13F | % shares held by institutions |
| `numest` | HTE moderator (B) | IBES | analyst count |

---

## Repository Structure

```
10K/
├── scripts/
│   ├── pull_transcripts.py        # Step 1: CIQ earnings call transcripts
│   ├── pull_ibes_sue.py           # Step 2: IBES quarterly SUE
│   ├── pull_crsp_car.py           # Step 3: CRSP CAR around call date
│   ├── pull_controls.py           # Step 4: Compustat controls + 13F IO
│   ├── merge_event_panel.py       # Step 5: merge into event panel
│   ├── compute_text_features.py   # Step 6: LM dictionary + Fog Index
│   ├── study_a_ols.py             # Step 7a: OLS baseline ERC
│   ├── study_a_dml.py             # Step 7b: DML partialling-out
│   └── run_pipeline.sh            # Run all steps in order
│
├── data/                          # gitignored
│   ├── transcript_events.parquet  # CIQ event metadata
│   ├── transcripts_tmp/           # Full text chunks (keep for Project B)
│   │   └── chunk_0000–0015.parquet
│   ├── ibes_sue.parquet
│   ├── crsp_car.parquet
│   ├── firm_controls.parquet
│   ├── io_pct.parquet
│   ├── text_features.parquet
│   ├── event_panel.parquet        # master panel
│   └── study_a_residuals.parquet  # DML residuals → Project B input
│
├── results/
│   ├── study_a_ols_table.csv
│   ├── study_a_dml_results.csv
│   └── study_a_dml_summary.txt
│
└── README.md
```

---

## Key Design Decisions

**Why earnings calls, not 10-Ks?**  
10-K filings arrive weeks after earnings announcements. The earnings call is the simultaneous release — hard number (EPS) and narrative arrive together, so investor attention allocation is directly testable.

**Why DML, not OLS with text controls?**  
OLS with text controls only removes the linear projection of text on returns. DML allows the confounding function g(Z) to be arbitrary and estimated via machine learning, removing both linear and nonlinear text confounds before identifying θ.

**Why LM dictionary, not LLM, for text features (in this step)?**  
LM features are used as *confounders to be partialled out*, not as the object of interest. For this role, coverage and speed matter more than nuance. LLM-based features are planned for Project B where the narrative quality measure is itself the quantity of interest.

**Why save `study_a_residuals.parquet`?**  
Project B's HTE analysis uses (Ỹ, D̃) — the DML residuals from Project A — as the outcome and treatment in the heterogeneous effects regression. This avoids re-running the expensive nuisance estimation.

---

## Dependencies

```bash
pip install wrds pandas numpy scipy statsmodels scikit-learn pyarrow
```

---

## References

- Chernozhukov, V. et al. (2018). Double/debiased machine learning. *Econometrics Journal*, 21(1), C1–C68.
- Loughran, T. & McDonald, B. (2011). When is a liability not a liability? *Journal of Finance*, 66(1), 35–65.
- Ball, R. & Brown, P. (1968). An empirical evaluation of accounting income numbers. *Journal of Accounting Research*, 6(2), 159–178.
- Bushee, B., Gow, I. & Taylor, D. (2018). Linguistic complexity in firm disclosures. *Journal of Accounting Research*, 56(1), 33–82.
