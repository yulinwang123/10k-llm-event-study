# Earnings Announcements: Hard Numbers vs. Managerial Narrative

**Author:** Yulin Wang, University of Chicago  
**Data:** WRDS (CIQ Transcripts + IBES + CRSP + Compustat + 13F), 2010-2023

---

## Research Question

Once narrative is measured more semantically, how much independent weight does the market place on hard earnings surprise versus managerial narrative, and does this weighting vary with the information environment?

Earnings announcements contain both quantitative information, such as standardized unexpected earnings (SUE), and qualitative interpretation, such as managerial tone, uncertainty, forward-looking language, defensiveness, and Q&A discussion. The current dictionary / word-bag measures are a baseline proxy for the narrative channel. The planned next step is to replace them with LLM-extracted semantic narrative measures and ask whether the estimated relative weight on hard earnings news versus narrative changes.

The goal is not to show that LLMs predict returns better for their own sake. The economic question is whether richer narrative measurement changes what we infer about how investors price hard earnings numbers and managerial explanations.

---

## Current Status

This repository currently implements the data pipeline and baseline empirical analysis.

**Implemented:**

- Build a firm-quarter earnings announcement panel from CIQ, IBES, CRSP, Compustat, and 13F.
- Compute announcement-window CAR around earnings announcement / call dates.
- Construct SUE from IBES pre-announcement consensus forecasts.
- Extract word-bag narrative measures from earnings call transcripts using Loughran-McDonald dictionary features and readability measures.
- Estimate baseline ERC regressions with controls, text measures, and fixed effects.
- Estimate DML / cross-fitting specifications as a robustness and adjustment tool.
- Estimate heterogeneity in hard-news pricing by earnings volatility, analyst coverage, and institutional ownership.

**Planned next step:**

- Run transcript-level LLM inference to extract structured semantic narrative measures.
- Re-run the same empirical pipeline by replacing word-bag narrative proxies with LLM semantic measures.
- Compare the relative pricing weights on hard earnings surprise and managerial narrative.

---

## Empirical Design

### Step 1: Hard-News Benchmark and Word-Bag Narrative Baseline

The first step estimates the standard ERC relationship between SUE and announcement-window CAR:

```text
CAR_i = theta * SUE_i + controls + FE + error_i
```

I then add traditional word-bag narrative proxies from the earnings call:

```text
CAR_i = theta_LM * SUE_i + beta_LM * Narrative_LM_i + controls + FE + error_i
```

Current word-bag narrative measures include:

- prepared remarks tone, negative ratio, uncertainty, litigious language, fog index, and word count
- Q&A tone, uncertainty, fog index, and word count

This establishes how much the hard-news coefficient changes once the baseline narrative proxy is included, and whether the dictionary-based narrative proxy has independent explanatory power.

### Step 2: Planned LLM Semantic Narrative Measures

The planned LLM step replaces `Narrative_LM` with structured semantic measures extracted from earnings call transcripts:

```text
CAR_i = theta_LLM * SUE_i + beta_LLM * Narrative_LLM_i + controls + FE + error_i
```

Candidate LLM variables include:

- contextual sentiment
- uncertainty / hedging
- forward-looking tone
- defensiveness
- Q&A informativeness or evasiveness
- management optimism conditional on reported earnings

The key comparison is:

```text
theta:      hard-news ERC with no narrative measure
theta_LM:   hard-news ERC after controlling for word-bag narrative
theta_LLM:  hard-news ERC after controlling for LLM semantic narrative

beta_LM:    independent pricing weight on word-bag narrative
beta_LLM:   independent pricing weight on LLM semantic narrative
```

If LLM-based narrative measures reduce the hard-news ERC more than word-bag measures, or explain substantially more return variation, that would suggest traditional dictionary measures understate the role of narrative. If the ERC remains stable even after LLM narrative controls, that would suggest hard earnings news has an independent pricing role beyond managerial narrative.

### Step 3: Information-Environment Heterogeneity

The heterogeneity analysis asks whether the market's relative weight on hard news versus narrative changes with:

- **Cognitive load:** earnings volatility (`eps_vol`)
- **Information intermediation:** analyst coverage (`numest`)
- **Investor sophistication:** institutional ownership (`io_pct`)

A future LLM-based version can estimate:

```text
CAR_i = theta * SUE_i
      + beta * Narrative_i
      + theta_X * SUE_i * X_i
      + beta_X * Narrative_i * X_i
      + controls + FE + error_i
```

where `X` is an information-environment moderator.

### Step 4: DML / Cross-Fitting Robustness

Double Machine Learning is used as a robustness and adjustment tool, not as the main contribution. The current DML specification partials out observed text features, firm controls, and time effects:

```text
CAR_i = theta * SUE_i + g(Z_i) + error_i
SUE_i = m(Z_i) + residual_i
```

I interpret these estimates conservatively as narrative-adjusted / control-adjusted ERCs, not as fully causal estimates.

---

## Data Sources

| Source | Content |
|---|---|
| CIQ via WRDS | Earnings call events and transcript text |
| IBES | Quarterly analyst consensus forecasts and actual EPS |
| CRSP | Daily returns and market returns |
| Compustat | Firm fundamentals and controls |
| 13F | Institutional ownership |

**Event definition:** Quarterly earnings call date from CIQ, matched to the IBES announcement date within a +/- 7 day window.

**Outcome:** `car_m1_p1_win`, market-adjusted CAR over [-1, +1] around the announcement / call date.

**Hard-news variable:** `sue_win`, winsorized standardized unexpected earnings:

```text
SUE = (actual EPS - median analyst forecast) / forecast standard deviation
```

---

## Pipeline

### Data Collection

```text
scripts/pull_transcripts.py
  CIQ earnings call events and transcript text

scripts/pull_ibes_sue.py
  IBES quarterly consensus forecasts and SUE

scripts/pull_crsp_car.py
  CRSP announcement-window and post-announcement CAR

scripts/pull_controls.py
  Compustat controls and 13F institutional ownership

scripts/merge_event_panel.py
  Merge all sources into data/event_panel.parquet
```

### Text Features

```text
scripts/compute_text_features.py
  Current baseline word-bag narrative proxy
  Loughran-McDonald dictionary + readability features
```

### Estimation

```text
scripts/study_a_ols.py
  Baseline ERC and word-bag narrative regressions

scripts/study_a_dml.py
  DML / cross-fitting robustness

scripts/study_b_hte.py
  Current heterogeneity analysis for hard-news pricing
```

Run order:

```bash
bash scripts/run_pipeline.sh
python -u scripts/study_b_hte.py
```

WRDS-dependent data collection requires institutional WRDS access / VPN. Raw WRDS data are not included in this public repository.

---

## Current Preliminary Findings

The current pilot sample contains approximately 65,000 firm-quarter observations across more than 2,700 firms.

Using word-bag narrative proxies:

- SUE is strongly associated with announcement-window CAR.
- A one-standard-deviation increase in SUE is associated with roughly a 0.50-0.54 percentage point increase in CAR[-1,+1].
- Adding dictionary-based text features reduces the hard-news coefficient only slightly.
- DML / cross-fitting estimates also leave the word-bag-adjusted ERC near 0.005.

Current heterogeneity results:

- Higher earnings volatility is associated with a lower ERC.
- Higher analyst coverage is associated with a higher ERC.
- Institutional ownership has no clear linear effect in the current specification.

These results motivate the LLM step rather than conclude the paper. The current word-bag results suggest that dictionary narrative controls do not materially change the estimated hard-news ERC. The next question is whether that conclusion survives when narrative is measured with richer semantic signals from an LLM.

---

## Repository Map

```text
scripts/
  pull_transcripts.py       CIQ earnings call events and transcript text
  pull_ibes_sue.py          IBES SUE construction
  pull_crsp_car.py          CRSP CAR construction
  pull_controls.py          Compustat controls and 13F IO
  merge_event_panel.py      Main firm-quarter panel merge
  compute_text_features.py  LM dictionary and readability features
  study_a_ols.py            Baseline ERC regressions
  study_a_dml.py            DML / cross-fitting robustness
  study_b_hte.py            Heterogeneity analysis

results/
  writeup_draft.md          Current draft write-up
  study_a_ols_stats.txt     Baseline OLS summary
  study_a_dml_summary.txt   DML robustness summary
  study_b_blp_summary.txt   Heterogeneity summary

faculty_outreach_memo_en.md
  Short research memo aligned with the current framing
```

---

## Limitations and Next Steps

- Current narrative measures are dictionary-based proxies and may miss context, hedging, defensiveness, and semantic nuance.
- Daily returns identify the pricing of the earnings announcement package, not a clean causal effect of the call itself. Intraday data would allow a stronger design separating the press-release-to-call window from the call-window reaction.
- The planned LLM step will begin with a small audited pilot before scaling to the full transcript sample.
- LLM prompts should use only transcript text and must not condition on returns or future outcomes.

---

## Dependencies

```bash
pip install wrds pandas numpy scipy statsmodels scikit-learn pyarrow
```

---

## References

- Ball, R. & Brown, P. (1968). An empirical evaluation of accounting income numbers. *Journal of Accounting Research*, 6(2), 159-178.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning. *Econometrics Journal*, 21(1), C1-C68.
- Loughran, T. & McDonald, B. (2011). When is a liability not a liability? *Journal of Finance*, 66(1), 35-65.
- Nie, X. & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299-319.
