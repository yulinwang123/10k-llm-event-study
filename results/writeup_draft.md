# Hard Earnings News and Managerial Narrative in Earnings Announcements
## Current Baseline Results and Planned LLM Extension

**Yulin Wang — University of Chicago**  
**Data: WRDS (CIQ + IBES + CRSP + Compustat + 13F), 2010-2023**

---

## 1. Research Question

This project studies how investors price hard earnings numbers and managerial narratives when both arrive during earnings announcements.

The main research question is:

> Once narrative is measured more semantically, how much independent weight does the market place on hard earnings surprise versus managerial narrative, and does this weighting vary with the information environment?

The current repository implements the data pipeline and a baseline analysis using Loughran-McDonald dictionary features and readability measures as word-bag proxies for the narrative channel. These measures are useful as a benchmark, but they are not the intended final narrative measurement strategy. The planned next step is to replace them with LLM-extracted semantic measures from earnings call transcripts and re-estimate the relative pricing weights on hard earnings surprise and managerial narrative.

---

## 2. Data and Sample

The sample covers quarterly earnings calls from 2010 to 2023, matched across five data sources:

- **CIQ earnings call transcripts:** prepared remarks and Q&A text.
- **IBES:** quarterly analyst consensus forecasts and actual EPS.
- **CRSP:** daily stock returns and market returns.
- **Compustat:** firm controls such as size, book-to-market, leverage, ROA, loss indicator, and earnings volatility.
- **13F institutional holdings:** institutional ownership percentage.

The hard-news variable is standardized unexpected earnings:

```text
SUE = (actual EPS - median analyst forecast) / forecast standard deviation
```

The outcome is announcement-window CAR over [-1, +1] around the earnings announcement / call date.

The current estimation sample contains approximately **65,000 firm-quarter observations** across more than **2,700 firms**.

---

## 3. Baseline: Hard-News ERC and Word-Bag Narrative

The first step estimates the standard ERC relationship between earnings surprise and announcement returns, then adds word-bag narrative measures from the earnings call.

Current text features are computed separately for prepared remarks and Q&A:

- LM tone
- negative word ratio
- uncertainty word ratio
- litigious word ratio
- Gunning Fog index
- word counts

### 3.1 OLS Baseline

The OLS specifications use CAR[-1,+1] as the dependent variable and winsorized SUE as the hard-news variable.

| Specification | SUE coef | t-stat | R2 | N |
|---|---:|---:|---:|---:|
| Raw ERC | 0.0054 | 42.91 | 6.6% | 70,033 |
| + Firm controls | 0.0053 | 39.70 | 7.1% | 66,683 |
| + Text features | 0.0050 | 38.06 | 8.2% | 65,961 |
| + Firm FE + Time FE | 0.0053 | 36.94 | 15.3% | 65,961 |

A one-standard-deviation increase in SUE is associated with roughly a 0.50-0.54 percentage point increase in announcement-window CAR. Adding dictionary-based text features reduces the hard-news coefficient only slightly.

Prepared-remarks LM tone is also positively associated with announcement returns in the specification with text controls. This suggests that the current word-bag narrative proxy contains independent return-relevant information, although it may be too coarse to capture the full narrative channel.

---

## 4. DML / Cross-Fitting Robustness

Double Machine Learning is used as a robustness and adjustment tool, not as the main contribution. The current DML specification estimates:

```text
CAR_i = theta * SUE_i + g(Z_i) + error_i
SUE_i = m(Z_i) + residual_i
```

where Z includes word-bag narrative measures, firm controls, and time effects. The nuisance functions are estimated using cross-fitted Lasso, Ridge, and Random Forest models.

| Method | Nuisance | theta | SE | t-stat |
|---|---|---:|---:|---:|
| OLS benchmark | Linear | 0.0050 | 0.000131 | 38.05 |
| DML | Lasso | 0.0050 | 0.000094 | 53.21 |
| DML | Ridge | 0.0050 | 0.000094 | 53.29 |
| DML | Random Forest | 0.0050 | 0.000095 | 52.30 |

The word-bag-adjusted ERC remains close to 0.005 across these specifications. I interpret this conservatively: with the current word-bag proxy, hard earnings news remains strongly priced after adjusting for observed narrative, firm characteristics, and time effects. This should not be read as a fully causal estimate, because SUE is not randomly assigned and the current text measures are only proxies for narrative.

The key next question is whether this conclusion survives when narrative is measured with richer semantic signals from an LLM.

---

## 5. Current Heterogeneity Results

The current heterogeneity analysis uses DML residuals from the baseline word-bag specification and studies whether hard-news ERC varies with the information environment.

The moderators are:

- **Earnings volatility (`eps_vol`):** cognitive load / earnings complexity.
- **Institutional ownership (`io_pct`):** investor sophistication.
- **Analyst coverage (`numest`):** information intermediation.

### 5.1 Best Linear Predictor

| Term | Coefficient | SE | t-stat | Interpretation |
|---|---:|---:|---:|---|
| Baseline ERC | +0.00515 | 0.00013 | 38.91 | hard-news benchmark |
| SUE residual x earnings volatility | -0.00046 | 0.00011 | -4.06 | higher complexity, lower ERC |
| SUE residual x institutional ownership | +0.00010 | 0.00013 | +0.78 | not significant |
| SUE residual x analyst coverage | +0.00034 | 0.00013 | +2.60 | more intermediation, higher ERC |

### 5.2 Tercile Patterns

| Moderator | Low Tercile | Mid Tercile | High Tercile |
|---|---:|---:|---:|
| earnings volatility | 0.00685 | 0.00521 | 0.00395 |
| institutional ownership | 0.00472 | 0.00534 | 0.00496 |
| analyst coverage | 0.00448 | 0.00560 | 0.00540 |

The clearest pattern is that ERC is lower for firms with high earnings volatility. Analyst coverage is associated with a higher ERC, while institutional ownership is not monotonic and not significant in the linear interaction specification.

These findings motivate a richer relative-weighting analysis. If LLM semantic narrative measures are available, the next version can ask whether high-volatility or low-analyst-coverage firms place relatively more weight on narrative and less weight on hard earnings surprise.

---

## 6. Planned LLM Relative-Weighting Design

The planned LLM step will extract structured semantic narrative variables from earnings call transcripts, such as:

- contextual sentiment
- uncertainty / hedging
- forward-looking tone
- defensiveness
- Q&A informativeness or evasiveness
- management optimism conditional on reported earnings

The key comparison is:

```text
CAR_i = theta * SUE_i + controls + FE + error_i
CAR_i = theta_LM * SUE_i + beta_LM * Narrative_LM_i + controls + FE + error_i
CAR_i = theta_LLM * SUE_i + beta_LLM * Narrative_LLM_i + controls + FE + error_i
```

This design asks:

- Does LLM narrative explain announcement-window returns beyond SUE and word-bag text measures?
- Does including LLM narrative reduce the estimated hard-news ERC more than including dictionary measures?
- Do the relative weights on SUE and narrative vary with earnings volatility, analyst coverage, or institutional ownership?

If LLM-based narrative measures explain more return variation or materially change the hard-news coefficient, that would suggest traditional dictionary measures understate the role of managerial narrative. If the hard-news coefficient remains stable even with LLM semantic narrative controls, that would suggest hard earnings news has an independent pricing role beyond managerial narrative.

---

## 7. Limitations

The current analysis has several limitations:

- The current text measures are dictionary-based proxies and may miss context, hedging, defensiveness, and semantic nuance.
- Daily returns identify the pricing of the earnings announcement package, not the isolated causal effect of the earnings call itself.
- Intraday returns would allow a stronger design separating the press-release-to-call window from the call-window reaction.
- DML helps adjust for high-dimensional observed controls, but it does not make SUE randomly assigned.

---

## References

- Ball, R. & Brown, P. (1968). An empirical evaluation of accounting income numbers. *Journal of Accounting Research*, 6(2), 159-178.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1-C68.
- Loughran, T. & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35-65.
- Nie, X. & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299-319.
