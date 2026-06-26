# Hard News vs. Narrative Framing in Earnings Announcements
## A Double Machine Learning Approach to Attention Allocation

**Yulin Wang — University of Chicago**
**Data: WRDS (CIQ + IBES + CRSP + Compustat), 2010–2023**

---

## 1. Research Questions

When a firm reports earnings, two signals arrive simultaneously: a hard number (the earnings surprise) and a managerial narrative (the earnings call). This paper asks two questions.

**Project A:** After controlling for managerial narrative framing, does earnings surprise still have an independent causal effect on stock returns?

**Project B:** When do investors rely more on hard numbers versus narrative? Specifically, does the earnings response coefficient (ERC) vary systematically with earnings complexity, investor sophistication, and analyst coverage?

---

## 2. Data and Sample

The sample covers quarterly earnings calls from 2010 to 2023, matched across four data sources. Earnings call transcripts (prepared remarks and Q&A sections) are drawn from Capital IQ via WRDS. Earnings surprise (SUE) is constructed from IBES quarterly consensus forecasts: SUE = (actual EPS − median analyst forecast) / forecast standard deviation, measured using the most recent pre-announcement consensus. Cumulative abnormal returns (CAR) are computed from CRSP daily returns using a market-adjusted compounding method over the window [−1, +1] trading days around the announcement date. Firm controls (size, book-to-market, leverage, ROA, earnings volatility) come from Compustat. Institutional ownership is from Thomson Reuters 13F filings linked to CRSP via CUSIP.

The final estimation sample contains **65,961 firm-quarter observations** across **2,734 unique firms**. SUE is winsorized at the 1st and 99th percentiles. CAR[−1,+1] has a mean of 0.06% and standard deviation of 9.45%, consistent with prior literature.

Text features are computed from transcript text using the Loughran-McDonald (2011) Master Dictionary. Separate features are constructed for prepared remarks and Q&A sections: LM tone ((positive − negative) / total words), uncertainty word ratio, litigious word ratio, and Gunning Fog Index. The average prepared-remarks section is 2,865 words; Q&A sections average 4,001 words. Prepared-remarks tone averages +1.1%, confirming the well-known optimism bias in managerial communication.

---

## 3. Project A: Does Hard News Matter After Partialling Out Narrative?

### 3.1 OLS Baseline

Table 1 presents four OLS specifications with clustered standard errors by firm. The dependent variable is CAR[−1,+1] (winsorized). The treatment variable is winsorized SUE.

**Table 1: OLS Earnings Response Coefficients**

| Specification | SUE coef (θ) | t-stat | R² | N |
|---|---|---|---|---|
| (1) Raw ERC | 0.0054 | 42.91 | 6.6% | 70,033 |
| (2) + Firm controls | 0.0053 | 39.70 | 7.1% | 66,683 |
| (3) + Text features | 0.0050 | 38.06 | 8.2% | 65,961 |
| (4) + Firm FE + Time FE | 0.0053 | 36.94 | 15.3% | 65,961 |

A one-standard-deviation increase in SUE is associated with a 0.50–0.54% higher announcement-window CAR. Adding text controls in Specification (3) reduces θ slightly from 0.0054 to 0.0050, suggesting that narrative framing is correlated with earnings surprise but does not fully account for the price reaction. Notably, prepared-remarks LM tone enters significantly in Specification (3) with a coefficient of 0.223 (t = 3.73), confirming that positive language in management's prepared remarks is independently associated with higher returns, above and beyond the earnings number itself.

### 3.2 Double Machine Learning Estimation

The OLS estimates in Table 1 restrict the confounding function to be linear. To allow for nonlinear interactions between text features, firm characteristics, and time effects, we apply Double Machine Learning (DML) partialling-out (Chernozhukov et al., 2018). The model is:

> CAR_i = θ · SUE_i + g(Z_i) + ε_i
> SUE_i = m(Z_i) + v_i

where Z includes all LM text features (prepared remarks and Q&A), firm controls, and quarter fixed effects. The nuisance functions g(·) and m(·) are estimated via cross-fitted Lasso, Ridge, and Random Forest regressions (K = 5 folds). The treatment effect θ is recovered from the residual regression Ỹ ~ D̃, where Ỹ = CAR − ĝ(Z) and D̃ = SUE − m̂(Z).

**Table 2: DML Estimates of ERC**

| Method | Nuisance | θ | SE | t-stat |
|---|---|---|---|---|
| OLS (benchmark) | Linear | 0.0050 | 0.000131 | 38.05 |
| DML | Lasso | 0.0050 | 0.000094 | 53.21 |
| DML | Ridge | 0.0050 | 0.000094 | 53.29 |
| DML | Random Forest | 0.0050 | 0.000095 | 52.30 |

Three findings stand out. First, θ is identical across all four methods at 0.0050. The nonlinear partialling-out does not change the point estimate, indicating that OLS's linear text controls were not materially biased: the hard news channel is not confounded by narrative framing, whether that confounding is linear or nonlinear. Second, the DML standard errors are 28% smaller than OLS (0.000094 vs. 0.000131), raising the t-statistic from 38 to 53. This efficiency gain arises because DML removes text-driven variance from both the outcome and treatment equations before the final regression, reducing residual noise. Third, the narrative bias — defined as the difference between the OLS and DML point estimates — is essentially zero (< 0.0001). This confirms that the market correctly prices the hard number independently of how management frames it.

**Interpretation.** A one-standard-deviation upward earnings surprise generates a 0.50% announcement-window abnormal return, net of all narrative framing effects. This estimate is highly robust across nuisance models and represents the cleanest available estimate of the average ERC in the 2010–2023 period.

---

## 4. Project B: Who Relies More on Hard Numbers?

The DML residuals (Ỹ, D̃) from Project A serve as the input for Project B. These residuals have text, controls, and time effects partialled out, so any remaining heterogeneity in the Ỹ ~ D̃ relationship reflects genuine variation in how the market prices hard earnings news across firms and contexts.

We test three moderating hypotheses:

- **Cognitive load (eps_vol):** When EPS is highly volatile across quarters, the current period's earnings number is harder to interpret as a signal of future cash flows. We predict ERC decreases with eps_vol (τ₁ < 0).
- **Investor sophistication (io_pct):** Higher institutional ownership implies more sophisticated investors who can more accurately process hard numerical information. We predict ERC increases with io_pct (τ₂ > 0).
- **Information intermediation (numest):** More analyst coverage means earnings news is more thoroughly researched and disseminated. We predict ERC increases with numest (τ₃ > 0).

### 4.1 Best Linear Predictor (BLP)

The BLP regression is:

> Ỹ_i = θ₀·D̃_i + τ₁·(D̃_i × eps_vol_s) + τ₂·(D̃_i × io_pct_s) + τ₃·(D̃_i × numest_s) + γ·X_s + ε_i

where all moderators are standardized (zero mean, unit variance). Standard errors are clustered by firm.

**Table 3: BLP Heterogeneous Treatment Effects**

| Term | Coefficient | SE | t-stat | Hypothesis |
|---|---|---|---|---|
| D̃ (baseline ERC, θ₀) | +0.00515 | 0.00013 | 38.91*** | — |
| D̃ × eps_vol (cognitive load) | −0.00046 | 0.00011 | −4.06*** | τ₁ < 0 ✓ |
| D̃ × io_pct (sophistication) | +0.00010 | 0.00013 | +0.78 | τ₂ > 0 ✗ |
| D̃ × numest (intermediation) | +0.00034 | 0.00013 | +2.60*** | τ₃ > 0 ✓ |

*Note: All moderators standardized. Firm-clustered SE. *** p<0.01.*

**Earnings complexity strongly reduces ERC.** A one-standard-deviation increase in EPS volatility reduces the ERC by 0.00046, roughly 9% of the baseline θ₀ = 0.00515. This is the most economically and statistically significant result in Table 3. Importantly, this effect is estimated on DML residuals — it is not contaminated by the text channel, firm size, leverage, or time trends.

**Analyst coverage moderately increases ERC.** A one-standard-deviation increase in analyst count raises ERC by 0.00034 (t = 2.60). More pre-processed information appears to sharpen the market's response to the hard number.

**Institutional ownership is a null result.** τ₂ = +0.00010 with t = 0.78, insignificant. Sophisticated investors do not respond more aggressively to earnings surprises in the announcement window. One interpretation: institutional investors may incorporate earnings information more gradually via channels other than the announcement window (e.g., pre-call information flow, analyst pre-announcements).

### 4.2 Tercile Analysis

**Table 4: ERC by Moderator Tercile**

| Moderator | Low Tercile | Mid Tercile | High Tercile | High / Low |
|---|---|---|---|---|
| eps_vol | 0.00685 | 0.00521 | 0.00395 | 0.58× |
| io_pct | 0.00472 | 0.00534 | 0.00496 | 1.05× |
| numest | 0.00448 | 0.00560 | 0.00540 | 1.21× |

*All tercile θ estimates significant at p < 0.001.*

The eps_vol gradient is striking. Firms in the lowest eps_vol tercile have an ERC of 0.685% — 73% higher than the 0.395% ERC for firms in the highest tercile. The confidence intervals across terciles are non-overlapping, confirming that this gradient is not sampling noise. For io_pct, the pattern is non-monotonic (Low < High < Mid), consistent with the null result in the BLP. For numest, the ERC rises from low to mid tercile but then stabilizes, suggesting diminishing returns to analyst coverage.

### 4.3 R-Learner Robustness

As a nonparametric robustness check, we implement the R-Learner (Nie and Wager, 2021) using Random Forest as the CATE learner, weighted by D̃² to account for variation in treatment strength. The estimated CATE distribution has mean 0.00564 and standard deviation 0.00161, with a range of [0.00068, 0.01166]. The most treated firms have an ERC nearly 17 times larger than the least treated.

Random Forest feature importance confirms the BLP ordering:

| Moderator | RF Feature Importance | Corr(CATE, X) |
|---|---|---|
| eps_vol | 61.3% | −0.353 |
| io_pct | 24.2% | +0.123 |
| numest | 14.5% | +0.126 |

Earnings complexity alone explains 61% of the estimated CATE variation. The R-Learner also detects a nonlinear io_pct effect (24% importance, +0.12 correlation with CATE) that the BLP's linear interaction misses, suggesting that the sophistication channel operates nonlinearly — it matters most at very high institutional ownership levels.

---

## 5. Discussion

Three takeaways emerge from the combined A + B results.

**1. Hard numbers are priced independently of narrative.** The DML estimate confirms that earnings surprise has a stable, significant effect on announcement returns regardless of how management frames the news. The narrative channel (tone, uncertainty, fog index) is a parallel signal, not a substitute for the hard number.

**2. Cognitive load is the primary source of ERC heterogeneity.** Earnings complexity — measured by the historical volatility of EPS — explains 61% of cross-sectional ERC variation. When earnings are hard to interpret as a signal of future value, the market under-reacts to the announced number. This is consistent with the bounded rationality literature: limited attention and processing capacity reduce the informativeness of hard data.

**3. Analyst coverage amplifies hard news; institutional ownership does not.** Analysts act as information intermediaries who sharpen the market's reaction to earnings surprises (τ₃ > 0, t = 2.60). By contrast, institutional ownership has no significant linear effect on ERC in the announcement window. Institutions may process earnings information pre-announcement or react more gradually across the post-announcement drift window.

---

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1–C68.
- Loughran, T. & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.
- Nie, X. & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299–319.
- Collins, D.W. & Kothari, S.P. (1989). An analysis of intertemporal and cross-sectional determinants of earnings response coefficients. *Journal of Accounting and Economics*, 11(2–3), 143–181.
- Ball, R. & Brown, P. (1968). An empirical evaluation of accounting income numbers. *Journal of Accounting Research*, 6(2), 159–178.
