# Jump-Diffusion Risk & Allocation Engine

[![tests](https://github.com/marcoreyess22/jump-risk-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/marcoreyess22/jump-risk-engine/actions/workflows/tests.yml)

## How much capital does the wrong risk model cost you?

**Less. That's the problem.**

An audit of **12 VaR/ES specifications × 4 portfolios** across 8 asset classes and 3,923
out-of-sample days (2011–2026), validated with Kupiec, Christoffersen, Acerbi–Székely, and the
Basel capital traffic light.

*[Versión en español](README.es.md) · [Technical guide](docs/guia.html)*

---

## Results

| | |
|---|---|
| **Gaussian VaR produces 2.06× the expected exceptions** | Fails Kupiec, conditional coverage, and the ES backtest in all 4 portfolios |
| **Jump-diffusion VaR produces 0.97×** | Best frequency calibration of the ten |
| **39 of 40 combinations fail the independence test** | After an exception, the probability of another multiplies by 16 |
| **No model gets frequency, timing, and magnitude right at once** | The best on frequency is nearly the worst on clustering, and vice versa |
| **The model that fails everything costs 18.4% LESS capital** | The traffic-light penalty doesn't offset what underestimating saves |
| **CVaR optimization did not reduce realized tail loss** | Allocates 10 pp differently from mean-variance, for +0.1% CVaR |

**Recommendation.** Adopt **`mc_merton`** as the risk specification. Of the ten it is the only
one that gets both the *frequency* of extreme losses right (0.96×, the best calibrated) and
their *magnitude* (passes the Expected Shortfall backtest in all 4 portfolios) — and magnitude
is what the current framework measures, since FRTB's Internal Models Approach is built on
Expected Shortfall rather than VaR.

Its weakness is real and worth stating: **it does not model volatility clustering**, so its
exceptions bunch during crises. That extension has since been built — see
[Act 3](#act-3--closing-the-gap-the-project-left-open). `mc_merton_ewma` halves the clustering
and keeps the perfect ES record at a lower capital multiplier, but turns systematically
conservative on frequency (0.72× against 0.94×). The two tie at 8/16 and neither dominates:
**no model in the set solves all three axes.** Pick `mc_merton` if frequency accuracy matters
most, `mc_merton_ewma` if clustering does.

And do not treat the Basel traffic light as a sufficient approval criterion: on this data it
**economically rewards the models it fails**. The practical conclusion is not to tune the model
until it passes, but that the quantitative test alone is not enough.

---

## Method

**Data.** 8 ETFs (`SPY QQQ IWM EFA EEM TLT GLD DBC`), adjusted closes, 2007-01-03 to
2026-07-30. 4,924 days with no gaps. Downloaded once and cached to `data/prices.csv`.

**Model.** Merton jump-diffusion over daily log returns:

```
X = m + σ·Z + Σ_{i=1}^{N} Y_i        N ~ Poisson(λ),  Y_i ~ N(μ_J, σ_J²)
```

Calibrated by method of moments: the closed-form cumulants are matched to the sample variance,
skewness, and excess kurtosis. The calibration reproduces the empirical moments to `rtol=1e-6`.

**Systemic jump.** A single Poisson process shared by all eight assets, with jump sizes
correlated through the empirical matrix. Because λ is identical across assets, sharing the count
leaves every marginal unchanged — tail dependence comes for free, with no recalibration.

**Walk-forward.** 1,000-day rolling window. Weights are re-optimized on month changes
(188 rebalances); VaR is recomputed every day against the weights in force. Re-optimizing daily
would invalidate the measurement: the VaR would be measuring a portfolio that never existed for
a full day.

**Ten specifications.** Five unconditional with flat variance (`historico`, `normal`, `mc_gbm`,
`mc_merton`, `mc_merton_idio`), three with non-Gaussian tails but flat variance (`t_student`,
`cornish_fisher`, `evt`), and two with conditional volatility (`ewma`, `fhs`). All register
under a single signature `(returns, weights, level, rng) → (VaR, ES)`, so the backtest loop
knows no model by name.

**Validation.** Kupiec (frequency), Christoffersen (independence and conditional coverage),
Acerbi–Székely Test 2 (Expected Shortfall), and the Basel capital traffic light. The coverage
tests were verified against synthetic series with known answers **before** being applied to real
data: Kupiec rejects 9/100 under a true H₀ and 100/100 with an inflated rate; Christoffersen
rejects 100/100 clustered series whose unconditional rate is correct, where Kupiec rejects only
24/100.

Acerbi–Székely got the same treatment and **did not pass on the first attempt**: its null
distribution resampled the *observed* tail ratios, which contaminated it with the alternative —
with ES underestimated by 50% it rejected 4 out of 100. The null was rebuilt from the model
(excess over VaR as exponential with mean ES − VaR, the maximum-entropy choice consistent with
that pair) and power now rises with severity: 1/60 under H₀, 12/60 on t(6) data, 36/60 on t(3).
**The fix changed verdicts**, and the tables below are the post-fix ones.

---

## Act 1 — the diagnosis

*(Five unconditional specifications. Act 2 extends to ten and qualifies the conclusion.)*

### 1. Jumps fix the exception frequency

![exceptions](figures/1_excepciones_acumuladas.png)

| Model | Exceptions | Observed/expected | Kupiec |
|---|---|---|---|
| `normal` | 81 | **2.06×** | FAIL |
| `mc_gbm` | 84 | **2.14×** | FAIL |
| `historico` | 52 | 1.33× | pass |
| `mc_merton` | 38 | **0.97×** | pass |

*(minimum-variance portfolio; the pattern repeats across all four — see
[figure 3](figures/3_razon_excepciones.png))*

`normal` and `mc_gbm` land within simulation noise of each other (81 vs 84), as they must: the
same assumption reached two ways, one closed-form and one by Monte Carlo. Their agreement is a
cross-check nobody designed.

### 2. Idiosyncratic tails diversify away; systemic ones do not

With **independent** jumps across assets, the portfolio's ES/VaR ratio is 1.159 —
indistinguishable from Gaussian (1.162). Independent shocks average out across eight assets and
the tail goes back to normal. Only the **systemic** jump survives aggregation (1.460).

This shows up in the backtest: `mc_merton_idio` passes Kupiec on concentrated portfolios and
fails it on `risk_parity` and `igual_peso`, the two most diversified.

**Design consequence:** modeling fat tails asset by asset is useless for portfolio risk. What
matters is whether they jump together.

### 3. Getting the frequency right is not getting the timing right

![var vs realized](figures/2_var_vs_realizado.png)

**All five specifications in this act fail the independence test**, including the one that
passes Kupiec at 0.97×. The diagnostics:

```
π_01 = 0.0082      π_11 = 0.1351      →  16.4× after an exception

Exceptions per year:   2012: 0    2020: 10-16
                       2014: 0    2018:  5-12
                       2017: 0    2022:  4-5
```

The five worst losses of the period are four days in March 2020 and August 8, 2011.

This is not an implementation defect: **none of the five has time-varying volatility** — all use
a flat 1,000-day window, so they structurally cannot capture clustering. The verdict is correct
and marks the limit of the unconditional approach.

That limit is the question Act 2 opens: **does conditioning on volatility fix it?**

---

## Act 2 — auditing the ten unconditional-and-filtered specifications

Averaged over the 4 portfolios. `passes` counts tests cleared out of 16 (4 portfolios ×
Kupiec, independence, conditional coverage, Acerbi–Székely).

| model | ratio | persistence | ES (of 4) | multiplier | capital proxy (k$/10M) | passes (of 16) |
|---|---|---|---|---|---|---|
| `mc_merton` | **0.94** | 19.4 | **4** | 3.10 | 550 | **8** |
| `mc_merton_ewma` | 0.72 | **10.5** | **4** | **3.00** | 544 | **8** |
| `t_student` | 1.06 | 11.8 | 3 | 3.10 | 548 | 7 |
| `cornish_fisher` | 0.69 | 29.3 | **4** | 3.07 | 693 | 6 |
| `evt` | 1.08 | 13.2 | 2 | 3.12 | 531 | 6 |
| `historico` | 1.23 | 12.1 | 2 | 3.13 | 519 | 5 |
| `fhs` | 1.22 | 7.5 | 2 | 3.10 | 504 | 5 |
| `fhs_merton` | 1.22 | 7.4 | 1 | 3.09 | 498 | 5 |
| `mc_merton_idio` | 1.37 | 8.9 | 0 | 3.17 | 487 | 2 |
| `ewma` | 2.09 | **3.4** | 0 | 3.35 | 449 | 1 |
| `normal` | 1.78 | 7.0 | 0 | 3.23 | 465 | **0** |
| `mc_gbm` | 1.82 | 6.3 | 0 | 3.24 | 465 | **0** |

### Three axes, and no model gets all three

![frontier](figures/5_frontera.png)

| | Frequency | Timing | Magnitude (ES) | Mechanism |
|---|---|---|---|---|
| `ewma` | **worst** (2.04×) | **best** (π 3.6) | 0/4 | conditional vol, Gaussian tails |
| `fhs` | good (1.22×) | 2nd best (π 7.5) | 2/4 | conditional scale, empirical shape |
| `mc_merton` | **best** (0.96×) | nearly worst (π 20.4) | **4/4** | fat tails, flat vol |

Of 40 combinations, **exactly one passes independence**: `igual_peso/ewma`, at p = 0.146. EWMA
cuts persistence to 3.6 but fails the frequency because it still assumes normality: it knows
*when* risk rises, not *how much* tail there is.

FHS looked like the synthesis — EWMA's conditional scale plus the empirical tail shape — and it
does get frequency and timing right. But it **fails the magnitude**: it flunks the ES backtest
in 2 of 4 portfolios (p = 0.006 and 0.021). The mechanism follows from its construction: it
rescales historical residuals by current volatility, so a jump landing during a calm stretch is
measured with residuals standardized in calm and the severity comes out short.

Merton is the exact inverse: it gets *how much* right and *when* wrong. Since FRTB's IMA is
built on Expected Shortfall, **magnitude carries more weight than timing** for the metric the
framework actually uses — which is why it is the recommendation despite its worse persistence.

### The inverted incentive

![incentive](figures/6_incentivo.png)

Capital **proxy** = mean multiplier × mean VaR × $10M notional, minimum-variance portfolio.
This is a comparison yardstick under the *historical* Basel VaR traffic light, **not** a
regulatory capital figure — see [Scope](#scope-and-limitations):

```
normal         $389,877   +0.0%   ← fails all four tests
mc_gbm         $390,960   +0.3%   ← fails Kupiec, CC, and ES
ewma           $393,288   +0.9%   ← fails Kupiec and ES
fhs            $427,270   +9.6%
mc_merton      $461,661  +18.4%   ← the best calibrated
mc_merton_ewma $482,938  +23.9%
cornish_fisher $530,557  +36.1%
```

**The model that fails everything is the cheapest.** The mechanism: the multiplier penalizes at
most +33% (3.00 → 4.00), while underestimating VaR saves ~20% directly. And the effective
penalty falls far below that cap: at twice the exception rate, a model averages 5.2 exceptions
per 250-day window — barely the first amber band. **The traffic light has little power over a
250-day window.**

This is not a computational artifact: it is the documented critique of the Basel backtesting
framework, and one reason FRTB moved the IMA capital metric from VaR to Expected Shortfall while
keeping P&L attribution and qualitative approval requirements alongside the quantitative test.

*This project's original hypothesis was that failing the backtest would raise capital costs. It
turned out to be wrong in sign. Reported as it came out.*

---

## Act 3 — closing the gap the project left open

Act 2 ended on a specific unbuilt extension: *couple Merton marginals to an EWMA-style
conditional scale*. Two ways of doing it are now in the registry.

**Does EWMA leave anything for Merton to model?** Standardising each asset by its own EWMA
volatility absorbs **63% to 90%** of the excess kurtosis — most of the fat tails in daily
returns come from volatility clustering, not from jumps. But 2.06 of excess kurtosis survives
on average, and that residual is what the jump component fits. Had it been zero, the model
would have collapsed to plain `ewma`.

**`mc_merton_ewma`** standardises per asset, calibrates Merton on the residuals, generates
joint scenarios with the systemic jump, and rescales by tomorrow's conditional volatility. It
is a constant-conditional-correlation model with Merton innovations.

**`fhs_merton`** does the same on the portfolio series directly, avoiding the CCC assumption —
at the cost of losing the systemic jump, since there are no assets left to couple.

| | frequency | timing | magnitude | total |
|---|---|---|---|---|
| `mc_merton` | **0.94** (Kupiec 4/4) | 19.4 | **ES 4/4** | 8/16 |
| `mc_merton_ewma` | 0.72 (Kupiec 2/4) | **10.5** | **ES 4/4** | 8/16 |
| `fhs_merton` | 1.22 | 7.4 | ES 1/4 | 5/16 |

**The gap narrowed; it did not close.** `mc_merton_ewma` is the only specification that combines
a perfect ES record with any independence pass, it halves the clustering (19.4 → 10.5) and it
carries the lowest capital multiplier of the twelve (3.00 — every window green). But it bought
that by becoming systematically conservative: 0.72× where Merton sits at 0.94×. It traded one
bias for another, and 10.5 is still an order of magnitude from the 1.0 that independence wants.

**`fhs_merton` is a clean negative result.** Replacing FHS's empirical bootstrap with a
parametric tail made it *worse* — ES falls from 2/4 to 1/4. The mechanism is visible in the
tests: the parametric tail reaches −10σ where the worst observed residual is −5.14σ. That
ability to generate the unseen helps in the multivariate construction and overshoots in the
univariate one. The empirical residual distribution was doing better work than the parametric
fit. It stays in the registry, documented, rather than being quietly dropped.

---

## The negative result

**CVaR optimization did not reduce realized tail risk.**

| | ann. return | ann. vol | Sharpe | VaR99 | CVaR99 | worst day | max DD |
|---|---|---|---|---|---|---|---|
| `min_cvar` | 7.07% | 9.03% | 0.783 | 1.52% | 2.08% | **4.55%** | 23.75% |
| `min_var` | 6.97% | 8.72% | **0.799** | 1.52% | 2.08% | 5.02% | **22.67%** |
| `risk_parity` | 6.79% | 9.47% | 0.717 | 1.63% | 2.36% | 5.55% | 23.89% |
| `igual_peso` | 7.67% | 12.61% | 0.608 | 2.20% | 3.24% | 7.99% | 25.04% |

Realized CVaR: **+0.1%**. Maximum drawdown: **4.8% worse**. Sharpe: slightly lower. The only
metric it wins is the single worst day (−9.3%).

**It is not that the optimizers coincide.** The portfolios are genuinely different: 10
percentage points apart in weights, against a 1.8 pp dispersion across simulation seeds — real
signal, 5.5× above the noise. The CVaR optimizer does something different; it just doesn't pay.

The most plausible reading is estimation error. CVaR at 99% fits the extreme 1% of a
distribution that was itself estimated from 1,000 days. Dropping to β = 0.95 — where the tail has
five times the data — cuts the weight difference from 10 pp to 3 pp. The discrepancy lives
exactly where the data is thinnest.

---

## Declared limitations

- **λ is not identified** by moments 2–4: five parameters against four equations. Fixed at
  0.05/day (12.6 jumps per year) as a declared modeling choice, with sensitivity reported in
  `merton.sensibilidad_lambda`. Closing it would require the sixth sample cumulant, which is
  noise.
- **No transaction costs.** With monthly rebalancing and stable portfolios the effect would be
  modest, but it is not measured.
- **GARCH is not included.** Refitting it inside the walk-forward means ~3,900 daily estimations
  per asset with their convergence failures. EWMA is its non-estimated cousin and captures most
  of the clustering; FHS uses it as a filter. This is the natural extension.
- **No asymmetric tail dependence.** The systemic jump couples all eight assets with the same
  intensity. Per-asset jump loadings would give a richer structure — and are the likeliest
  candidate for making CVaR optimization actually pay.
- **Long-only**, `sum(w)=1, w≥0`.
- **A single confidence level** (99%) in the main backtest.
- **Moment-based calibration, not maximum likelihood.** The Merton likelihood is known to be
  unbounded; the robust estimator was preferred.
- **ES p-values are simulated.** At `n_boot=20,000` the spread across seeds is 0.003 — enough
  for the observed values, but a future borderline verdict would need more replications.

---

## Scope and limitations

**What this is.** A comparative study of VaR/ES specifications and portfolio allocation rules,
validated out of sample. Its purpose is methodological: to measure how model choice changes
measured risk, and to test every statistic before trusting it.

**What this is not.** It does not estimate regulatory capital for any institution. The Basel
traffic light reproduced in `src/basel.py` is the **historical** VaR-based backtesting table
(Basel II / Basel 2.5), used here purely as a common yardstick, and every monetary figure is
named `capital_proxy` for that reason. The current framework differs in ways this project does
not implement:

- FRTB's Internal Models Approach computes capital from **Expected Shortfall at 97.5%**, with
  liquidity-horizon scaling and stressed-period calibration — not from 1-day VaR.
- 1-day VaR backtesting survives under FRTB, but at *desk* level and paired with **P&L
  attribution** tests comparing risk-theoretical against hypothetical P&L. Not implemented here.
- Non-modellable risk factors, the standardised-approach floor, and supervisory add-ons are all
  out of scope.

Primary sources: [MAR32 — backtesting and P&L attribution](https://www.bis.org/basel_framework/chapter/MAR/32.htm)
· [BCBS d457 — Minimum capital requirements for market risk](https://www.bis.org/bcbs/publ/d457.htm)

**Statistical limitations that survive.** Bootstrap intervals on the exception ratio overlap
heavily across the surviving models (`mc_merton` [0.59, 1.43] vs `fhs` [0.87, 1.76] on
`min_var`): the data separates the Gaussian family from the rest, but **does not rank the
survivors**. The point ordering in the tables above should be read as a summary, not as a
significant result. Run `make scorecard` for the intervals.

**λ does real work in the recommendation.** The bounded sensitivity grid (`make sensitivity`,
3 λ × 3 windows on `min_var`) shows the gaussian-versus-jump ordering is robust — `normal` is
never better than 1.85, `mc_merton` never worse than 1.82. But `mc_merton`'s *advantage* is not:
at λ = 0.02 its ratio is 1.51–1.82, **behind both `fhs` and `historico`**. The declared choice of
λ = 0.05 happens to be the best-calibrated one, and λ is precisely the parameter moments 2–4 do
not identify. Read the recommendation as conditional on that modelling choice, not as a
data-driven ranking.

**The idiosyncratic counterfactual is not clean.** With independent jumps, D = Σ − J is not
positive semi-definite and must be projected, which distorts the target covariance by ~19%
(~28% in correlation). `mc_merton_idio` is a diagnostic of tail diversification, not a causal
attribution. Run `make report` for the diagnostic.

## References

- Merton, R. C. (1976). *Option pricing when underlying stock returns are discontinuous.*
  Journal of Financial Economics 3(1–2), 125–144.
- Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk.*
  Journal of Risk 2(3), 21–41.
- Kupiec, P. (1995). *Techniques for verifying the accuracy of risk measurement models.*
  Journal of Derivatives 3(2), 73–84.
- Christoffersen, P. (1998). *Evaluating interval forecasts.* International Economic Review
  39(4), 841–862.
- Acerbi, C., & Székely, B. (2014). *Backtesting Expected Shortfall.* Risk Magazine.
- Maillard, S., Roncalli, T., & Teïletche, J. (2010). *The properties of equally weighted risk
  contribution portfolios.* Journal of Portfolio Management 36(4), 60–70.
- Basel Committee on Banking Supervision. *MAR32 — Internal models approach: backtesting and
  P&L attribution.* https://www.bis.org/basel_framework/chapter/MAR/32.htm
- Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market
  risk* (d457). https://www.bis.org/bcbs/publ/d457.htm

---

## Reproducing

```bash
pip install -e ".[data]"       # omit [data] to run fully offline from the cached CSV
python tests/test_core.py      # 25 validation checks, ~4 min
python -m src.backtest         # walk-forward, 12 models × 4 portfolios, ~12 min
python -m src.basel            # capital traffic light and ES backtest
python -m src.plots            # figures
python -m src.diario --sembrar # daily run with persistent state (shadow mode)
```

`basel` and `plots` read `data/walkforward.csv`, which `backtest` produces — that file is not
versioned (15 MB of derived output), so run the backtest first.

`tests/test_core.py` holds the validation criterion for every phase. The four that matter most:

1. **The calibration reproduces the closed-form cumulants** (`rtol=1e-6`), with no simulation
   noise — isolating the calibration from the simulator.
2. **The LP's `α` converges to the empirical VaR**: 0.015208 vs 0.015208. The LP never sees a
   percentile; a sign or scale error would surface here.
3. **Under centered Gaussian scenarios, min-CVaR converges to minimum variance** (max|Δw| =
   0.0022) — the theoretical equivalence that validates the Rockafellar–Uryasev formulation.
4. **The coverage tests are validated against synthetic series** before being used.

## Layout

```
src/data.py       ingestion and cache
src/merton.py     calibration, cumulants, simulation, joint scenarios
src/optimize.py   min-CVaR (LP), Markowitz (QP), risk parity
src/risk.py       registry of the 12 VaR/ES models under one signature
src/backtest.py   walk-forward, Kupiec, Christoffersen
src/basel.py      capital traffic light and Acerbi–Székely
src/diario.py     daily run with persistent state
src/plots.py      figures
```

## Discarded hypotheses

Three of my own conjectures that the data refuted, documented because testing and discarding is
part of the work:

| Hypothesis | What the data said |
|---|---|
| The systemic jump makes the joint distribution elliptical, so min-CVaR coincides with minimum variance | False: the pure-mixture case gave the **largest** weight difference, not the smallest |
| Min-CVaR reduces realized tail loss versus mean-variance | False: +0.1% CVaR and 4.8% worse drawdown, despite allocating 10 pp differently |
| Failing the backtest raises the capital proxy | **Inverted**: the model that fails all four tests costs 18.4% less |
