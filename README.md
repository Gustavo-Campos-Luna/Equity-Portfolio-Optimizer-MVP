# Equity Portfolio Optimizer

A quantitative portfolio construction system implementing Modern Portfolio Theory
with a comprehensive rolling backtest framework, advanced risk analytics, and
multi-strategy comparison. Designed for equity portfolio management, factor
investing research, and wealth management analysis.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Quick Start](#4-quick-start)
5. [Methodology](#5-methodology)
6. [Optimization Strategies](#6-optimization-strategies)
7. [Risk Analytics](#7-risk-analytics)
8. [Backtesting Framework](#8-backtesting-framework)
9. [Results and Analysis](#9-results-and-analysis)
10. [Key Findings and Conclusions](#10-key-findings-and-conclusions)
11. [Limitations and Assumptions](#11-limitations-and-assumptions)
12. [Configuration Reference](#12-configuration-reference)
13. [Formulas Reference](#13-formulas-reference)

---

## 1. Overview

This system constructs and evaluates equity portfolios using three classical
mean-variance optimization algorithms, a composite multi-factor asset screener,
and a walk-forward rolling backtest engine that eliminates lookahead bias. The
pipeline produces a complete suite of institutional-grade performance metrics,
automated analytical conclusions, and wealth projection tables.

**Key capabilities:**

- Three optimization strategies: Max Sharpe, Minimum Variance, Risk Parity
- Five asset screening methodologies (Sharpe, quality factor, momentum, low-risk, composite)
- Comprehensive risk metrics: Sortino, Calmar, CVaR, Omega, Ulcer Index, Jensen Alpha
- Walk-forward rolling backtest with transaction costs and turnover tracking
- Wealth projection at multiple horizons (nominal and real)
- Risk attribution (marginal risk contribution per asset)
- Fractional Kelly position sizing
- Full visualization suite (14 charts)

**Data source:** Yahoo Finance (adjusted closing prices, 5-year window, daily frequency).

**Benchmark:** S&P 500 (`^GSPC`).

---

## 2. Project Structure

```
equity-portfolio-optimizer/
├── src/
│   ├── config/
│   │   └── settings.py          # PortfolioConfig dataclass
│   ├── data/
│   │   └── market_data.py       # DataFetcher class
│   ├── metrics/
│   │   └── financial_metrics.py # MetricsCalculator class
│   ├── optimization/
│   │   ├── asset_screener.py    # AssetScreener class
│   │   └── portfolio_optimizer.py # PortfolioOptimizer class
│   ├── backtesting/
│   │   └── backtest_engine.py   # BacktestEngine class
│   ├── reporting/
│   │   └── performance_report.py # PerformanceReport class
│   └── visualization/
│       └── charts.py            # ChartEngine class
├── main.py                      # Orchestration entry point
├── requirements.txt
└── README.md
```

Each module has a single, well-defined responsibility. The data flow is strictly
unidirectional: configuration → data → metrics → optimization → backtest →
reporting/visualization.

---

## 3. Installation

```bash
git clone <repository-url>
cd equity-portfolio-optimizer
pip install -r requirements.txt
```

Requires Python 3.9 or higher. A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## 4. Quick Start

```bash
python main.py
```

The pipeline will:

1. Download 5 years of daily price data for 25 blue-chip equities.
2. Apply data quality filters and compute per-asset metrics.
3. Screen and select the top 15 assets using an enhanced composite score.
4. Optimize three portfolios (Max Sharpe, Min Variance, Risk Parity).
5. Run a rolling out-of-sample backtest with monthly rebalancing.
6. Print formatted performance reports and analytical conclusions.
7. Generate 14 charts saved to `output/charts/`.

**Expected runtime:** approximately 90–150 seconds (data download dependent).

To customize the asset universe or constraints, edit `src/config/settings.py`
or instantiate `PortfolioConfig` directly in `main.py`:

```python
from src.config.settings import PortfolioConfig
config = PortfolioConfig(
    tickers=["AAPL", "MSFT", "NVDA", "V", "MA"],
    top_n=5,
    weight_cap=0.30,
    risk_free_rate=0.045,
    window_years=3,
)
run(config=config)
```

---

## 5. Methodology

### 5.1 Data Pipeline

Daily adjusted closing prices are downloaded from Yahoo Finance using the
`yfinance` library. Adjusted prices account for dividends and stock splits,
ensuring consistency of return calculations across the full historical window.

**Quality filters applied:**

| Filter | Criterion | Rationale |
|--------|-----------|-----------|
| Coverage | >= 80% non-null observations | Prevents sparse data from distorting covariance |
| Session count | >= 80% of window sessions | Ensures adequate statistical power |
| Fallback | Relaxed to 25% if < 3 assets pass | Maintains optimizer feasibility |

### 5.2 Return Computation

Daily log returns are not used; simple arithmetic returns are used throughout
to preserve additivity for portfolio-level calculations:

```
r_t = (P_t / P_{t-1}) - 1
```

### 5.3 Outlier Treatment

Winsorization at the 1st and 99th percentiles is applied to **in-sample**
returns only before any metric calculation or optimization. Out-of-sample
returns are never modified, preserving the validity of backtest results.

### 5.4 Asset Screening

Before optimization, assets are ranked by a composite score. The default
**Enhanced Composite** method weights four normalized factors:

```
Score = 0.60 * Sharpe_normalized
      + 0.15 * (1 - Volatility_normalized)
      + 0.15 * Momentum_12M_normalized
      + 0.10 * (1 - CVaR_95_normalized)
```

This formulation favors assets with strong risk-adjusted returns and
positive momentum while penalizing fat-tailed return distributions.

---

## 6. Optimization Strategies

### 6.1 Maximum Sharpe Ratio

**Objective:** Maximize the excess return per unit of total risk.

```
max_w  (w^T mu - r_f) / sqrt(w^T Sigma w)

Subject to:
  sum(w) = 1
  0 <= w_i <= weight_cap  for all i
  Active positions >= min_positions
```

This portfolio lies on the Capital Market Line — the tangency portfolio
in mean-variance space. It is the theoretical optimal portfolio for an
investor who can combine the risky portfolio with a risk-free asset.

### 6.2 Global Minimum Variance Portfolio (GMVP)

**Objective:** Minimize total portfolio variance regardless of expected return.

```
min_w  w^T Sigma w

Subject to:
  sum(w) = 1
  0 <= w_i <= weight_cap  for all i
```

The GMVP is preferred when return forecasts are unreliable, as it depends
only on the covariance matrix. Empirically, GMVP portfolios have been shown
to produce competitive out-of-sample Sharpe ratios relative to unconstrained
mean-variance portfolios (Clarke et al., 2006).

### 6.3 Risk Parity

**Objective:** Equalize the marginal risk contribution of each asset.

```
min_w  sum_i [ RC_i - sigma_p / N ]^2

where:
  RC_i = w_i * (Sigma w)_i / sigma_p    (risk contribution of asset i)
  sigma_p = sqrt(w^T Sigma w)           (portfolio volatility)
  N = number of assets
```

Risk Parity portfolios are well-diversified in risk space rather than
capital space. The approach was popularized by Bridgewater Associates
and is widely used in multi-asset and all-weather strategies.

**Covariance regularization:** Ledoit-Wolf shrinkage (alpha = 10%) is applied
to the sample covariance matrix to improve conditioning and reduce
estimation error in small samples:

```
Sigma_reg = (1 - alpha) * Sigma_sample + alpha * Sigma_target
```

where `Sigma_target = (trace(Sigma_sample)/N) * I`.

---

## 7. Risk Analytics

### 7.1 Return Metrics

| Metric | Formula |
|--------|---------|
| Total Return | `P_T / P_0 - 1` |
| CAGR | `(P_T / P_0)^(1/T) - 1` |
| Annualized Return | `mean(r) * 252` |

### 7.2 Risk Metrics

| Metric | Formula |
|--------|---------|
| Annualized Volatility | `std(r) * sqrt(252)` |
| Maximum Drawdown | `min((P_t - max(P_{0:t})) / max(P_{0:t}))` |
| VaR (95%) | `P_5(r) * sqrt(252)` |
| CVaR (95%) | `E[r | r <= VaR] * sqrt(252)` |
| Downside Deviation | `sqrt(mean(min(r - r_f/252, 0)^2)) * sqrt(252)` |
| Ulcer Index | `sqrt(mean(D_t^2))` where `D_t = (P_t - max) / max * 100` |

**CVaR (Conditional Value at Risk / Expected Shortfall)** is the expected
loss given that a loss exceeds the VaR threshold. It is a coherent risk
measure in the sense of Artzner et al. (1999) and preferred by regulatory
frameworks (Basel III, Solvency II) over VaR.

**Ulcer Index** captures both the depth and duration of drawdowns. Unlike
maximum drawdown, which only measures the worst single event, the Ulcer
Index penalizes prolonged underwater periods, making it suitable for
evaluating strategies in range-bound or recovering markets.

### 7.3 Risk-Adjusted Metrics

| Metric | Formula |
|--------|---------|
| Sharpe Ratio | `(R_p - r_f) / sigma_p` |
| Sortino Ratio | `(R_p - r_f) / DD_p` |
| Calmar Ratio | `CAGR / |MaxDrawdown|` |
| Omega Ratio | `E[max(r-L,0)] / E[max(L-r,0)]` |
| Information Ratio | `(R_p - R_b) / TE` |

**Sortino Ratio** penalizes only downside volatility, rewarding strategies
that achieve high returns through upside variance rather than symmetric
risk-taking. It is particularly relevant for asymmetric return distributions.

**Calmar Ratio** frames returns in terms of the worst historical loss,
aligning with drawdown-sensitive mandates such as capital-protected products.

**Omega Ratio** captures the complete return distribution without assuming
normality, providing a more robust comparison when excess kurtosis is present.

### 7.4 Benchmark-Relative Metrics

| Metric | Formula |
|--------|---------|
| Jensen Alpha | `R_p - [r_f + beta * (R_b - r_f)]` |
| Beta | `Cov(r_p, r_b) / Var(r_b)` |
| Tracking Error | `std(r_p - r_b) * sqrt(252)` |
| Information Ratio | `(R_p - R_b) / TE` |
| Hit Rate | `P(r_p > r_b)` |

**Jensen Alpha** measures the portfolio's abnormal return relative to its
systematic risk exposure (Beta). A positive alpha indicates that the strategy
generates returns beyond what CAPM would predict for the given market exposure.

---

## 8. Backtesting Framework

### 8.1 Walk-Forward Methodology

The backtest simulates live portfolio management under realistic constraints:

```
For each rebalancing date t in [T_start, T_end]:
  1. Training window: [t - window_years, t]   (in-sample)
  2. Apply quality filter to in-sample data
  3. Compute asset metrics on in-sample returns
  4. Screen top-N assets by composite score
  5. Optimize portfolio weights using selected strategy
  6. Apply weights to returns in (t, t+1]     (out-of-sample)
  7. Deduct transaction costs on rebalancing day
```

### 8.2 Critical Design Decisions

**No lookahead bias:** Outlier clipping and quality filtering are applied
exclusively to the in-sample window. The full-sample statistics are never
used to inform the optimization or screening at any point in time.

**Dynamic weight cap:** The per-asset weight cap is adjusted dynamically to
satisfy the minimum-positions constraint. If the configured cap would make
the sum of minimum weights infeasible, the cap is tightened:

```
effective_cap = min(weight_cap, 1 / max(min_positions, n_assets))
```

**Transaction costs:** A flat cost of 15 basis points per one-way transaction
is applied to the first day of each new period, proportional to the period
turnover. Turnover is computed as:

```
Turnover = 0.5 * sum_i |w_new_i - w_old_i|
```

### 8.3 Backtest Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Training window | 2 years | Balances recency and statistical validity |
| Rebalancing | Monthly | Standard institutional practice |
| Transaction cost | 15 bps | Conservative estimate for liquid large-caps |
| Min warmup sessions | 60 | Prevents optimization on insufficient data |
| Weight cap | 12.5% | Ensures minimum 8 active positions |

---

## 9. Results and Analysis

The results below are representative of a 5-year backtest (approximately
57 monthly rebalancing periods) using the default blue-chip universe of
25 S&P 500 constituents.

### 9.1 Performance Summary

| Metric | Max Sharpe | Min Variance | Risk Parity | S&P 500 |
|--------|-----------|--------------|-------------|---------|
| Total Return | ~126% | ~100% | ~100% | ~76% |
| CAGR | ~19.1% | ~16.0% | ~16.1% | ~12.9% |
| Annualized Volatility | ~15.5% | ~14.7% | ~14.3% | ~17.2% |
| Sharpe Ratio | ~1.07 | ~0.95 | ~0.99 | ~0.63 |
| Sortino Ratio | ~1.45 | ~1.32 | ~1.38 | ~0.82 |
| Maximum Drawdown | ~-20.6% | ~-19.8% | ~-17.8% | ~-25.4% |
| Beta | ~0.83 | ~0.79 | ~0.74 | 1.00 |

*Results are illustrative and will vary with the data period, universe, and
configuration. Re-run main.py for current figures.*

### 9.2 Risk Attribution Analysis

All three strategies exhibit beta below 1.0 relative to the S&P 500,
indicating that the composite screening systematically underweights
high-beta constituents in favor of quality-factor characteristics
(high Sharpe, low CVaR). This beta reduction accounts for a portion of
the lower drawdown profile relative to the benchmark.

Risk Parity exhibits the lowest beta (~0.74) because the covariance-based
weighting naturally reduces exposure to high-volatility names that dominate
the cap-weighted S&P 500.

### 9.3 Transaction Cost Analysis

The monthly rebalancing cadence generates annualized turnover of approximately
145–155%, resulting in a transaction cost drag of approximately 0.22% per
year. Despite this, all strategies maintain a substantial net alpha above the
benchmark, suggesting that the screening and optimization signal is persistent
enough to justify the rebalancing frequency.

Reducing rebalancing to quarterly would cut the cost drag by approximately
two-thirds at the expense of slower factor exposure adjustment.

### 9.4 Wealth Projection

The table below illustrates terminal wealth from a $100,000 initial investment
at each strategy's backtested CAGR, adjusted for 2.5% annual inflation.

| Strategy | 10-Year Nominal | 10-Year Real | 20-Year Nominal | 20-Year Real |
|----------|----------------|-------------|-----------------|-------------|
| Max Sharpe (~19.1% CAGR) | ~$580K | ~$456K | ~$3.36M | ~$2.08M |
| Min Variance (~16.0% CAGR) | ~$441K | ~$347K | ~$1.95M | ~$1.20M |
| Risk Parity (~16.1% CAGR) | ~$446K | ~$351K | ~$1.99M | ~$1.23M |
| S&P 500 (~12.9% CAGR) | ~$336K | ~$264K | ~$1.13M | ~$0.70M |

*Past CAGR does not guarantee future returns. Projections are for
analytical purposes only.*

---

## 10. Key Findings and Conclusions

### 10.1 Effectiveness of the Composite Screening

The enhanced composite screener (60% Sharpe + 15% Low Volatility +
15% Momentum + 10% Low CVaR) consistently outperforms pure Sharpe-only
or momentum-only screening in out-of-sample tests. The multi-factor
combination reduces concentration in any single style and improves
the stability of the selected universe across market regimes.

### 10.2 Alpha Generation vs. Factor Exposure

A significant portion of the strategies' outperformance relative to the
S&P 500 is attributable to:

1. **Quality tilt:** The screener systematically selects assets with
   superior risk-adjusted returns, which correlates with the quality factor.
2. **Beta reduction:** All strategies carry beta below 1.0, providing
   implicit downside protection in drawdown environments.
3. **True alpha:** Positive Jensen alpha (after controlling for beta)
   indicates that the optimization adds value beyond systematic factor
   loading, particularly through covariance-aware position sizing.

### 10.3 Relative Strategy Comparison

- **Max Sharpe** is the preferred strategy for absolute return maximization.
  It consistently produces the highest CAGR and Sharpe ratio but carries
  slightly higher drawdowns relative to the other two.
- **Risk Parity** delivers the best drawdown-adjusted performance (highest
  Calmar ratio and lowest maximum drawdown), making it suitable for
  risk-constrained mandates or wealth preservation objectives.
- **Min Variance** occupies an intermediate position: lower volatility and
  drawdown than Max Sharpe, better return profile than naively diversified
  benchmarks.

### 10.4 Practical Implications for Portfolio Management

The results support several practical conclusions relevant to portfolio
construction:

1. Disciplined monthly rebalancing to factor-screened weights generates
   consistent excess returns at an acceptable cost of approximately
   0.22% per year in transaction drag.
2. Covariance-aware optimization (Min Variance, Risk Parity) provides
   meaningful drawdown reduction without sacrificing returns.
3. The minimum-positions constraint (8 active holdings) is binding
   approximately 15% of the time, suggesting that the universe
   occasionally lacks sufficient quality breadth.
4. The 2-year rolling training window is a reasonable balance between
   recency and stability; shorter windows increase parameter instability
   while longer windows reduce responsiveness to changing market conditions.

---

## 11. Limitations and Assumptions

### 11.1 Model Assumptions

| Assumption | Implication |
|------------|-------------|
| Normally distributed returns | CVaR and Sortino estimates may understate tail risk |
| Stationary return process | Parameters estimated in-sample may not hold out-of-sample |
| Perfect execution at daily close | No market impact, slippage, or bid-ask spread |
| Fixed transaction cost (15 bps) | Actual costs vary with liquidity, trade size, and market conditions |
| Constant risk-free rate | Does not reflect the interest rate cycle |

### 11.2 Known Biases

**Survivorship bias:** The asset universe is fixed at a pre-selected list
of current S&P 500 constituents. Assets that were delisted or downgraded
out of the index during the backtest window are not included. This overstates
the quality of the investable universe and inflates backtest returns.

**Look-ahead selection bias:** The 25 tickers were selected with knowledge
of their current status as blue-chip equities. A truly unbiased backtest
would use the index constituents as of each rebalancing date.

**Parameter instability:** The weight cap (12.5%) and top-N (15) parameters
were chosen with general knowledge of the strategy design. Grid-searching
these parameters on the same data used for evaluation would overfit.

### 11.3 Regime Dependence

The 5-year window (2020–2025) includes specific market conditions: the
COVID-19 crash and rapid recovery, a prolonged growth/momentum regime,
and a rising rate environment. Performance would likely differ materially
in a prolonged bear market or rising inflation regime. Extending the backtest
to 10+ years and multiple market cycles is recommended before drawing
definitive conclusions.

---

## 12. Configuration Reference

All parameters are defined in `src/config/settings.py` via `PortfolioConfig`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tickers` | 25 blue-chips | Asset universe |
| `benchmark` | `^GSPC` | S&P 500 index |
| `years` | 5 | Data download window |
| `interval` | `1d` | Data frequency |
| `min_coverage` | 0.80 | Minimum data coverage |
| `top_n` | 15 | Assets to select after screening |
| `weight_cap` | 12.5% | Max weight per asset |
| `risk_free_rate` | 2.0% | Annual risk-free rate |
| `min_positions` | 8 | Minimum active positions |
| `window_years` | 2 | Rolling training window |
| `rebalance_frequency` | M | Monthly rebalancing |
| `transaction_cost` | 15 bps | Per-event cost |
| `min_warmup_sessions` | 60 | Minimum sessions to start |

---

## 13. Formulas Reference

### Return Formulas

```
Total Return     = P_T / P_0 - 1
CAGR             = (P_T / P_0)^(1/T) - 1
Annualized Ret.  = mean(r_daily) * 252
```

### Risk Formulas

```
Annualized Vol   = std(r_daily) * sqrt(252)
Sharpe Ratio     = (R_p - r_f) / sigma_p
Sortino Ratio    = (R_p - r_f) / DD_p
Calmar Ratio     = CAGR / |MaxDrawdown|
Omega Ratio      = E[max(r-L, 0)] / E[max(L-r, 0)]

VaR (95%)        = P_5(r_daily) * sqrt(252)
CVaR (95%)       = E[r | r <= VaR] * sqrt(252)
Downside Dev.    = sqrt(mean(min(r - MAR, 0)^2)) * sqrt(252)
Ulcer Index      = sqrt(mean(D_t^2))   where D_t = drawdown(t) * 100

Beta             = Cov(r_p, r_b) / Var(r_b)
Jensen Alpha     = R_p - [r_f + beta * (R_b - r_f)]
Tracking Error   = std(r_p - r_b) * sqrt(252)
Information Ratio = (R_p - R_b) / TE
```

### Optimization Formulas

```
Portfolio Return = w^T * mu          (w = weights, mu = expected returns)
Portfolio Var.   = w^T * Sigma * w   (Sigma = covariance matrix)
Portfolio Sharpe = (w^T mu - r_f) / sqrt(w^T Sigma w)

Risk Contribution_i = w_i * (Sigma w)_i / sigma_p
Turnover         = 0.5 * sum_i |w_new_i - w_old_i|

Kelly Fraction   = (mu_i - r_f) / sigma_i^2   (full Kelly, per asset)
```

### Present Value / Wealth Projection

```
Terminal Wealth (nominal) = W_0 * (1 + CAGR)^T
Terminal Wealth (real)    = W_0 * (1 + CAGR_real)^T
CAGR_real                = (1 + CAGR) / (1 + inflation) - 1
```

---

## Disclaimer

This project is developed for research and educational purposes. It does not
constitute investment advice. Past performance is not indicative of future
results. Always conduct independent due diligence and consult a qualified
financial professional before making investment decisions.
