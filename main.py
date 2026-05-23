"""
Equity Portfolio Optimizer — main execution entry point.

Orchestrates the full pipeline:
  1. Data acquisition
  2. Asset screening and portfolio optimization (3 strategies)
  3. Rolling out-of-sample backtest
  4. Performance analytics and conclusions
  5. Visualization suite

Run
---
    python main.py

Optional arguments (edit PortfolioConfig in src/config/settings.py to
customize universe, constraints, and backtest parameters).
"""
from __future__ import annotations

import logging
import sys
from typing import Dict

import pandas as pd

from src.config.settings import PortfolioConfig
from src.backtesting.backtest_engine import BacktestEngine
from src.data.market_data import DataFetcher
from src.metrics.financial_metrics import MetricsCalculator
from src.optimization.asset_screener import AssetScreener
from src.optimization.portfolio_optimizer import PortfolioOptimizer
from src.reporting.performance_report import PerformanceReport
from src.visualization.charts import ChartEngine

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

STRATEGIES = ["max_sharpe", "min_variance", "risk_parity"]


def run(config: PortfolioConfig = None, initial_capital: float = 100_000.0) -> None:
    """
    Execute the full portfolio optimization and backtesting pipeline.

    Parameters
    ----------
    config : PortfolioConfig, optional
        Custom configuration. Defaults to PortfolioConfig() with standard
        blue-chip universe and constraints.
    initial_capital : float
        Starting portfolio value used for wealth projection calculations.
    """
    config = config or PortfolioConfig()
    config.validate()

    fetcher = DataFetcher(config)
    calc = MetricsCalculator(config)
    screener = AssetScreener(config)
    optimizer = PortfolioOptimizer(config)
    backtest = BacktestEngine(config)
    reporter = PerformanceReport(config)
    charts = ChartEngine(config, output_dir="output/charts", show=False)

    # -----------------------------------------------------------------------
    # Step 1: Data acquisition
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("STEP 1: Acquiring market data")
    logger.info("=" * 70)
    prices = fetcher.fetch_prices()

    # -----------------------------------------------------------------------
    # Step 2: Quality filter and current metrics
    # -----------------------------------------------------------------------
    logger.info("STEP 2: Data quality filter and asset metrics")
    clean_prices = calc.quality_filter(prices, verbose=True)
    metrics = calc.compute_asset_metrics(
        clean_prices, benchmark=config.benchmark, rf=config.risk_free_rate
    )

    print("\n--- Top 15 Assets by Sharpe Ratio ---")
    display_cols = [
        "CAGR", "AnnualizedVol", "Sharpe", "Sortino",
        "Calmar", "CVaR_95", "Momentum12M", "MaxDrawdown",
    ]
    print(metrics[display_cols].head(15).round(4).to_string())

    # -----------------------------------------------------------------------
    # Step 3: Asset screening
    # -----------------------------------------------------------------------
    logger.info("STEP 3: Enhanced composite asset screening")
    selected = screener.screen(metrics, method="enhanced_composite")
    picks = selected.index.tolist()

    print(f"\n--- Selected Assets ({len(picks)}) ---")
    print(selected[["CAGR", "AnnualizedVol", "Sharpe", "Sortino", "Score"]].round(4).to_string())

    # -----------------------------------------------------------------------
    # Step 4: Current portfolio optimization
    # -----------------------------------------------------------------------
    logger.info("STEP 4: Portfolio optimization (current snapshot)")
    current_weights: Dict[str, pd.Series] = {}

    for strategy_key, label in [
        ("max_sharpe", "Max Sharpe"),
        ("min_variance", "Min Variance"),
        ("risk_parity", "Risk Parity"),
    ]:
        if strategy_key == "max_sharpe":
            weights, perf = optimizer.optimize_max_sharpe(clean_prices, picks)
        elif strategy_key == "min_variance":
            weights, perf = optimizer.optimize_min_variance(clean_prices, picks)
        else:
            weights, perf = optimizer.optimize_risk_parity(clean_prices, picks)

        current_weights[strategy_key] = weights
        print(f"\n--- {label} Portfolio ---")
        print(weights.round(4).to_string())
        print(
            f"  Expected Return: {perf[0]:.2%} | "
            f"Volatility: {perf[1]:.2%} | "
            f"Sharpe: {perf[2]:.2f}"
        )

        # Risk attribution
        sub_returns = clean_prices[picks].pct_change().dropna()
        risk_attr = calc.portfolio_risk_attribution(weights, sub_returns[weights.index])
        print("  Risk Contribution (%):")
        print("  " + risk_attr.sort_values(ascending=False).round(4).to_string())

    # -----------------------------------------------------------------------
    # Step 5: Kelly position sizing (supplementary)
    # -----------------------------------------------------------------------
    logger.info("STEP 5: Fractional Kelly position sizing (informational)")
    kelly_w = optimizer.kelly_weights(clean_prices, picks, fractional=0.5)
    print("\n--- Fractional Kelly Weights (0.5x) ---")
    print(kelly_w[kelly_w > 0.005].sort_values(ascending=False).round(4).to_string())

    # -----------------------------------------------------------------------
    # Step 6: Rolling backtest
    # -----------------------------------------------------------------------
    logger.info("STEP 6: Rolling out-of-sample backtest")
    universe = [c for c in clean_prices.columns if c != config.benchmark]
    backtest_results: Dict[str, Dict] = {}
    strategy_returns: Dict[str, pd.Series] = {}

    bench_ret = clean_prices[config.benchmark].pct_change().dropna()

    for strategy in STRATEGIES:
        logger.info("  Running backtest: %s", strategy)
        bt_returns, turnover_df = backtest.run(
            clean_prices, universe,
            optimization_method=strategy,
            screening_method="enhanced_composite",
        )

        if bt_returns.empty:
            logger.warning("  No returns generated for %s.", strategy)
            continue

        bench_aligned = bench_ret.reindex(bt_returns.index).dropna()
        bt_aligned = bt_returns.reindex(bench_aligned.index)

        strategy_returns[strategy] = bt_aligned
        result = reporter.compute_metrics(bt_aligned, bench_aligned, turnover_df)
        backtest_results[strategy] = result

        print(reporter.format_report(result, strategy.replace("_", " ").title()))

    # -----------------------------------------------------------------------
    # Step 7: Conclusions and strategy comparison
    # -----------------------------------------------------------------------
    if backtest_results:
        logger.info("STEP 7: Generating analytical conclusions")

        comparison = reporter.comparison_table(backtest_results)
        print("\n--- Strategy Comparison Table ---")
        print(comparison.round(4).to_string())

        print(reporter.generate_conclusions(backtest_results))

        # Wealth projection
        projection = reporter.wealth_projection(
            initial_capital=initial_capital,
            strategies=backtest_results,
            horizon_years=[5, 10, 15, 20, 30],
            inflation_rate=0.025,
        )
        print("\n--- Wealth Projection (Initial Capital: ${:,.0f}) ---".format(initial_capital))
        print(projection.to_string())

    # -----------------------------------------------------------------------
    # Step 8: Visualization suite
    # -----------------------------------------------------------------------
    logger.info("STEP 8: Generating visualization suite")

    for strategy in STRATEGIES:
        if strategy in strategy_returns:
            port_ret = strategy_returns[strategy]
            bench_aligned = bench_ret.reindex(port_ret.index).dropna()
            charts.performance_dashboard(
                port_ret, bench_aligned,
                strategy.replace("_", " ").title(),
            )

    if backtest_results and strategy_returns:
        projection_for_charts = (
            reporter.wealth_projection(
                initial_capital=initial_capital,
                strategies=backtest_results,
                horizon_years=[5, 10, 15, 20, 30],
            )
            if backtest_results
            else None
        )
        charts.comprehensive_dashboard(
            backtest_results=backtest_results,
            strategy_returns=strategy_returns,
            current_weights=current_weights,
            projection_df=projection_for_charts,
            initial_capital=initial_capital,
        )

    # -----------------------------------------------------------------------
    # Step 9: Configuration summary
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Pipeline complete.")
    print("\n--- Configuration Summary ---")
    print(f"  Asset universe:         {len(config.tickers)} tickers")
    print(f"  Benchmark:              {config.benchmark}")
    print(f"  Historical window:      {config.years} years")
    print(f"  Training window:        {config.window_years} years")
    print(f"  Rebalancing:            {config.rebalance_frequency}")
    print(f"  Risk-free rate:         {config.risk_free_rate:.2%}")
    print(f"  Weight cap:             {config.weight_cap:.1%}")
    print(f"  Transaction cost:       {config.transaction_cost:.2%}")
    print(f"  Top-N assets:           {config.top_n}")
    print(f"  Minimum positions:      {config.min_positions}")
    print(f"  Output charts:          output/charts/")


if __name__ == "__main__":
    run()
