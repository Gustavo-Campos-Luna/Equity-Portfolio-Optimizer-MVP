"""
Rolling backtest engine.

Implements a strict walk-forward backtesting framework with no lookahead
bias. At each rebalancing date t:
  1. The model is trained on the window [t - window_years, t].
  2. Weights are applied to returns in (t, t+1] (out-of-sample).
  3. Transaction costs are deducted from the first day of each period.

Critical design decisions
-------------------------
- clip_outliers is applied to in-sample returns only. Applying it to the
  full time series before splitting would introduce lookahead bias.
- The quality filter is called with window_years rather than the total
  sample size, so the minimum-session threshold reflects the rolling window.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import PortfolioConfig, DEFAULT_CONFIG
from src.metrics.financial_metrics import MetricsCalculator
from src.optimization.asset_screener import AssetScreener
from src.optimization.portfolio_optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

_FREQ_PERIODS: Dict[str, int] = {"M": 12, "Q": 4, "A": 1, "W": 52}


class BacktestEngine:
    """
    Walk-forward rolling backtest framework.

    Parameters
    ----------
    config : PortfolioConfig
        Controls window size, rebalancing frequency, costs, and constraints.
    """

    def __init__(self, config: PortfolioConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._calc = MetricsCalculator(config)
        self._screener = AssetScreener(config)
        self._optimizer = PortfolioOptimizer(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.DataFrame,
        universe: List[str],
        optimization_method: str = "max_sharpe",
        screening_method: str = "enhanced_composite",
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Execute rolling backtest over the full price history.

        Parameters
        ----------
        prices : pd.DataFrame
            Full price history including benchmark column.
        universe : List[str]
            Asset tickers to consider (benchmark excluded automatically).
        optimization_method : str
            One of: 'max_sharpe', 'min_variance', 'risk_parity'.
        screening_method : str
            Passed to AssetScreener.screen(). See AssetScreener docs.

        Returns
        -------
        portfolio_returns : pd.Series
            Daily net-of-cost out-of-sample portfolio returns.
        turnover_df : pd.DataFrame
            Per-period turnover statistics with columns ['date', 'turnover'].
        """
        cfg = self.config
        logger.info(
            "Starting backtest: method=%s, screening=%s, "
            "window=%dY, rebalance=%s.",
            optimization_method, screening_method,
            cfg.window_years, cfg.rebalance_frequency,
        )

        px = prices[universe].dropna(how="all")
        # Raw returns — no outlier clipping here (in-sample only)
        returns = px.pct_change()
        dates = returns.index

        rebalance_dates = (
            pd.Series(index=dates, data=1)
            .resample(cfg.rebalance_frequency)
            .last()
            .index
        )

        portfolio_returns = pd.Series(index=dates, dtype=float)
        turnover_records: List[Dict] = []
        previous_weights = pd.Series(dtype=float)
        rebalance_count = 0

        for t in rebalance_dates:
            start_date = t - pd.DateOffset(years=cfg.window_years)
            hist_data = px.loc[
                (px.index > start_date) & (px.index <= t)
            ].dropna(how="all", axis=1)

            if len(hist_data) < cfg.min_warmup_sessions:
                logger.debug(
                    "Skipping %s: %d sessions < %d required.",
                    t.date(), len(hist_data), cfg.min_warmup_sessions,
                )
                continue

            hist_data = self._calc.quality_filter(
                hist_data, window_years=cfg.window_years
            )

            if hist_data.shape[1] < cfg.min_positions:
                logger.debug(
                    "Skipping %s: only %d assets pass quality filter.",
                    t.date(), hist_data.shape[1],
                )
                continue

            try:
                weights = self._compute_weights(
                    hist_data, optimization_method, screening_method
                )
            except Exception as exc:
                logger.warning("Skipping rebalance at %s: %s", t.date(), exc)
                continue

            turnover = self._calc.calculate_turnover(weights, previous_weights)
            turnover_records.append({"date": t, "turnover": turnover})
            previous_weights = weights.copy()

            # Out-of-sample period: (t, next_rebalance_date]
            future_dates = rebalance_dates[rebalance_dates > t]
            end_date = future_dates[0] if len(future_dates) > 0 else dates[-1]

            period_mask = (returns.index > t) & (returns.index <= end_date)
            period_ret = returns.loc[period_mask, weights.index]

            if period_ret.empty:
                continue

            gross = period_ret @ weights.values
            net = gross.copy()
            net.iloc[0] -= cfg.transaction_cost * turnover
            portfolio_returns.loc[period_ret.index] = net
            rebalance_count += 1

        logger.info(
            "Backtest complete: %d rebalancing periods executed.",
            rebalance_count,
        )

        turnover_df = pd.DataFrame(turnover_records)
        return portfolio_returns.dropna(), turnover_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        hist_data: pd.DataFrame,
        optimization_method: str,
        screening_method: str,
    ) -> pd.Series:
        """Screen assets and run the chosen optimizer."""
        benchmark = self.config.benchmark
        metrics = self._calc.compute_asset_metrics(
            hist_data, benchmark=benchmark, rf=self.config.risk_free_rate
        )
        selected = self._screener.screen(
            metrics, top_n=self.config.top_n, method=screening_method
        )
        picks = selected.index.tolist()

        if len(picks) < self.config.min_positions:
            raise ValueError(
                f"Only {len(picks)} assets selected; "
                f"minimum {self.config.min_positions} required."
            )

        opt = self._optimizer
        if optimization_method == "max_sharpe":
            weights, _ = opt.optimize_max_sharpe(hist_data, picks)
        elif optimization_method == "min_variance":
            weights, _ = opt.optimize_min_variance(hist_data, picks)
        elif optimization_method == "risk_parity":
            weights, _ = opt.optimize_risk_parity(hist_data, picks)
        else:
            raise ValueError(
                f"Unknown optimization_method: '{optimization_method}'. "
                "Valid options: 'max_sharpe', 'min_variance', 'risk_parity'."
            )

        return weights
