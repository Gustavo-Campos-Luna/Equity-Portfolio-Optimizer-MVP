"""
Portfolio optimization module.

Implements three classical mean-variance optimization strategies and
a Kelly-criterion-based position sizing utility.

Strategies
----------
Max Sharpe    Maximize risk-adjusted excess return over the risk-free rate.
Min Variance  Minimize total portfolio variance (GMVP).
Risk Parity   Equalize the marginal risk contribution of each asset.

All strategies share a common constraint framework:
  - Weights sum to 1 (fully invested, long-only).
  - 0 <= w_i <= weight_cap for all assets.
  - Minimum number of active positions enforced via a dynamic cap.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

from src.config.settings import DEFAULT_CONFIG, PortfolioConfig
from src.metrics.financial_metrics import MetricsCalculator

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class PortfolioOptimizer:
    """
    Mean-variance portfolio optimizer with risk-parity extension.

    Parameters
    ----------
    config : PortfolioConfig
        Optimization constraints and risk-free rate.
    """

    def __init__(self, config: PortfolioConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._metrics = MetricsCalculator(config)

    # ------------------------------------------------------------------
    # Public optimizers
    # ------------------------------------------------------------------

    def optimize_max_sharpe(
        self,
        prices: pd.DataFrame,
        picks: List[str],
    ) -> Tuple[pd.Series, Tuple[float, float, float]]:
        """
        Maximize portfolio Sharpe ratio.

        Objective (minimization form):
            min_w  -[(w^T mu - rf) / sqrt(w^T Sigma w)]

        Constraints: sum(w) = 1,  0 <= w_i <= cap

        Parameters
        ----------
        prices : pd.DataFrame
            Historical adjusted prices (in-sample window only).
        picks : list of str
            Asset tickers to include in optimization.

        Returns
        -------
        weights : pd.Series
            Optimized weights, indexed by ticker.
        performance : tuple
            (annualized_return, annualized_vol, sharpe_ratio)
        """
        returns, cap, n = self._prepare(prices, picks)

        def objective(w: np.ndarray) -> float:
            _, _, sharpe = self._portfolio_stats(w, returns)
            return -sharpe

        result = self._run_optimizer(objective, n, cap)
        weights = self._build_weights(result.x, picks)
        perf = self._portfolio_stats(result.x, returns)
        return weights, perf

    def optimize_min_variance(
        self,
        prices: pd.DataFrame,
        picks: List[str],
    ) -> Tuple[pd.Series, Tuple[float, float, float]]:
        """
        Global Minimum Variance Portfolio (GMVP).

        Objective:
            min_w  w^T Sigma w

        Returns portfolio with the lowest attainable variance given
        the diversification constraints.
        """
        returns, cap, n = self._prepare(prices, picks)
        cov = returns.cov().values * TRADING_DAYS

        def objective(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        result = self._run_optimizer(objective, n, cap)
        weights = self._build_weights(result.x, picks)
        perf = self._portfolio_stats(result.x, returns)
        return weights, perf

    def optimize_risk_parity(
        self,
        prices: pd.DataFrame,
        picks: List[str],
    ) -> Tuple[pd.Series, Tuple[float, float, float]]:
        """
        Risk Parity (Equal Risk Contribution) portfolio.

        Objective:
            min_w  sum_i [ w_i * (Sigma w)_i / sigma_p - sigma_p/N ]^2

        Each asset contributes an equal fraction of total portfolio risk.
        Ledoit-Wolf shrinkage is applied to the sample covariance matrix
        to improve conditioning in small samples.

        Bounds: 1% <= w_i <= 40% (prevent degenerate solutions).
        """
        returns, _, n = self._prepare(prices, picks)

        # Ledoit-Wolf shrinkage: blend sample cov with scaled identity
        cov_sample = returns.cov().values * TRADING_DAYS
        shrinkage = 0.10
        cov_target = np.eye(n) * np.trace(cov_sample) / n
        cov = (1 - shrinkage) * cov_sample + shrinkage * cov_target

        def objective(w: np.ndarray) -> float:
            vol = np.sqrt(w @ cov @ w)
            if vol <= 0:
                return 1e6
            marginal = (cov @ w) / vol
            risk_contrib = w * marginal
            target = vol / n
            return float(np.sum((risk_contrib - target) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((0.01, 0.40) for _ in range(n))
        x0 = np.full(n, 1.0 / n)

        result: OptimizeResult = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            logger.warning(
                "Risk Parity optimizer did not converge (%s). Falling back to equal weight.",
                result.message,
            )

        final_w = result.x if result.success else x0
        weights = self._build_weights(final_w, picks)
        perf = self._portfolio_stats(final_w, returns)
        return weights, perf

    # ------------------------------------------------------------------
    # Kelly position sizing (supplementary)
    # ------------------------------------------------------------------

    def kelly_weights(
        self,
        prices: pd.DataFrame,
        picks: List[str],
        fractional: float = 0.5,
    ) -> pd.Series:
        """
        Compute Kelly-optimal position sizes (fractional Kelly).

        Full Kelly:  f_i* = (mu_i - rf) / sigma_i^2

        A fractional Kelly multiplier (default 0.5) is applied to reduce
        variance while preserving the directional signal. Weights are
        normalized to sum to 1. Negative Kelly values are set to zero
        (no short positions).

        Parameters
        ----------
        prices : pd.DataFrame
            In-sample price history.
        picks : list of str
            Assets to consider.
        fractional : float
            Kelly fraction in (0, 1]. Use < 1.0 to reduce volatility.

        Returns
        -------
        pd.Series
            Normalized Kelly weights indexed by ticker.
        """
        rf = self.config.risk_free_rate
        sub = prices[picks].dropna()
        returns = sub.pct_change().dropna()
        ann_ret = returns.mean() * TRADING_DAYS
        ann_var = (returns.std() * np.sqrt(TRADING_DAYS)) ** 2

        kelly = fractional * (ann_ret - rf) / ann_var.replace(0, np.nan)
        kelly = kelly.clip(lower=0).fillna(0)

        total = kelly.sum()
        if total <= 0:
            return pd.Series(1.0 / len(picks), index=picks)
        return kelly / total

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _prepare(
        self,
        prices: pd.DataFrame,
        picks: List[str],
    ) -> Tuple[pd.DataFrame, float, int]:
        """Prepare returns, compute dynamic weight cap, return (returns, cap, n)."""
        sub = prices[picks].dropna()
        returns = sub.pct_change().dropna()
        returns = MetricsCalculator.clip_outliers(returns)
        n = len(picks)
        cap = self._dynamic_cap(n)
        return returns, cap, n

    def _dynamic_cap(self, n_assets: int) -> float:
        """
        Compute the effective per-asset weight cap.

        Ensures that (n_assets * cap >= 1) to maintain feasibility, while
        also enforcing the minimum-positions requirement.
        """
        cfg = self.config
        natural_cap = 1.0 / max(cfg.min_positions, n_assets)
        cap = min(cfg.weight_cap, natural_cap)
        min_feasible = 1.0 / n_assets
        if cap * n_assets < 1.01:
            cap = max(min_feasible, cap * 1.10)
        return cap

    def _portfolio_stats(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
    ) -> Tuple[float, float, float]:
        """Return (annualized_return, annualized_vol, sharpe_ratio)."""
        rf = self.config.risk_free_rate
        ann_ret = float((returns.mean() * TRADING_DAYS).dot(weights))
        cov = returns.cov().values * TRADING_DAYS
        ann_vol = float(np.sqrt(weights @ cov @ weights))
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
        return ann_ret, ann_vol, sharpe

    def _run_optimizer(
        self,
        objective: callable,
        n: int,
        cap: float,
    ) -> OptimizeResult:
        """Run SLSQP optimizer with standard constraints and fallback."""
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((0.0, cap) for _ in range(n))
        x0 = np.full(n, 1.0 / n)

        result: OptimizeResult = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            logger.warning(
                "Optimizer did not converge (%s). Using equal weight.",
                result.message,
            )
            result.x = x0

        return result

    @staticmethod
    def _build_weights(raw_weights: np.ndarray, picks: List[str]) -> pd.Series:
        """Construct a Series of weights, dropping near-zero positions."""
        weights = pd.Series(raw_weights, index=picks)
        return weights[weights > 1e-4].sort_values(ascending=False)
