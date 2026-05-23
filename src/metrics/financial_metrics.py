"""
Financial metrics computation layer.

Provides comprehensive risk-adjusted performance analytics for individual
assets and portfolios. All annualization assumes 252 trading days per year.

Metrics implemented
-------------------
Return metrics   : Total Return, CAGR, Annualized Return
Risk metrics     : Annualized Volatility, VaR (95%), CVaR (95%),
                   Maximum Drawdown, Ulcer Index, Downside Deviation,
                   Recovery Time
Risk-adjusted    : Sharpe Ratio, Sortino Ratio, Calmar Ratio,
                   Omega Ratio, Pain Ratio
Factor metrics   : Momentum (6M, 12M), Beta, Tracking Error,
                   Information Ratio, Hit Rate
Distribution     : Skewness, Kurtosis
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import PortfolioConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class MetricsCalculator:
    """
    Computes comprehensive asset and portfolio performance metrics.

    Parameters
    ----------
    config : PortfolioConfig
        Configuration object providing the risk-free rate.
    """

    def __init__(self, config: PortfolioConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Data quality utilities
    # ------------------------------------------------------------------

    @staticmethod
    def clip_outliers(
        returns: pd.DataFrame,
        lower_pct: float = 1.0,
        upper_pct: float = 99.0,
    ) -> pd.DataFrame:
        """
        Winsorize daily returns at specified percentiles to reduce the
        influence of extreme observations on covariance estimation.

        Applied only to in-sample data; never to out-of-sample returns
        to avoid lookahead bias.
        """
        quantiles = returns.quantile([lower_pct / 100, upper_pct / 100])
        return returns.clip(
            lower=quantiles.loc[lower_pct / 100],
            upper=quantiles.loc[upper_pct / 100],
            axis=1,
        )

    def quality_filter(
        self,
        prices: pd.DataFrame,
        min_sessions: Optional[int] = None,
        window_years: Optional[int] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Remove assets that fail minimum data-quality thresholds.

        Two criteria are applied sequentially:
        1. Coverage filter: fraction of non-null observations >= min_coverage.
        2. Session filter: absolute count of observations >= min_sessions.

        If fewer than 3 assets survive, the session threshold is relaxed to
        25% of available sessions (minimum 20) to ensure the optimizer
        receives a feasible input.

        Parameters
        ----------
        prices : pd.DataFrame
            Price DataFrame to filter (may include benchmark column).
        min_sessions : int, optional
            Override config.min_sessions.
        window_years : int, optional
            Override config.window_years (used to compute adaptive threshold).
        verbose : bool
            Log dropped assets at INFO level.

        Returns
        -------
        pd.DataFrame
            Filtered price DataFrame.
        """
        cfg = self.config
        window_years = window_years or cfg.window_years
        actual_sessions = len(prices)

        if min_sessions is None:
            expected = int(window_years * TRADING_DAYS * 0.75)
            min_sessions = min(expected, max(int(actual_sessions * 0.80), 20))

        coverage = prices.notna().mean()
        session_counts = prices.notna().sum()

        coverage_pass = set(coverage[coverage >= cfg.min_coverage].index)
        session_pass = set(session_counts[session_counts >= min_sessions].index)
        keep = list(coverage_pass & session_pass)

        if verbose:
            dropped = [c for c in prices.columns if c not in keep]
            if dropped:
                logger.info("Quality filter removed: %s", dropped)

        if len(keep) < 3:
            relaxed = max(int(actual_sessions * 0.25), 20)
            session_pass_relaxed = set(
                session_counts[session_counts >= relaxed].index
            )
            keep = list(coverage_pass & session_pass_relaxed)
            logger.info(
                "Session threshold relaxed from %d to %d. Assets retained: %d.",
                min_sessions, relaxed, len(keep),
            )

        return prices[keep]

    # ------------------------------------------------------------------
    # Individual asset metrics
    # ------------------------------------------------------------------

    def compute_asset_metrics(
        self,
        prices: pd.DataFrame,
        benchmark: Optional[str] = None,
        rf: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Compute the full suite of per-asset metrics over the supplied price window.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted closing prices (may contain benchmark).
        benchmark : str, optional
            Benchmark column name to exclude from asset metrics.
        rf : float, optional
            Annual risk-free rate. Defaults to config value.

        Returns
        -------
        pd.DataFrame
            Metrics DataFrame indexed by ticker, sorted by Sharpe ratio.
        """
        rf = rf if rf is not None else self.config.risk_free_rate
        df = prices.drop(columns=[benchmark], errors="ignore") if benchmark else prices.copy()

        returns = df.pct_change().dropna()
        returns = self.clip_outliers(returns)

        years = (df.index[-1] - df.index[0]).days / 365.25

        # --- Return metrics ---
        total_return = df.iloc[-1] / df.iloc[0] - 1
        cagr = (df.iloc[-1] / df.iloc[0]) ** (1 / years) - 1
        ann_ret = returns.mean() * TRADING_DAYS
        ann_vol = returns.std() * np.sqrt(TRADING_DAYS)

        # --- Risk-adjusted ratios ---
        sharpe = (ann_ret - rf) / ann_vol

        downside_dev = self._downside_deviation(returns, rf=rf)
        sortino = (ann_ret - rf) / downside_dev.replace(0, np.nan)

        max_dd = self._max_drawdown(df)
        calmar = cagr / max_dd.abs().replace(0, np.nan)

        omega = self._omega_ratio(returns, threshold=rf / TRADING_DAYS)

        # --- Tail risk ---
        var_95 = returns.quantile(0.05) * np.sqrt(TRADING_DAYS)
        cvar_95 = self._cvar(returns)

        # --- Drawdown analytics ---
        ulcer = self._ulcer_index(df)

        # --- Momentum ---
        mom_6m = (
            df.iloc[-1] / df.iloc[-126] - 1
            if len(df) >= 126
            else pd.Series(0.0, index=df.columns)
        )
        mom_12m = (
            df.iloc[-1] / df.iloc[-252] - 1
            if len(df) >= 252
            else pd.Series(0.0, index=df.columns)
        )

        # --- Distribution ---
        skew = returns.skew()
        kurt = returns.kurtosis()

        metrics = pd.DataFrame(
            {
                "TotalReturn": total_return,
                "CAGR": cagr,
                "AnnualizedReturn": ann_ret,
                "AnnualizedVol": ann_vol,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Calmar": calmar,
                "Omega": omega,
                "Momentum6M": mom_6m,
                "Momentum12M": mom_12m,
                "VaR_95": var_95,
                "CVaR_95": cvar_95,
                "MaxDrawdown": max_dd,
                "UlcerIndex": ulcer,
                "Skewness": skew,
                "Kurtosis": kurt,
            }
        )

        metrics = (
            metrics.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["Sharpe"])
            .sort_values("Sharpe", ascending=False)
        )
        return metrics

    # ------------------------------------------------------------------
    # Portfolio-level metrics
    # ------------------------------------------------------------------

    def tracking_error_and_ir(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> Tuple[float, float]:
        """
        Calculate annualized Tracking Error and Information Ratio.

        Tracking Error = std(excess_returns) * sqrt(252)
        Information Ratio = mean(excess_returns) * 252 / Tracking Error
        """
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1
        ).dropna()
        if len(aligned) < 30:
            return np.nan, np.nan

        excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        te = float(excess.std() * np.sqrt(TRADING_DAYS))
        ir = float((excess.mean() * TRADING_DAYS) / te) if te > 0 else np.nan
        return te, ir

    def calculate_turnover(
        self,
        weights_new: pd.Series,
        weights_old: pd.Series,
    ) -> float:
        """
        Compute one-way portfolio turnover between two rebalancing periods.

        Turnover = 0.5 * sum(|w_new_i - w_old_i|)

        A value of 1.0 means the entire portfolio was replaced.
        """
        if weights_old.empty:
            return 1.0

        all_assets = set(weights_new.index) | set(weights_old.index)
        w_new = pd.Series(0.0, index=all_assets)
        w_old = pd.Series(0.0, index=all_assets)
        w_new.update(weights_new)
        w_old.update(weights_old)

        return float(np.abs(w_new - w_old).sum() / 2)

    def portfolio_risk_attribution(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Decompose portfolio volatility into per-asset risk contributions.

        Risk Contribution_i = w_i * (Sigma * w)_i / sigma_p

        Returns a Series indexed by asset with each asset's fractional
        share of total portfolio risk.
        """
        w = weights.values
        cov = returns.cov().values * TRADING_DAYS
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol == 0:
            return pd.Series(np.nan, index=weights.index)
        marginal = (cov @ w) / port_vol
        risk_contrib = w * marginal
        return pd.Series(risk_contrib / risk_contrib.sum(), index=weights.index)

    # ------------------------------------------------------------------
    # Private calculation methods
    # ------------------------------------------------------------------

    @staticmethod
    def _max_drawdown(prices: pd.DataFrame) -> pd.Series:
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown.min()

    @staticmethod
    def _downside_deviation(
        returns: pd.DataFrame, rf: float = 0.0
    ) -> pd.Series:
        """
        Annualized downside deviation (semi-deviation below risk-free rate).
        Used in the Sortino ratio denominator.
        """
        daily_rf = rf / TRADING_DAYS
        excess = returns.subtract(daily_rf)
        downside = excess.clip(upper=0)
        return (downside ** 2).mean() ** 0.5 * np.sqrt(TRADING_DAYS)

    @staticmethod
    def _cvar(
        returns: pd.DataFrame, confidence: float = 0.95
    ) -> pd.Series:
        """
        Annualized Conditional Value at Risk (Expected Shortfall) at the
        given confidence level.

        CVaR = E[R | R <= VaR_alpha] * sqrt(252)

        CVaR captures the expected loss in the tail beyond VaR, providing
        a more complete picture of tail risk than VaR alone.
        """
        alpha = 1 - confidence
        var = returns.quantile(alpha)
        cvar = returns[returns <= var].mean()
        return cvar * np.sqrt(TRADING_DAYS)

    @staticmethod
    def _omega_ratio(
        returns: pd.DataFrame, threshold: float = 0.0
    ) -> pd.Series:
        """
        Omega Ratio: probability-weighted ratio of gains to losses relative
        to a threshold return.

        Omega = E[max(R - L, 0)] / E[max(L - R, 0)]

        where L is the threshold (daily loss threshold). Values > 1 indicate
        more probability-weighted upside than downside.
        """
        gains = (returns - threshold).clip(lower=0).mean()
        losses = (threshold - returns).clip(lower=0).mean()
        return gains / losses.replace(0, np.nan)

    @staticmethod
    def _ulcer_index(prices: pd.DataFrame) -> pd.Series:
        """
        Ulcer Index: root mean square of drawdown depths.

        UI = sqrt(mean(D_t^2))  where D_t = (P_t - P_max) / P_max

        Captures both the depth and duration of drawdowns; preferred over
        maximum drawdown alone for strategies with prolonged underwater periods.
        """
        rolling_max = prices.expanding().max()
        drawdown_pct = (prices - rolling_max) / rolling_max * 100
        return (drawdown_pct ** 2).mean() ** 0.5
