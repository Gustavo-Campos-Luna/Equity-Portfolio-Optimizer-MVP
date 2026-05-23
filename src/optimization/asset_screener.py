"""
Asset screening module.

Reduces the investable universe to a curated subset of high-quality assets
prior to portfolio optimization. Multiple scoring methodologies are provided
to accommodate different investment mandates.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.config.settings import PortfolioConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalization to [0, 1]. Returns 0.5 for degenerate inputs."""
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    span = s.max() - s.min()
    if span == 0 or s.std() == 0:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / span


class AssetScreener:
    """
    Multi-factor asset screener.

    Constructs a composite score from normalized risk and return metrics,
    then selects the top-N assets for portfolio construction.

    Available methods
    -----------------
    enhanced_composite
        0.60 * Sharpe + 0.15 * (1 - Volatility) + 0.15 * Momentum(12M)
        + 0.10 * (1 - CVaR). Balanced risk-adjusted quality filter.
    sharpe_only
        1.00 * Sharpe. Pure risk-adjusted return ranking.
    momentum_tilt
        0.50 * Sharpe + 0.35 * Momentum(12M) + 0.15 * (1 - CVaR).
        Suitable for trend-following mandates.
    low_risk
        0.50 * (1 - Volatility) + 0.30 * (1 - MaxDrawdown)
        + 0.20 * (1 - CVaR). Suitable for capital-preservation mandates.
    quality_factor
        0.40 * Sharpe + 0.30 * Sortino + 0.30 * Calmar.
        Multi-ratio quality filter emphasizing downside-adjusted metrics.

    Parameters
    ----------
    config : PortfolioConfig
        Configuration providing the top_n default.
    """

    _METHODS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}

    def __init__(self, config: PortfolioConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._register_methods()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def screen(
        self,
        metrics: pd.DataFrame,
        top_n: Optional[int] = None,
        method: str = "enhanced_composite",
    ) -> pd.DataFrame:
        """
        Score and rank assets, returning the top-N by composite score.

        Parameters
        ----------
        metrics : pd.DataFrame
            Output of MetricsCalculator.compute_asset_metrics().
        top_n : int, optional
            Number of assets to return. Defaults to config.top_n.
        method : str
            Scoring methodology. See class docstring for options.

        Returns
        -------
        pd.DataFrame
            Filtered metrics DataFrame with an additional 'Score' column,
            sorted by Score descending.
        """
        top_n = top_n or self.config.top_n

        if method not in self._METHODS:
            raise ValueError(
                f"Unknown screening method '{method}'. "
                f"Valid options: {list(self._METHODS.keys())}."
            )

        m = metrics.copy()
        score_fn = self._METHODS[method]
        m["Score"] = score_fn(m)
        m = m.sort_values("Score", ascending=False).head(top_n)

        logger.info(
            "AssetScreener [%s]: selected %d of %d assets.",
            method, len(m), len(metrics),
        )
        return m

    def available_methods(self) -> list:
        """Return list of registered screening method names."""
        return list(self._METHODS.keys())

    # ------------------------------------------------------------------
    # Method registry
    # ------------------------------------------------------------------

    def _register_methods(self) -> None:
        def enhanced_composite(m: pd.DataFrame) -> pd.Series:
            sharpe_n = _normalize(m["Sharpe"])
            vol_n = _normalize(m["AnnualizedVol"])
            mom_n = _normalize(m.get("Momentum12M", pd.Series(0.5, index=m.index)))
            cvar_n = _normalize(m.get("CVaR_95", pd.Series(0.0, index=m.index)).abs())
            return (
                0.60 * sharpe_n
                + 0.15 * (1 - vol_n)
                + 0.15 * mom_n
                + 0.10 * (1 - cvar_n)
            )

        def sharpe_only(m: pd.DataFrame) -> pd.Series:
            return _normalize(m["Sharpe"])

        def momentum_tilt(m: pd.DataFrame) -> pd.Series:
            sharpe_n = _normalize(m["Sharpe"])
            mom_n = _normalize(m.get("Momentum12M", pd.Series(0.5, index=m.index)))
            cvar_n = _normalize(m.get("CVaR_95", pd.Series(0.0, index=m.index)).abs())
            return 0.50 * sharpe_n + 0.35 * mom_n + 0.15 * (1 - cvar_n)

        def low_risk(m: pd.DataFrame) -> pd.Series:
            vol_n = _normalize(m["AnnualizedVol"])
            dd_n = _normalize(m["MaxDrawdown"].abs())
            cvar_n = _normalize(m.get("CVaR_95", pd.Series(0.0, index=m.index)).abs())
            return 0.50 * (1 - vol_n) + 0.30 * (1 - dd_n) + 0.20 * (1 - cvar_n)

        def quality_factor(m: pd.DataFrame) -> pd.Series:
            sharpe_n = _normalize(m["Sharpe"])
            sortino_n = _normalize(m.get("Sortino", m["Sharpe"]))
            calmar_n = _normalize(m.get("Calmar", m["Sharpe"]))
            return 0.40 * sharpe_n + 0.30 * sortino_n + 0.30 * calmar_n

        self._METHODS = {
            "enhanced_composite": enhanced_composite,
            "sharpe_only": sharpe_only,
            "momentum_tilt": momentum_tilt,
            "low_risk": low_risk,
            "quality_factor": quality_factor,
        }
