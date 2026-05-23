"""
Configuration module for the equity portfolio optimization engine.

All rate parameters are expressed as annual decimals (e.g., 0.07 = 7%).
Modify this file to customize the asset universe, optimization constraints,
and backtesting parameters before running main.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PortfolioConfig:
    """
    Central configuration class for portfolio optimization and backtesting.

    Parameters
    ----------
    tickers : List[str]
        Equity universe. All tickers must be valid Yahoo Finance symbols.
    benchmark : str
        Benchmark ticker for performance attribution (default: S&P 500).
    years : int
        Historical window (in years) to download from Yahoo Finance.
    interval : str
        Data frequency. Options: '1d' (daily), '1wk', '1mo'.
    min_coverage : float
        Minimum fraction of non-null price observations required per asset
        to pass the data quality filter.
    min_sessions : int
        Absolute minimum number of trading sessions required within any
        rolling backtest window.
    top_n : int
        Number of assets to select after the composite screening step.
    weight_cap : float
        Maximum weight constraint per individual asset in optimized portfolio.
    risk_free_rate : float
        Annual risk-free rate used in Sharpe, Sortino, and related ratios.
    min_positions : int
        Minimum number of active (non-zero) positions enforced during
        optimization. Prevents excessive concentration.
    window_years : int
        In-sample training window (years) for the rolling backtest.
    rebalance_frequency : str
        Pandas offset alias for rebalancing cadence. Options: 'M' (monthly),
        'Q' (quarterly), 'A' (annual).
    transaction_cost : float
        One-way transaction cost applied per rebalancing event (e.g., 0.0015
        = 15 basis points).
    min_warmup_sessions : int
        Minimum number of sessions required before the first rebalancing
        period is allowed to begin.
    """

    tickers: List[str] = field(default_factory=list)
    benchmark: str = "^GSPC"

    # --- Data ---
    years: int = 5
    interval: str = "1d"
    min_coverage: float = 0.80
    min_sessions: int = 400

    # --- Optimization ---
    top_n: int = 15
    weight_cap: float = 0.125
    risk_free_rate: float = 0.02
    min_positions: int = 8

    # --- Backtesting ---
    window_years: int = 2
    rebalance_frequency: str = "M"
    transaction_cost: float = 0.0015
    min_warmup_sessions: int = 60

    def __post_init__(self) -> None:
        if not self.tickers:
            self.tickers = [
                "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
                "JPM", "XOM", "CVX", "UNH", "JNJ", "PEP", "KO",
                "PG", "HD", "BAC", "WMT", "DIS", "CRM", "NFLX",
                "V", "MA", "TSM", "ABBV", "TMO",
            ]

    def validate(self) -> None:
        """Raise ValueError if any constraint is internally inconsistent."""
        if self.weight_cap * self.min_positions > 1.0 + 1e-6:
            raise ValueError(
                f"weight_cap ({self.weight_cap:.1%}) x min_positions "
                f"({self.min_positions}) exceeds 100%. Reduce weight_cap "
                "or min_positions."
            )
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1].")
        if self.rebalance_frequency not in ("M", "Q", "A", "W"):
            raise ValueError(
                f"Unsupported rebalance_frequency: '{self.rebalance_frequency}'. "
                "Use 'M', 'Q', 'A', or 'W'."
            )


# Module-level default instance — import this in other modules.
DEFAULT_CONFIG = PortfolioConfig()
