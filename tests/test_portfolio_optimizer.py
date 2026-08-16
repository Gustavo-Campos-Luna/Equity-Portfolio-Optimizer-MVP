"""Tests for PortfolioOptimizer, focused on constraint satisfaction and
degenerate inputs (single asset, all-negative excess returns, zero variance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import PortfolioConfig
from src.metrics.financial_metrics import MetricsCalculator
from src.optimization.portfolio_optimizer import PortfolioOptimizer
from tests.helpers import make_prices


@pytest.fixture
def config() -> PortfolioConfig:
    return PortfolioConfig(
        tickers=["A0", "A1", "A2", "A3", "A4"],
        weight_cap=0.40,
        min_positions=3,
        risk_free_rate=0.02,
    )


@pytest.fixture
def optimizer(config: PortfolioConfig) -> PortfolioOptimizer:
    return PortfolioOptimizer(config)


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices(5, 300, seed=20, columns=["A0", "A1", "A2", "A3", "A4"])


class TestOptimizeMaxSharpe:
    def test_weights_sum_to_one_and_respect_cap(
        self, optimizer: PortfolioOptimizer, prices: pd.DataFrame, config: PortfolioConfig
    ) -> None:
        picks = list(prices.columns)
        weights, perf = optimizer.optimize_max_sharpe(prices, picks)

        assert weights.sum() == pytest.approx(1.0, abs=1e-3)
        assert (weights <= config.weight_cap + 1e-6).all()
        assert (weights >= 0).all()

    def test_performance_tuple_is_finite(
        self, optimizer: PortfolioOptimizer, prices: pd.DataFrame
    ) -> None:
        picks = list(prices.columns)
        _, (ret, vol, sharpe) = optimizer.optimize_max_sharpe(prices, picks)
        assert np.isfinite(ret)
        assert np.isfinite(vol)
        assert np.isfinite(sharpe)


class TestOptimizeMinVariance:
    def test_weights_sum_to_one(self, optimizer: PortfolioOptimizer, prices: pd.DataFrame) -> None:
        picks = list(prices.columns)
        weights, _ = optimizer.optimize_min_variance(prices, picks)
        assert weights.sum() == pytest.approx(1.0, abs=1e-3)

    def test_variance_not_worse_than_naive_equal_weight(
        self, optimizer: PortfolioOptimizer, prices: pd.DataFrame
    ) -> None:
        """The GMVP solution must never have higher variance than the
        equal-weight portfolio, since equal-weight is itself a feasible
        point under the same (dynamically capped) bounds. Compared against
        the same winsorized covariance the optimizer solves against."""
        picks = list(prices.columns)
        weights, (_, opt_vol, _) = optimizer.optimize_min_variance(prices, picks)

        returns = MetricsCalculator.clip_outliers(prices[picks].pct_change().dropna())
        cov = returns.cov().values * 252
        n = len(picks)
        equal_w = np.full(n, 1.0 / n)
        naive_vol = float(np.sqrt(equal_w @ cov @ equal_w))

        assert opt_vol <= naive_vol + 1e-6


class TestOptimizeRiskParity:
    def test_weights_sum_to_one_within_bounds(
        self, optimizer: PortfolioOptimizer, prices: pd.DataFrame
    ) -> None:
        picks = list(prices.columns)
        weights, _ = optimizer.optimize_risk_parity(prices, picks)

        assert weights.reindex(picks).fillna(0).sum() == pytest.approx(1.0, abs=1e-3)
        # Risk parity uses its own 1%-40% bounds, independent of config.weight_cap.
        assert (weights <= 0.40 + 1e-6).all()

    def test_falls_back_to_equal_weight_when_infeasible(
        self, optimizer: PortfolioOptimizer
    ) -> None:
        """A single asset can't satisfy sum(w)=1 under 1%-40% bounds; the
        optimizer must fall back to x0 (equal weight) rather than raise."""
        prices = make_prices(1, 300, seed=21, columns=["A0"])
        weights, perf = optimizer.optimize_risk_parity(prices, ["A0"])

        assert not weights.empty
        assert np.isfinite(perf[0])


class TestKellyWeights:
    def test_all_negative_excess_returns_falls_back_to_equal_weight(
        self, optimizer: PortfolioOptimizer
    ) -> None:
        """When every asset's Kelly fraction clips to zero (all expected
        returns below the risk-free rate), total <= 0 and the method must
        fall back to equal weighting instead of dividing by zero."""
        index = pd.bdate_range("2020-01-01", periods=300)
        # Strictly declining prices -> negative expected return for every asset.
        prices = pd.DataFrame(
            {
                "A0": np.linspace(100, 50, len(index)),
                "A1": np.linspace(100, 60, len(index)),
            },
            index=index,
        )

        weights = optimizer.kelly_weights(prices, ["A0", "A1"], fractional=0.5)

        assert weights.sum() == pytest.approx(1.0)
        assert weights.to_numpy() == pytest.approx([0.5, 0.5])

    def test_zero_variance_asset_does_not_raise_division_by_zero(
        self, optimizer: PortfolioOptimizer
    ) -> None:
        index = pd.bdate_range("2020-01-01", periods=300)
        prices = pd.DataFrame(
            {
                "FLAT": 100.0,
                "A1": np.linspace(100, 130, len(index)),
            },
            index=index,
        )

        weights = optimizer.kelly_weights(prices, ["FLAT", "A1"], fractional=0.5)

        assert np.isfinite(weights).all()
        assert weights["FLAT"] == pytest.approx(0.0)

    def test_weights_sum_to_one_in_normal_case(
        self, optimizer: PortfolioOptimizer, prices: pd.DataFrame
    ) -> None:
        picks = list(prices.columns)
        weights = optimizer.kelly_weights(prices, picks, fractional=0.5)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)


class TestDynamicCap:
    def test_single_asset_universe_does_not_raise(self, optimizer: PortfolioOptimizer) -> None:
        cap = optimizer._dynamic_cap(1)
        assert cap > 0
        assert cap <= 1.0 + 1e-9

    def test_cap_times_n_assets_is_feasible(self, optimizer: PortfolioOptimizer) -> None:
        for n in [1, 2, 5, 15, 30]:
            cap = optimizer._dynamic_cap(n)
            assert cap * n >= 1.0 - 1e-6
