"""Tests for MetricsCalculator, focused on edge cases: NaN, short series,
zero variance, and degenerate inputs that could cause silent division by zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import PortfolioConfig
from src.metrics.financial_metrics import MetricsCalculator
from tests.helpers import make_prices


@pytest.fixture
def calc() -> MetricsCalculator:
    return MetricsCalculator(PortfolioConfig(tickers=["A0", "A1", "A2"]))


class TestQualityFilter:
    def test_drops_assets_below_coverage_threshold(self, calc: MetricsCalculator) -> None:
        prices = make_prices(3, 300, seed=1)
        # A1 has 50% missing data; below the 80% default min_coverage.
        prices.loc[prices.index[::2], "A1"] = np.nan

        filtered = calc.quality_filter(prices, verbose=False)

        assert "A1" not in filtered.columns
        assert "A0" in filtered.columns
        assert "A2" in filtered.columns

    def test_relaxes_threshold_when_fewer_than_three_survive(self, calc: MetricsCalculator) -> None:
        prices = make_prices(3, 300, seed=2)
        # Knock out two of three assets entirely -> triggers the relaxed path.
        prices["A1"] = np.nan
        prices["A2"] = np.nan

        filtered = calc.quality_filter(prices, verbose=False)

        # A0 alone still passes; the relaxed branch must not crash on <3 survivors.
        assert "A0" in filtered.columns

    def test_full_coverage_keeps_all_assets(self, calc: MetricsCalculator) -> None:
        prices = make_prices(3, 300, seed=3)
        filtered = calc.quality_filter(prices, verbose=False)
        assert set(filtered.columns) == {"A0", "A1", "A2"}


class TestComputeAssetMetrics:
    def test_short_series_falls_back_to_zero_momentum(self, calc: MetricsCalculator) -> None:
        """With fewer than 126/252 sessions, momentum must default to 0.0
        rather than raise an IndexError on df.iloc[-126]."""
        prices = make_prices(2, 60, seed=4)
        metrics = calc.compute_asset_metrics(prices)

        assert (metrics["Momentum6M"] == 0.0).all()
        assert (metrics["Momentum12M"] == 0.0).all()

    def test_zero_volatility_asset_is_dropped_not_crashed(self, calc: MetricsCalculator) -> None:
        """A flat (zero-return) asset produces AnnualizedVol == 0, which would
        make Sharpe = x/0 = inf. The method must not raise, and must drop the
        resulting inf/NaN row instead of returning garbage."""
        prices = make_prices(2, 300, seed=5)
        prices["FLAT"] = 100.0  # constant price -> zero variance

        metrics = calc.compute_asset_metrics(prices)

        assert "FLAT" not in metrics.index
        assert np.isfinite(metrics["Sharpe"]).all()

    def test_cagr_matches_known_growth_rate(self, calc: MetricsCalculator) -> None:
        """Deterministic 1-asset case: price doubles over exactly 1 year."""
        index = pd.bdate_range("2020-01-01", periods=253)
        prices = pd.DataFrame({"A0": np.linspace(100, 200, len(index))}, index=index)

        metrics = calc.compute_asset_metrics(prices)

        years = (index[-1] - index[0]).days / 365.25
        expected_cagr = (200 / 100) ** (1 / years) - 1
        assert metrics.loc["A0", "CAGR"] == pytest.approx(expected_cagr, rel=1e-9)

    def test_benchmark_column_excluded(self, calc: MetricsCalculator) -> None:
        prices = make_prices(2, 300, seed=6, columns=["A0", "A1"])
        prices["^GSPC"] = make_prices(1, 300, seed=7, columns=["B"])["B"].values

        metrics = calc.compute_asset_metrics(prices, benchmark="^GSPC")

        assert "^GSPC" not in metrics.index


class TestTrackingErrorAndIR:
    def test_insufficient_observations_returns_nan(self, calc: MetricsCalculator) -> None:
        port = pd.Series(np.random.default_rng(8).normal(0, 0.01, 10))
        bench = pd.Series(np.random.default_rng(9).normal(0, 0.01, 10))

        te, ir = calc.tracking_error_and_ir(port, bench)

        assert np.isnan(te)
        assert np.isnan(ir)

    def test_identical_series_has_zero_tracking_error_and_nan_ir(
        self, calc: MetricsCalculator
    ) -> None:
        """Zero tracking error means zero excess-return dispersion, which would
        divide-by-zero in a naive IR calculation; the method must return NaN
        instead of raising or returning inf."""
        returns = pd.Series(np.random.default_rng(10).normal(0, 0.01, 60))

        te, ir = calc.tracking_error_and_ir(returns, returns.copy())

        assert te == pytest.approx(0.0, abs=1e-12)
        assert np.isnan(ir)

    def test_normal_case_returns_finite_values(self, calc: MetricsCalculator) -> None:
        rng = np.random.default_rng(11)
        port = pd.Series(rng.normal(0.0006, 0.01, 100))
        bench = pd.Series(rng.normal(0.0004, 0.01, 100))

        te, ir = calc.tracking_error_and_ir(port, bench)

        assert te > 0
        assert np.isfinite(ir)


class TestCalculateTurnover:
    def test_empty_previous_weights_is_full_turnover(self, calc: MetricsCalculator) -> None:
        new = pd.Series({"A0": 0.5, "A1": 0.5})
        old = pd.Series(dtype=float)
        assert calc.calculate_turnover(new, old) == 1.0

    def test_identical_weights_is_zero_turnover(self, calc: MetricsCalculator) -> None:
        w = pd.Series({"A0": 0.6, "A1": 0.4})
        assert calc.calculate_turnover(w, w.copy()) == pytest.approx(0.0)

    def test_completely_disjoint_holdings_is_full_turnover(self, calc: MetricsCalculator) -> None:
        old = pd.Series({"A0": 1.0})
        new = pd.Series({"A1": 1.0})
        assert calc.calculate_turnover(new, old) == pytest.approx(1.0)

    def test_partial_rebalance_is_between_zero_and_one(self, calc: MetricsCalculator) -> None:
        old = pd.Series({"A0": 0.5, "A1": 0.5})
        new = pd.Series({"A0": 0.7, "A1": 0.3})
        turnover = calc.calculate_turnover(new, old)
        assert 0.0 < turnover < 1.0
        assert turnover == pytest.approx(0.2)


class TestPortfolioRiskAttribution:
    def test_zero_volatility_portfolio_returns_nan_not_crash(self, calc: MetricsCalculator) -> None:
        """All-zero weights make port_vol == 0; must return NaN series
        instead of dividing by zero."""
        weights = pd.Series({"A0": 0.0, "A1": 0.0})
        returns = make_prices(2, 100, seed=12).pct_change().dropna()
        returns.columns = ["A0", "A1"]

        result = calc.portfolio_risk_attribution(weights, returns)

        assert result.isna().all()

    def test_risk_contributions_sum_to_one(self, calc: MetricsCalculator) -> None:
        prices = make_prices(3, 300, seed=13, columns=["A0", "A1", "A2"])
        returns = prices.pct_change().dropna()
        weights = pd.Series({"A0": 0.5, "A1": 0.3, "A2": 0.2})

        result = calc.portfolio_risk_attribution(weights, returns)

        assert result.sum() == pytest.approx(1.0, rel=1e-6)


class TestOmegaRatio:
    def test_no_downside_returns_nan_not_zero_division_error(self, calc: MetricsCalculator) -> None:
        """When every daily return is above the threshold, the loss
        denominator is 0; _omega_ratio must return NaN, not raise."""
        returns = pd.DataFrame({"A0": [0.01, 0.02, 0.015, 0.03]})
        omega = MetricsCalculator._omega_ratio(returns, threshold=0.0)
        assert np.isnan(omega["A0"])
