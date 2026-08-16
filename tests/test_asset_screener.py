"""Tests for AssetScreener, focused on normalization edge cases (constant
series, degenerate spans) and screening contract (ranking, unknown methods).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import PortfolioConfig
from src.optimization.asset_screener import AssetScreener, _normalize


@pytest.fixture
def screener() -> AssetScreener:
    return AssetScreener(PortfolioConfig(tickers=["A", "B", "C"], top_n=2))


def _metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sharpe": [2.0, 1.0, 0.0],
            "Sortino": [2.5, 1.2, 0.1],
            "Calmar": [1.5, 0.8, 0.05],
            "AnnualizedVol": [0.15, 0.25, 0.40],
            "MaxDrawdown": [-0.10, -0.20, -0.35],
            "Momentum12M": [0.30, 0.10, -0.10],
            "CVaR_95": [-0.20, -0.30, -0.45],
        },
        index=["HIGH", "MID", "LOW"],
    )


class TestNormalize:
    def test_constant_series_returns_half_not_nan_or_inf(self) -> None:
        """Zero span (all values identical) must not divide by zero."""
        s = pd.Series([3.0, 3.0, 3.0], index=["a", "b", "c"])
        result = _normalize(s)
        assert (result == 0.5).all()

    def test_single_value_series_returns_half(self) -> None:
        s = pd.Series([5.0], index=["a"])
        result = _normalize(s)
        assert result.iloc[0] == 0.5

    def test_normal_series_maps_to_unit_range(self) -> None:
        s = pd.Series([0.0, 5.0, 10.0], index=["a", "b", "c"])
        result = _normalize(s)
        assert result["a"] == pytest.approx(0.0)
        assert result["c"] == pytest.approx(1.0)
        assert result["b"] == pytest.approx(0.5)

    def test_inf_values_are_excluded_not_propagated(self) -> None:
        s = pd.Series([1.0, np.inf, 3.0], index=["a", "b", "c"])
        result = _normalize(s)
        assert "b" not in result.index
        assert np.isfinite(result).all()


class TestScreen:
    def test_unknown_method_raises_value_error(self, screener: AssetScreener) -> None:
        with pytest.raises(ValueError, match="Unknown screening method"):
            screener.screen(_metrics_frame(), method="not_a_real_method")

    def test_enhanced_composite_ranks_best_quality_asset_first(
        self, screener: AssetScreener
    ) -> None:
        result = screener.screen(_metrics_frame(), method="enhanced_composite", top_n=3)
        # HIGH dominates on every factor (best Sharpe, lowest vol, best
        # momentum, lowest CVaR), so it must rank first regardless of weights.
        assert result.index[0] == "HIGH"
        assert result.index[-1] == "LOW"

    def test_top_n_limits_result_size(self, screener: AssetScreener) -> None:
        result = screener.screen(_metrics_frame(), method="sharpe_only", top_n=2)
        assert len(result) == 2

    def test_top_n_larger_than_universe_returns_all_without_crash(
        self, screener: AssetScreener
    ) -> None:
        result = screener.screen(_metrics_frame(), method="sharpe_only", top_n=100)
        assert len(result) == 3

    def test_all_five_methods_run_without_error(self, screener: AssetScreener) -> None:
        for method in screener.available_methods():
            result = screener.screen(_metrics_frame(), method=method)
            assert "Score" in result.columns
            assert np.isfinite(result["Score"]).all()
