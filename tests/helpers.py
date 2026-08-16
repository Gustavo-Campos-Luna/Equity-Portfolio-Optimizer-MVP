"""Synthetic price data generators shared across test modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_prices(
    n_assets: int,
    n_days: int,
    seed: int = 0,
    daily_drift: float = 0.0004,
    daily_vol: float = 0.01,
    start_price: float = 100.0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a synthetic geometric-random-walk price DataFrame.

    Deterministic given the seed, so tests using it are reproducible.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(daily_drift, daily_vol, size=(n_days, n_assets))
    prices = start_price * np.cumprod(1 + returns, axis=0)
    index = pd.bdate_range("2020-01-01", periods=n_days)
    cols = columns or [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=index, columns=cols)
