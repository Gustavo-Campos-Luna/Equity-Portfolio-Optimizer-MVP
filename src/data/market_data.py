"""
Market data acquisition and cleaning layer.

All external data dependencies are isolated here. Swapping the data provider
(e.g., Bloomberg, Refinitiv) requires changes only in this module.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

from src.config.settings import PortfolioConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Downloads and sanitizes adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    config : PortfolioConfig
        Configuration object controlling download window and frequency.
    """

    def __init__(self, config: PortfolioConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_prices(
        self,
        tickers: Optional[List[str]] = None,
        benchmark: Optional[str] = None,
        years: Optional[int] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download split- and dividend-adjusted closing prices.

        Parameters
        ----------
        tickers : list of str, optional
            Override config tickers.
        benchmark : str, optional
            Override config benchmark.
        years : int, optional
            Override config years.
        interval : str, optional
            Override config interval.

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame of adjusted closing prices.
            The benchmark column, if requested, is included alongside assets.
        """
        tickers = tickers or self.config.tickers
        benchmark = benchmark or self.config.benchmark
        years = years or self.config.years
        interval = interval or self.config.interval

        end = date.today()
        start = end - timedelta(days=365 * years)

        all_symbols = list(tickers)
        if benchmark and benchmark not in all_symbols:
            all_symbols.append(benchmark)

        logger.info(
            "Fetching %d symbols from %s to %s (interval=%s).",
            len(all_symbols), start, end, interval,
        )

        raw = yf.download(
            all_symbols,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        close = self._extract_close(raw, all_symbols)
        close = self._clean(close)

        loaded = list(close.columns)
        missing = [t for t in all_symbols if t not in loaded]
        if missing:
            logger.warning("Symbols not loaded: %s", missing)
        logger.info(
            "Loaded %d symbols. Date range: %s to %s (%d sessions).",
            len(loaded),
            close.index.min().date(),
            close.index.max().date(),
            len(close),
        )

        return close

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_close(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Extract the 'Close' panel from a multi-level or single-level DataFrame."""
        if "Close" in raw.columns:
            if isinstance(raw["Close"], pd.Series):
                return raw["Close"].to_frame(tickers[0])
            return raw["Close"].copy()
        raise ValueError(
            "Unexpected yfinance DataFrame structure: 'Close' column not found."
        )

    @staticmethod
    def _clean(prices: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicated dates, all-NaN rows, and all-NaN columns."""
        prices = prices.sort_index()
        prices = prices.loc[~prices.index.duplicated(keep="last")]
        prices = prices.dropna(how="all")
        prices = prices.dropna(axis=1, how="all")
        return prices
