"""
Visualization module for portfolio performance analysis.

All chart functions produce publication-quality figures with a consistent
professional style. Charts can be saved to disk, displayed interactively,
or both.

Usage
-----
    engine = ChartEngine(output_dir="output/charts")
    engine.cumulative_returns(port_ret, bench_ret, "Max Sharpe")
    engine.drawdown(port_ret, "Max Sharpe")
    engine.rolling_sharpe(port_ret, bench_ret, "Max Sharpe")
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter

from src.config.settings import PortfolioConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# --- Global style ---
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")

TRADING_DAYS = 252
_PALETTE = {
    "portfolio": "#1A5276",
    "benchmark": "#922B21",
    "accent": "#1E8449",
    "fill_neg": "#E74C3C",
    "fill_pos": "#27AE60",
}


def _pct_formatter(y: float, _: object) -> str:
    return f"{y:.0%}"


def _pct_formatter_1d(y: float, _: object) -> str:
    return f"{y:.1%}"


class ChartEngine:
    """
    Centralized chart factory for portfolio analysis visualizations.

    Parameters
    ----------
    config : PortfolioConfig
        Provides risk-free rate and rebalancing frequency for annotations.
    output_dir : str, optional
        Directory where charts are saved. If None, charts are only displayed.
    show : bool
        Whether to call plt.show() after each chart.
    """

    def __init__(
        self,
        config: PortfolioConfig = DEFAULT_CONFIG,
        output_dir: Optional[str] = None,
        show: bool = True,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.show = show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Priority 1: Core performance charts
    # ------------------------------------------------------------------

    def cumulative_returns(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        strategy_name: str,
    ) -> None:
        """
        Cumulative return chart: portfolio vs. benchmark.

        Displays growth of $1 invested from the start of the backtest,
        with a performance summary annotation.
        """
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1
        ).dropna()
        if aligned.empty:
            logger.warning("No aligned data; skipping cumulative returns chart.")
            return

        port_cum = (1 + aligned.iloc[:, 0]).cumprod()
        bench_cum = (1 + aligned.iloc[:, 1]).cumprod()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            port_cum.index, port_cum.values,
            linewidth=2.2, label=strategy_name, color=_PALETTE["portfolio"],
        )
        ax.plot(
            bench_cum.index, bench_cum.values,
            linewidth=1.8, label="S&P 500", color=_PALETTE["benchmark"],
            alpha=0.85,
        )

        ax.set_title(
            f"Cumulative Returns  |  {strategy_name} vs S&P 500",
            fontsize=14, fontweight="bold", pad=16,
        )
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Cumulative Return", fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y - 1:.0%}"))
        self._format_xaxis(ax)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, framealpha=0.9)

        port_total = port_cum.iloc[-1] - 1
        bench_total = bench_cum.iloc[-1] - 1
        summary = (
            f"{strategy_name}: {port_total:+.1%}\n"
            f"S&P 500: {bench_total:+.1%}\n"
            f"Excess: {port_total - bench_total:+.1%}"
        )
        ax.text(
            0.98, 0.04, summary,
            transform=ax.transAxes, fontsize=9,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="whitesmoke", alpha=0.9),
        )

        plt.tight_layout()
        self._save_or_show(fig, f"cumulative_returns_{self._slug(strategy_name)}.png")

    def drawdown(
        self,
        portfolio_returns: pd.Series,
        strategy_name: str,
    ) -> None:
        """
        Underwater (drawdown) chart.

        Visualizes peak-to-trough losses over time. The area fill and
        annotated maximum drawdown provide a quick read of worst-case risk.
        """
        if portfolio_returns.empty:
            logger.warning("Empty returns; skipping drawdown chart.")
            return

        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        dd = (cumulative - rolling_max) / rolling_max

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(dd.index, 0, dd.values, color=_PALETTE["fill_neg"], alpha=0.65)
        ax.plot(dd.index, dd.values, color="#C0392B", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.7)

        max_dd = float(dd.min())
        max_dd_date = dd.idxmin()

        ax.set_title(
            f"Drawdown Analysis  |  {strategy_name}",
            fontsize=14, fontweight="bold", pad=16,
        )
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Drawdown", fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(_pct_formatter_1d))
        self._format_xaxis(ax)
        ax.grid(True, alpha=0.3)

        annotation = (
            f"Maximum Drawdown: {max_dd:.2%}\n"
            f"Trough Date: {max_dd_date.strftime('%Y-%m-%d')}"
        )
        ax.text(
            0.02, 0.04, annotation,
            transform=ax.transAxes, fontsize=9,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FADBD8", alpha=0.9),
        )

        plt.tight_layout()
        self._save_or_show(fig, f"drawdown_{self._slug(strategy_name)}.png")

    def rolling_sharpe(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        strategy_name: str,
        window_months: int = 12,
    ) -> None:
        """
        Rolling Sharpe ratio chart.

        Assesses strategy consistency. A stable, positive Sharpe over time
        indicates robust risk-adjusted performance.
        """
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns], axis=1
        ).dropna()
        if aligned.empty:
            logger.warning("No data for rolling Sharpe chart.")
            return

        port = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]
        window = window_months * 21
        rf = self.config.risk_free_rate

        if len(port) < window:
            logger.warning(
                "Insufficient data (%d days) for %d-month rolling Sharpe.",
                len(port), window_months,
            )
            return

        def _rolling_sr(ret: pd.Series) -> pd.Series:
            excess = (ret.rolling(window).mean() * TRADING_DAYS) - rf
            vol = ret.rolling(window).std() * np.sqrt(TRADING_DAYS)
            return excess / vol.replace(0, np.nan)

        port_sr = _rolling_sr(port)
        bench_sr = _rolling_sr(bench)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            port_sr.index, port_sr.values,
            linewidth=2.2, label=strategy_name, color=_PALETTE["portfolio"],
        )
        ax.plot(
            bench_sr.index, bench_sr.values,
            linewidth=1.8, label="S&P 500", color=_PALETTE["benchmark"],
            alpha=0.85,
        )
        ax.axhline(0, color="red", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(1, color="green", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(ax.get_xlim()[1], 1.02, "SR = 1.0", fontsize=8, color="green", va="bottom")
        ax.text(ax.get_xlim()[1], 0.02, "SR = 0", fontsize=8, color="red", va="bottom")

        ax.set_title(
            f"{window_months}-Month Rolling Sharpe  |  {strategy_name} vs S&P 500",
            fontsize=14, fontweight="bold", pad=16,
        )
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Rolling Sharpe Ratio", fontsize=11)
        self._format_xaxis(ax)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, framealpha=0.9)

        avg_port = port_sr.mean()
        avg_bench = bench_sr.mean()
        summary = (
            f"{strategy_name} avg: {avg_port:.2f}\n"
            f"S&P 500 avg: {avg_bench:.2f}"
        )
        ax.text(
            0.98, 0.04, summary,
            transform=ax.transAxes, fontsize=9,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.9),
        )

        plt.tight_layout()
        self._save_or_show(fig, f"rolling_sharpe_{self._slug(strategy_name)}.png")

    # ------------------------------------------------------------------
    # Priority 2: Analytical charts
    # ------------------------------------------------------------------

    def portfolio_composition(
        self,
        weights: pd.Series,
        strategy_name: str,
        min_weight: float = 0.01,
    ) -> None:
        """
        Pie chart of current portfolio weights.

        Positions below min_weight are grouped into an 'Other' slice.
        """
        if weights.empty:
            logger.warning("Empty weights; skipping composition chart.")
            return

        significant = weights[weights >= min_weight].copy()
        other = weights[weights < min_weight].sum()
        if other > 0:
            significant["Other"] = other

        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.tab20(np.linspace(0, 1, len(significant)))
        wedges, texts, autotexts = ax.pie(
            significant.values,
            labels=significant.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            pctdistance=0.82,
        )
        for t in texts:
            t.set_fontsize(9)
        for at in autotexts:
            at.set_fontsize(8)
            at.set_fontweight("bold")

        ax.set_title(
            f"Portfolio Composition  |  {strategy_name}",
            fontsize=13, fontweight="bold", pad=18,
        )

        n_pos = int((weights > 1e-4).sum())
        max_w = weights.max()
        hhi = (weights ** 2).sum()  # Herfindahl-Hirschman Index
        stats = f"Positions: {n_pos}  |  Max: {max_w:.1%}  |  HHI: {hhi:.3f}"
        ax.text(
            0.5, -0.04, stats,
            transform=ax.transAxes, fontsize=9,
            ha="center", color="gray",
        )

        plt.tight_layout()
        self._save_or_show(fig, f"composition_{self._slug(strategy_name)}.png")

    def monthly_returns_heatmap(
        self,
        portfolio_returns: pd.Series,
        strategy_name: str,
    ) -> None:
        """
        Monthly returns heatmap (years x months).

        Highlights seasonal patterns and consistency of returns.
        """
        import calendar

        if len(portfolio_returns) < 60:
            logger.warning("Less than 60 days; skipping heatmap.")
            return

        monthly = portfolio_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly.index = pd.to_datetime(monthly.index)
        matrix = (
            monthly
            .groupby([monthly.index.year, monthly.index.month])
            .first()
            .unstack()
        )
        matrix.columns = [calendar.month_abbr[m] for m in matrix.columns]

        fig, ax = plt.subplots(figsize=(13, max(4, len(matrix) * 0.6 + 2)))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            center=0,
            vmin=-0.12,
            vmax=0.12,
            linewidths=0.4,
            ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Monthly Return"},
        )
        ax.set_title(
            f"Monthly Returns Heatmap  |  {strategy_name}",
            fontsize=13, fontweight="bold", pad=16,
        )
        ax.set_xlabel("Month", fontsize=10)
        ax.set_ylabel("Year", fontsize=10)
        ax.tick_params(axis="x", rotation=0)

        plt.tight_layout()
        self._save_or_show(fig, f"monthly_heatmap_{self._slug(strategy_name)}.png")

    def risk_return_scatter(
        self,
        backtest_results: Dict[str, Dict],
    ) -> None:
        """
        Risk-return scatter plot with Sharpe-colored markers.

        Compares all strategies and the benchmark in annualized
        (volatility, return) space.
        """
        if not backtest_results:
            logger.warning("No results; skipping risk-return scatter.")
            return

        names, returns, vols, sharpes = [], [], [], []
        for name, res in backtest_results.items():
            names.append(name)
            returns.append(res.get("Portfolio CAGR", np.nan))
            vols.append(res.get("Portfolio Annual Vol", np.nan))
            sharpes.append(res.get("Portfolio Sharpe", np.nan))

        first = next(iter(backtest_results.values()))
        names.append("S&P 500")
        returns.append(first.get("Benchmark CAGR", np.nan))
        vols.append(first.get("Benchmark Annual Vol", np.nan))
        sharpes.append(first.get("Benchmark Sharpe", np.nan))

        sharpes_arr = np.array(sharpes, dtype=float)
        max_sr = np.nanmax(sharpes_arr) or 1
        colors = plt.cm.RdYlGn(sharpes_arr / max_sr)

        fig, ax = plt.subplots(figsize=(10, 7))
        for i, (name, ret, vol, color) in enumerate(
            zip(names, returns, vols, colors)
        ):
            marker = "s" if name == "S&P 500" else "o"
            size = 120 if name == "S&P 500" else 160
            ax.scatter(vol, ret, c=[color], s=size, marker=marker,
                       edgecolors="black", linewidth=1.2, zorder=3)
            ax.annotate(
                name, (vol, ret),
                xytext=(6, 4), textcoords="offset points",
                fontsize=9, fontweight="bold",
            )

        ax.set_title(
            "Risk-Return Analysis  |  Annualized CAGR vs Volatility",
            fontsize=13, fontweight="bold", pad=16,
        )
        ax.set_xlabel("Annualized Volatility", fontsize=11)
        ax.set_ylabel("Annualized Return (CAGR)", fontsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(_pct_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(_pct_formatter))
        ax.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, max_sr))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.8, label="Sharpe Ratio")

        plt.tight_layout()
        self._save_or_show(fig, "risk_return_scatter.png")

    def turnover_analysis(
        self,
        strategy_returns: Dict[str, pd.Series],
        backtest_results: Dict[str, Dict],
    ) -> None:
        """
        Side-by-side bar chart: annual turnover and transaction cost impact.
        """
        if not strategy_returns or not backtest_results:
            logger.warning("Missing data; skipping turnover analysis.")
            return

        strategies = list(strategy_returns.keys())
        turnovers = [
            backtest_results[s].get("Annual Turnover", 0) for s in strategies
        ]
        gross_rets = [
            backtest_results[s].get("Portfolio CAGR", 0) for s in strategies
        ]
        drags = [
            backtest_results[s].get("Transaction Cost Drag", 0) for s in strategies
        ]
        net_rets = [g - d for g, d in zip(gross_rets, drags)]
        labels = [s.replace("_", " ").title() for s in strategies]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        colors = [_PALETTE["portfolio"], _PALETTE["benchmark"], _PALETTE["accent"]]

        bars = ax1.bar(labels, turnovers, color=colors, edgecolor="black", alpha=0.8)
        ax1.set_title("Annual Portfolio Turnover", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Annual Turnover", fontsize=10)
        ax1.yaxis.set_major_formatter(FuncFormatter(_pct_formatter))
        for bar, val in zip(bars, turnovers):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        ax1.grid(True, alpha=0.3, axis="y")

        x = np.arange(len(labels))
        w = 0.35
        b2 = ax2.bar(x - w / 2, gross_rets, w, label="Gross CAGR",
                     color=_PALETTE["accent"], edgecolor="black", alpha=0.85)
        b3 = ax2.bar(x + w / 2, net_rets, w, label="Net CAGR",
                     color=_PALETTE["fill_neg"], edgecolor="black", alpha=0.85)
        ax2.set_title("Transaction Cost Impact on Returns", fontsize=12, fontweight="bold")
        ax2.set_ylabel("CAGR", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.yaxis.set_major_formatter(FuncFormatter(_pct_formatter_1d))
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(b2, gross_rets):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.1%}", ha="center", va="bottom", fontsize=8,
            )
        for bar, val in zip(b3, net_rets):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.1%}", ha="center", va="bottom", fontsize=8,
            )

        plt.tight_layout()
        self._save_or_show(fig, "turnover_analysis.png")

    def wealth_projection_chart(
        self,
        projection_df: pd.DataFrame,
        initial_capital: float,
    ) -> None:
        """
        Multi-strategy terminal wealth projection across time horizons.

        Displays both nominal and real (inflation-adjusted) values.
        """
        if projection_df.empty:
            return

        strategies = projection_df.index.get_level_values("Strategy").unique()
        horizons = projection_df.index.get_level_values("Horizon (yr)").unique()
        colors = [_PALETTE["portfolio"], _PALETTE["benchmark"], _PALETTE["accent"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for i, strategy in enumerate(strategies):
            sub = projection_df.loc[strategy]
            color = colors[i % len(colors)]
            ax1.plot(
                sub.index, sub["Terminal Wealth (nominal)"] / 1e3,
                marker="o", linewidth=2, label=strategy, color=color,
            )
            ax2.plot(
                sub.index, sub["Terminal Wealth (real)"] / 1e3,
                marker="o", linewidth=2, label=strategy, color=color,
                linestyle="--",
            )

        for ax, title in [
            (ax1, "Nominal Terminal Wealth"),
            (ax2, "Real Terminal Wealth (Inflation-Adjusted)"),
        ]:
            ax.axhline(
                initial_capital / 1e3, color="gray",
                linestyle=":", linewidth=1.2, label="Initial Capital",
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Investment Horizon (years)", fontsize=10)
            ax.set_ylabel("Terminal Wealth ($ thousands)", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Wealth Projection  |  Initial Capital: ${initial_capital:,.0f}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        self._save_or_show(fig, "wealth_projection.png")

    # ------------------------------------------------------------------
    # Dashboard wrappers
    # ------------------------------------------------------------------

    def performance_dashboard(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        strategy_name: str,
    ) -> None:
        """Render cumulative returns, drawdown, and rolling Sharpe for one strategy."""
        self.cumulative_returns(portfolio_returns, benchmark_returns, strategy_name)
        self.drawdown(portfolio_returns, strategy_name)
        self.rolling_sharpe(portfolio_returns, benchmark_returns, strategy_name)

    def comprehensive_dashboard(
        self,
        backtest_results: Dict[str, Dict],
        strategy_returns: Dict[str, pd.Series],
        current_weights: Dict[str, pd.Series],
        projection_df: Optional[pd.DataFrame] = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        """
        Render the full suite of analytical charts.

        Includes portfolio compositions, monthly heatmaps, risk-return
        scatter, turnover analysis, and (optionally) a wealth projection.
        """
        for strategy, weights in current_weights.items():
            if not weights.empty:
                self.portfolio_composition(
                    weights, strategy.replace("_", " ").title()
                )

        for strategy, returns in strategy_returns.items():
            if not returns.empty:
                self.monthly_returns_heatmap(
                    returns, strategy.replace("_", " ").title()
                )

        self.risk_return_scatter(backtest_results)
        self.turnover_analysis(strategy_returns, backtest_results)

        if projection_df is not None and not projection_df.empty:
            self.wealth_projection_chart(projection_df, initial_capital)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_xaxis(ax: plt.Axes) -> None:
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    def _save_or_show(self, fig: plt.Figure, filename: str) -> None:
        if self.output_dir:
            path = os.path.join(self.output_dir, filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Chart saved: %s", path)
        if self.show:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _slug(name: str) -> str:
        return name.lower().replace(" ", "_").replace("/", "_")
