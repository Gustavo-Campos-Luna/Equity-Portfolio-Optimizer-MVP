"""
Performance reporting and analytical commentary module.

Produces structured performance summaries, strategy comparisons,
and evidence-based conclusions suitable for institutional presentations
and investment research documentation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.settings import PortfolioConfig, DEFAULT_CONFIG
from src.metrics.financial_metrics import MetricsCalculator

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
_FREQ_PERIODS: Dict[str, int] = {"ME": 12, "QE": 4, "YE": 1, "W": 52}


class PerformanceReport:
    """
    Generates comprehensive performance analytics and written commentary
    for portfolio strategies vs. a benchmark.

    Parameters
    ----------
    config : PortfolioConfig
        Provides risk-free rate and rebalancing frequency.
    """

    def __init__(self, config: PortfolioConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._calc = MetricsCalculator(config)

    # ------------------------------------------------------------------
    # Core analytics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        turnover_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Compute the complete set of portfolio performance statistics.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily net-of-cost portfolio returns.
        benchmark_returns : pd.Series
            Daily benchmark returns aligned to the same index.
        turnover_df : pd.DataFrame, optional
            Output of BacktestEngine.run() containing per-period turnover.

        Returns
        -------
        dict
            Flat dictionary of performance metrics.
        """
        rf = self.config.risk_free_rate
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if aligned.empty:
            return {}

        port = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]
        years = len(port) / TRADING_DAYS

        # --- Return metrics ---
        port_cum = (1 + port).cumprod().iloc[-1] - 1
        bench_cum = (1 + bench).cumprod().iloc[-1] - 1
        port_cagr = (1 + port_cum) ** (1 / years) - 1
        bench_cagr = (1 + bench_cum) ** (1 / years) - 1
        port_ann = port.mean() * TRADING_DAYS
        bench_ann = bench.mean() * TRADING_DAYS

        # --- Risk metrics ---
        port_vol = port.std() * np.sqrt(TRADING_DAYS)
        bench_vol = bench.std() * np.sqrt(TRADING_DAYS)

        # Downside metrics
        dd_port = self._max_drawdown_series(port)
        max_dd = float(dd_port.min())
        recovery_days = self._recovery_time(dd_port)

        downside_dev = float(
            np.sqrt(((port.clip(upper=0)) ** 2).mean()) * np.sqrt(TRADING_DAYS)
        )
        ulcer = float(
            np.sqrt(
                (((1 + port).cumprod() / (1 + port).cumprod().expanding().max() - 1) ** 2).mean()
            ) * 100
        )

        # VaR and CVaR
        var_95 = float(port.quantile(0.05) * np.sqrt(TRADING_DAYS))
        cvar_95 = float(
            port[port <= port.quantile(0.05)].mean() * np.sqrt(TRADING_DAYS)
        )

        # --- Risk-adjusted ratios ---
        port_sharpe = (port_ann - rf) / port_vol if port_vol > 0 else np.nan
        bench_sharpe = (bench_ann - rf) / bench_vol if bench_vol > 0 else np.nan
        sortino = (port_ann - rf) / downside_dev if downside_dev > 0 else np.nan
        calmar = port_cagr / abs(max_dd) if max_dd != 0 else np.nan
        omega = float(
            port.clip(lower=0).mean() / (-port.clip(upper=0)).mean()
            if port.clip(upper=0).mean() != 0
            else np.nan
        )

        # --- Benchmark-relative metrics ---
        te, ir = self._calc.tracking_error_and_ir(port, bench)
        excess_ann = port_ann - bench_ann
        hit_rate = float((port > bench).mean())
        beta = (
            float(np.cov(port, bench)[0, 1] / np.var(bench))
            if np.var(bench) > 0
            else np.nan
        )
        alpha = port_ann - (rf + beta * (bench_ann - rf)) if not np.isnan(beta) else np.nan

        # --- Transaction analysis ---
        k = _FREQ_PERIODS.get(self.config.rebalance_frequency, 12)
        avg_turnover = (
            float(turnover_df["turnover"].mean())
            if turnover_df is not None and not turnover_df.empty
            else np.nan
        )
        annual_turnover = avg_turnover * k if not np.isnan(avg_turnover) else np.nan
        cost_drag = (
            annual_turnover * self.config.transaction_cost
            if not np.isnan(annual_turnover)
            else np.nan
        )

        # --- Distribution ---
        skewness = float(port.skew())
        excess_kurtosis = float(port.kurtosis())

        return {
            # Return
            "Portfolio Total Return": port_cum,
            "Benchmark Total Return": bench_cum,
            "Portfolio CAGR": port_cagr,
            "Benchmark CAGR": bench_cagr,
            "Portfolio Annual Return": port_ann,
            "Benchmark Annual Return": bench_ann,
            "Excess Annual Return": excess_ann,
            # Risk
            "Portfolio Annual Vol": port_vol,
            "Benchmark Annual Vol": bench_vol,
            "Downside Deviation": downside_dev,
            "VaR 95%": var_95,
            "CVaR 95%": cvar_95,
            "Max Drawdown": max_dd,
            "Recovery Time (days)": recovery_days,
            "Ulcer Index": ulcer,
            # Risk-adjusted
            "Portfolio Sharpe": port_sharpe,
            "Benchmark Sharpe": bench_sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Omega Ratio": omega,
            # Benchmark-relative
            "Alpha (Jensen)": alpha,
            "Beta": beta,
            "Tracking Error": te,
            "Information Ratio": ir,
            "Hit Rate": hit_rate,
            # Transactions
            "Average Turnover": avg_turnover,
            "Annual Turnover": annual_turnover,
            "Transaction Cost Drag": cost_drag,
            "Rebalances": len(turnover_df) if turnover_df is not None else 0,
            # Distribution
            "Return Skewness": skewness,
            "Excess Kurtosis": excess_kurtosis,
            # Metadata
            "Observation Days": len(port),
        }

    # ------------------------------------------------------------------
    # Formatted report
    # ------------------------------------------------------------------

    def format_report(self, metrics: Dict, strategy_name: str) -> str:
        """
        Render a formatted text report block from a metrics dictionary.

        Parameters
        ----------
        metrics : dict
            Output of compute_metrics().
        strategy_name : str
            Display label for the strategy.

        Returns
        -------
        str
            Multi-line formatted report string.
        """

        def _fmt(key: str, fmt: str = ".2%") -> str:
            v = metrics.get(key, np.nan)
            if np.isnan(v):
                return "N/A"
            return format(v, fmt)

        def _fmt_f(key: str, decimals: int = 2) -> str:
            v = metrics.get(key, np.nan)
            if np.isnan(v):
                return "N/A"
            return f"{v:.{decimals}f}"

        lines = [
            "",
            "=" * 72,
            f"PERFORMANCE REPORT  |  {strategy_name.upper()}",
            "=" * 72,
            "",
            "RETURN METRICS",
            "-" * 36,
            f"  Portfolio Total Return     {_fmt('Portfolio Total Return'):>10}",
            f"  Portfolio CAGR             {_fmt('Portfolio CAGR'):>10}",
            f"  Benchmark Total Return     {_fmt('Benchmark Total Return'):>10}",
            f"  Benchmark CAGR             {_fmt('Benchmark CAGR'):>10}",
            f"  Excess Annual Return       {_fmt('Excess Annual Return'):>10}",
            "",
            "RISK METRICS",
            "-" * 36,
            f"  Portfolio Volatility       {_fmt('Portfolio Annual Vol'):>10}",
            f"  Benchmark Volatility       {_fmt('Benchmark Annual Vol'):>10}",
            f"  Downside Deviation         {_fmt('Downside Deviation'):>10}",
            f"  VaR (95%, annualized)      {_fmt('VaR 95%'):>10}",
            f"  CVaR (95%, annualized)     {_fmt('CVaR 95%'):>10}",
            f"  Maximum Drawdown           {_fmt('Max Drawdown'):>10}",
            f"  Recovery Time              {metrics.get('Recovery Time (days)', 'N/A')} days",
            f"  Ulcer Index                {_fmt_f('Ulcer Index', 2):>10}",
            "",
            "RISK-ADJUSTED METRICS",
            "-" * 36,
            f"  Portfolio Sharpe Ratio     {_fmt_f('Portfolio Sharpe'):>10}",
            f"  Benchmark Sharpe Ratio     {_fmt_f('Benchmark Sharpe'):>10}",
            f"  Sortino Ratio              {_fmt_f('Sortino Ratio'):>10}",
            f"  Calmar Ratio               {_fmt_f('Calmar Ratio'):>10}",
            f"  Omega Ratio                {_fmt_f('Omega Ratio'):>10}",
            "",
            "BENCHMARK-RELATIVE METRICS",
            "-" * 36,
            f"  Jensen Alpha               {_fmt('Alpha (Jensen)'):>10}",
            f"  Beta                       {_fmt_f('Beta'):>10}",
            f"  Tracking Error             {_fmt('Tracking Error'):>10}",
            f"  Information Ratio          {_fmt_f('Information Ratio'):>10}",
            f"  Hit Rate                   {_fmt('Hit Rate'):>10}",
            "",
            "TRANSACTION ANALYSIS",
            "-" * 36,
            f"  Average Turnover / Period  {_fmt('Average Turnover'):>10}",
            f"  Annualized Turnover        {_fmt('Annual Turnover'):>10}",
            f"  Transaction Cost Drag      {_fmt('Transaction Cost Drag'):>10}",
            f"  Number of Rebalances       {metrics.get('Rebalances', 'N/A'):>10}",
            "",
            "RETURN DISTRIBUTION",
            "-" * 36,
            f"  Skewness                   {_fmt_f('Return Skewness'):>10}",
            f"  Excess Kurtosis            {_fmt_f('Excess Kurtosis'):>10}",
            "",
            "=" * 72,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Comparison table and conclusions
    # ------------------------------------------------------------------

    def comparison_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Build a strategy comparison table from a dictionary of metric dicts.

        Parameters
        ----------
        results : dict
            Keys are strategy names; values are output of compute_metrics().

        Returns
        -------
        pd.DataFrame
            Transposed DataFrame with strategies as rows.
        """
        key_metrics = [
            "Portfolio CAGR",
            "Portfolio Annual Vol",
            "Portfolio Sharpe",
            "Sortino Ratio",
            "Calmar Ratio",
            "Max Drawdown",
            "CVaR 95%",
            "Alpha (Jensen)",
            "Beta",
            "Information Ratio",
            "Hit Rate",
            "Annual Turnover",
            "Transaction Cost Drag",
        ]
        df = pd.DataFrame(results).T[key_metrics]
        df.index.name = "Strategy"
        return df

    def generate_conclusions(self, results: Dict[str, Dict]) -> str:
        """
        Generate evidence-based analytical commentary from backtest results.

        Parameters
        ----------
        results : dict
            Output of compute_metrics() for each strategy.

        Returns
        -------
        str
            Formatted analytical conclusions section.
        """
        if not results:
            return "Insufficient data to generate conclusions."

        df = pd.DataFrame(results).T
        lines = [
            "",
            "=" * 72,
            "ANALYTICAL CONCLUSIONS",
            "=" * 72,
            "",
        ]

        # --- Best strategy identification ---
        best_sharpe = df["Portfolio Sharpe"].idxmax()
        best_ir = df["Information Ratio"].idxmax()
        best_dd = df["Max Drawdown"].idxmax()  # Least negative
        best_calmar = df["Calmar Ratio"].idxmax()

        lines += [
            "Strategy Ranking",
            "-" * 40,
            f"  Highest Sharpe Ratio:      {best_sharpe} "
            f"({df.loc[best_sharpe, 'Portfolio Sharpe']:.2f})",
            f"  Best Information Ratio:    {best_ir} "
            f"({df.loc[best_ir, 'Information Ratio']:.2f})",
            f"  Lowest Maximum Drawdown:   {best_dd} "
            f"({df.loc[best_dd, 'Max Drawdown']:.2%})",
            f"  Best Calmar Ratio:         {best_calmar} "
            f"({df.loc[best_calmar, 'Calmar Ratio']:.2f})",
            "",
        ]

        # --- Alpha generation ---
        lines += ["Alpha Generation", "-" * 40]
        for strategy, row in df.iterrows():
            alpha = row.get("Alpha (Jensen)", np.nan)
            if not np.isnan(alpha):
                direction = "positive" if alpha > 0 else "negative"
                lines.append(
                    f"  {strategy}: Jensen alpha = {alpha:.2%} ({direction} "
                    "stock selection / factor exposure)"
                )
        lines.append("")

        # --- Risk-return efficiency ---
        lines += ["Risk-Return Efficiency", "-" * 40]
        bench_vol = df["Benchmark Sharpe"].mean()  # proxy
        for strategy, row in df.iterrows():
            sharpe_vs_bench = (
                row["Portfolio Sharpe"] / df["Benchmark Sharpe"].mean()
                if df["Benchmark Sharpe"].mean() > 0
                else np.nan
            )
            rel = (
                f"{sharpe_vs_bench:.1f}x benchmark Sharpe"
                if not np.isnan(sharpe_vs_bench)
                else "N/A"
            )
            lines.append(f"  {strategy}: {rel}")
        lines.append("")

        # --- Drawdown and risk observations ---
        lines += ["Drawdown and Tail Risk", "-" * 40]
        for strategy, row in df.iterrows():
            dd = row.get("Max Drawdown", np.nan)
            cvar = row.get("CVaR 95%", np.nan)
            if not np.isnan(dd) and not np.isnan(cvar):
                lines.append(
                    f"  {strategy}: Max DD = {dd:.2%}, "
                    f"CVaR (95%) = {cvar:.2%}"
                )
        lines.append("")

        # --- Transaction efficiency ---
        lines += ["Transaction Efficiency", "-" * 40]
        for strategy, row in df.iterrows():
            drag = row.get("Transaction Cost Drag", np.nan)
            if not np.isnan(drag):
                lines.append(
                    f"  {strategy}: cost drag = {drag:.2%} p.a. "
                    f"(turnover = {row.get('Annual Turnover', 0):.0%})"
                )
        lines.append("")

        # --- Overall recommendation ---
        lines += ["Investment Recommendation", "-" * 40]
        recommendation = self._recommend(df)
        lines += [f"  {line}" for line in recommendation]
        lines.append("")
        lines.append("=" * 72)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Wealth projection (present value framework)
    # ------------------------------------------------------------------

    def wealth_projection(
        self,
        initial_capital: float,
        strategies: Dict[str, Dict],
        horizon_years: List[int] = None,
        inflation_rate: float = 0.025,
    ) -> pd.DataFrame:
        """
        Project terminal wealth and present values for each strategy.

        Computes nominal and real (inflation-adjusted) terminal wealth
        at multiple horizons using each strategy's backtested CAGR.

        Parameters
        ----------
        initial_capital : float
            Starting portfolio value in base currency.
        strategies : dict
            Output of compute_metrics() for each strategy.
        horizon_years : list of int
            Time horizons in years (default: [5, 10, 15, 20]).
        inflation_rate : float
            Annual inflation rate for deflating nominal projections.

        Returns
        -------
        pd.DataFrame
            Multi-index DataFrame with (strategy, horizon) rows.
        """
        horizon_years = horizon_years or [5, 10, 15, 20]
        rows = []

        for strategy, metrics in strategies.items():
            cagr = metrics.get("Portfolio CAGR", np.nan)
            if np.isnan(cagr):
                continue
            real_cagr = (1 + cagr) / (1 + inflation_rate) - 1

            for h in horizon_years:
                fv_nominal = initial_capital * (1 + cagr) ** h
                fv_real = initial_capital * (1 + real_cagr) ** h
                # Present value of terminal wealth discounted at CAGR
                pv_terminal = initial_capital  # by definition (FV discounted at CAGR)
                gain = fv_nominal - initial_capital
                multiple = fv_nominal / initial_capital

                rows.append(
                    {
                        "Strategy": strategy,
                        "Horizon (yr)": h,
                        "CAGR (nominal)": cagr,
                        "CAGR (real)": real_cagr,
                        "Terminal Wealth (nominal)": fv_nominal,
                        "Terminal Wealth (real)": fv_real,
                        "Capital Gain": gain,
                        "Wealth Multiple": multiple,
                    }
                )

        return pd.DataFrame(rows).set_index(["Strategy", "Horizon (yr)"])

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _max_drawdown_series(returns: pd.Series) -> pd.Series:
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        return (cumulative - rolling_max) / rolling_max

    @staticmethod
    def _recovery_time(drawdown_series: pd.Series) -> Optional[int]:
        """Return the number of days required to recover from the max drawdown."""
        if drawdown_series.empty:
            return None
        trough_idx = drawdown_series.idxmin()
        post_trough = drawdown_series.loc[trough_idx:]
        recovered = post_trough[post_trough >= 0]
        if recovered.empty:
            return None
        recovery_idx = recovered.index[0]
        return int((recovery_idx - trough_idx).days)

    @staticmethod
    def _recommend(df: pd.DataFrame) -> List[str]:
        """Derive concise strategic recommendations from the comparison table."""
        lines = []
        best_sharpe = df["Portfolio Sharpe"].idxmax()
        best_dd = df["Max Drawdown"].idxmax()
        best_ir = df["Information Ratio"].idxmax()

        lines.append(
            f"For risk-adjusted return maximization: {best_sharpe} is preferred, "
            "exhibiting the highest Sharpe ratio and consistent alpha generation."
        )
        lines.append(
            f"For capital preservation and drawdown control: {best_dd} is "
            "recommended, combining lower maximum drawdown with reduced tail risk."
        )
        if best_ir != best_sharpe:
            lines.append(
                f"For active management mandates benchmarked to the S&P 500: "
                f"{best_ir} delivers the highest Information Ratio, indicating "
                "superior consistency of excess returns per unit of active risk."
            )
        lines.append(
            "All strategies demonstrate positive Jensen alpha, suggesting that "
            "the composite screening and optimization framework generates returns "
            "beyond what systematic market exposure alone would explain."
        )
        return lines
