# professional_portfolio_optimizer_fixed.py
"""
Professional Equity Portfolio Optimizer - FIXED VERSION
=======================================================

CRITICAL FIXES APPLIED:
1. ✅ Eliminated data leakage in rolling_backtest (removed OOS clip_outliers)
2. ✅ Fixed MIN_POSITIONS vs WEIGHT_CAP conflict
3. ✅ Improved backtest warm-up logic
4. ✅ Corrected turnover annualization calculation
5. ✅ Reduced repetitive logging

Enhanced Features:
- Fixed critical bugs in Sharpe calculation and data handling
- Improved asset screening with momentum factor
- Transaction costs and turnover analysis
- More robust outlier handling and data quality controls
- Enhanced reporting with additional professional metrics

Author: Portfolio Management System
Version: 2.2 FIXED
"""
import calendar
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.optimize import minimize
import warnings
from typing import Tuple, Dict, List, Optional
import logging
from dataclasses import dataclass
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# ================================
# ENHANCED CONFIGURATION PARAMETERS
# ================================

@dataclass
class PortfolioConfig:
    """Enhanced configuration class for portfolio optimization parameters."""
    # Asset Universe
    TICKERS: List[str] = None
    BENCHMARK: str = "^GSPC"
    
    # Data Parameters
    YEARS: int = 5
    INTERVAL: str = "1d"
    MIN_COVERAGE: float = 0.80  # Minimum data coverage required
    MIN_SESSIONS: int = 400     # Minimum trading sessions in rolling window (2Y ≈ 504 days)
    
    # Optimization Parameters
    TOP_N: int = 15             # Increased from 10
    WEIGHT_CAP: float = 0.125   # ✅ FIXED: 12.5% to be consistent with MIN_POSITIONS=8
    RF: float = 0.02            # Risk-free rate (annual)
    MIN_POSITIONS: int = 8      # Minimum active positions
    
    # Backtesting Parameters
    WINDOW_YEARS: int = 2       # Reduced from 3 for less lag
    REBALANCE: str = "M"        # Monthly instead of quarterly
    TRANSACTION_COST: float = 0.0015  # 15 bps per rebalancing
    MIN_WARMUP_SESSIONS: int = 60     # ✅ NEW: Minimum sessions before starting backtest
    
    def __post_init__(self):
        if self.TICKERS is None:
            self.TICKERS = [
                "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", 
                "JPM", "XOM", "CVX", "UNH", "JNJ", "PEP", "KO", 
                "PG", "HD", "BAC", "WMT", "DIS", "CRM", "NFLX",
                "V", "MA", "TSM", "ABBV", "TMO"
            ]

# Initialize configuration
CONFIG = PortfolioConfig()

# ================================
# ENHANCED UTILITY FUNCTIONS
# ================================

def enforce_dynamic_cap_fixed(n_assets: int, cap: float, min_positions: int = CONFIG.MIN_POSITIONS) -> float:
    """✅ FIXED: Ensure feasible weight constraints with MIN_POSITIONS logic."""
    # Calculate theoretical cap needed for min_positions
    cap_target = 1.0 / max(min_positions, n_assets)
    
    # Use the more restrictive of the two
    adjusted_cap = min(cap, cap_target)
    
    # Ensure it's feasible
    min_feasible_cap = 1.0 / n_assets
    if adjusted_cap * n_assets < 1.01:  # Small margin for numerical precision
        adjusted_cap = max(min_feasible_cap, adjusted_cap * 1.1)  # Add 10% margin
        logger.debug(f"Weight cap adjusted from {cap:.2%} to {adjusted_cap:.2%} for {n_assets} assets (min_pos={min_positions})")
    
    return adjusted_cap

def quality_filter_fixed(prices: pd.DataFrame, min_coverage: float = CONFIG.MIN_COVERAGE,
                         min_sessions: int = None, window_years: int = CONFIG.WINDOW_YEARS,
                         log_level: str = "INFO") -> pd.DataFrame:
    """✅ FIXED: Enhanced quality filter with better warm-up logic and reduced logging."""
    
    # Calculate actual available sessions in the data
    actual_sessions = len(prices)
    
    # ✅ FIXED: Better adaptive session count requirement
    if min_sessions is None:
        expected_sessions = int(window_years * 252 * 0.75)  # 75% of expected
        min_sessions = min(expected_sessions, max(int(actual_sessions * 0.8), 20))  # At least 80% of available data, min 20 days
    
    # Coverage filter
    coverage = prices.notna().mean()
    coverage_keep = coverage[coverage >= min_coverage].index
    
    # Session count filter  
    session_counts = prices.notna().sum()
    session_keep = session_counts[session_counts >= min_sessions].index
    
    # Combined filter
    keep = list(set(coverage_keep) & set(session_keep))
    
    # ✅ FIXED: Reduced logging spam
    if log_level == "INFO":
        coverage_dropped = [c for c in prices.columns if c not in coverage_keep]
        session_dropped = [c for c in prices.columns if c not in session_keep]
        
        if coverage_dropped:
            logger.debug(f"Excluded due to low coverage (<{min_coverage:.0%}): {coverage_dropped}")
        if session_dropped and len(session_dropped) < len(prices.columns):
            logger.debug(f"Excluded due to insufficient sessions (<{min_sessions}): {session_dropped}")
    
    # ✅ FIXED: Only log relaxation if it actually happens
    original_min_sessions = min_sessions
    
    # Fallback: if no assets pass, relax the session requirement
    if len(keep) < 3:
        min_sessions_relaxed = max(int(actual_sessions * 0.25), 20)  # 25% of available, min 20 days
        session_keep_relaxed = session_counts[session_counts >= min_sessions_relaxed].index
        keep = list(set(coverage_keep) & set(session_keep_relaxed))
        if log_level == "INFO":
            logger.info(f"Relaxed filter: {original_min_sessions}→{min_sessions_relaxed} sessions, kept {len(keep)} assets")
    
    return prices[keep]

def clip_outliers(returns: pd.DataFrame, lower_pct: float = 1, upper_pct: float = 99) -> pd.DataFrame:
    """Robust outlier clipping using percentiles instead of winsorization."""
    quantiles = returns.quantile([lower_pct/100, upper_pct/100])
    returns_clipped = returns.clip(
        lower=quantiles.loc[lower_pct/100], 
        upper=quantiles.loc[upper_pct/100], 
        axis=1
    )
    return returns_clipped

# ================================
# DATA FETCHING
# ================================

def fetch_prices(tickers: List[str], benchmark: Optional[str] = None,
                 years: int = CONFIG.YEARS, interval: str = CONFIG.INTERVAL) -> pd.DataFrame:
    """Download adjusted prices from Yahoo Finance."""
    end = date.today()
    start = end - timedelta(days=365*years)

    # Include benchmark in download
    if benchmark and benchmark not in tickers:
        tickers = tickers + [benchmark]

    logger.info(f"Downloading {len(tickers)} symbols from {start} to {end}")
    
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # Handle different yfinance return structures
    if isinstance(raw, pd.DataFrame) and "Close" in raw.columns:
        close = raw["Close"].copy()
    else:
        close = raw["Close"].to_frame(tickers[0]) if len(tickers) == 1 else raw["Close"]

    # Data cleaning
    close = close.sort_index()
    close = close.loc[~close.index.duplicated(keep="last")]
    close = close.dropna(how="all")
    close = close.dropna(axis=1, how="all")

    # Reporting
    loaded = list(close.columns)
    missing = [t for t in tickers if t not in loaded]
    logger.info(f"Loaded {len(loaded)} symbols successfully")
    if missing:
        logger.warning(f"Missing data for: {missing}")
    logger.info(f"Date range: {close.index.min().date()} → {close.index.max().date()} ({close.shape[0]} trading days)")

    return close

# ================================
# ENHANCED METRICS CALCULATION
# ================================

def compute_metrics(prices: pd.DataFrame, benchmark: Optional[str] = None, 
                   rf: float = CONFIG.RF) -> pd.DataFrame:
    """Calculate comprehensive asset metrics with momentum factor."""
    df = prices.copy()
    
    # Exclude benchmark from asset metrics
    if benchmark and benchmark in df.columns:
        df_assets = df.drop(columns=[benchmark])
    else:
        df_assets = df

    returns = df_assets.pct_change().dropna()
    returns = clip_outliers(returns)  # ✅ Enhanced outlier handling IN-SAMPLE only
    
    # Time period calculation
    years = (df_assets.index[-1] - df_assets.index[0]).days / 365.25
    
    # Core metrics
    total_return = (df_assets.iloc[-1] / df_assets.iloc[0]) - 1
    cagr = (df_assets.iloc[-1] / df_assets.iloc[0]) ** (1/years) - 1
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol
    
    # Momentum factors (6-month and 12-month) - FIXED: Now calculating returns, not ratios
    momentum_6m = (df_assets.iloc[-1] / df_assets.iloc[-126] - 1).fillna(0) if len(df_assets) >= 126 else pd.Series(0, index=df_assets.columns)
    momentum_12m = (df_assets.iloc[-1] / df_assets.iloc[-252] - 1).fillna(0) if len(df_assets) >= 252 else pd.Series(0, index=df_assets.columns)
    
    # Risk metrics
    var_95 = returns.quantile(0.05) * np.sqrt(252)
    max_dd = calculate_max_drawdown(df_assets)
    
    # Skewness and Kurtosis
    skew = returns.skew()
    kurt = returns.kurtosis()
    
    metrics = pd.DataFrame({
        "TotalReturn": total_return,
        "CAGR": cagr,
        "AnnualizedReturn": ann_ret,
        "AnnualizedVol": ann_vol,
        "Sharpe": sharpe,
        "Momentum6M": momentum_6m,
        "Momentum12M": momentum_12m,
        "VaR_95": var_95,
        "MaxDrawdown": max_dd,
        "Skewness": skew,
        "Kurtosis": kurt
    })
    
    return metrics.replace([np.inf, -np.inf], np.nan).dropna().sort_values("Sharpe", ascending=False)

def calculate_max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """Calculate maximum drawdown for each asset."""
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown.min()

def tracking_error_ir(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
    """Calculate Tracking Error and Information Ratio."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 30:
        return np.nan, np.nan
    
    excess_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = excess_returns.std() * np.sqrt(252)
    ir = (excess_returns.mean() * 252) / te if te > 0 else np.nan
    
    return float(te), float(ir)

def calculate_turnover(weights_new: pd.Series, weights_old: pd.Series) -> float:
    """Calculate portfolio turnover between rebalancing periods."""
    if weights_old.empty:
        return 1.0  # First period is 100% turnover
    
    # Align weights
    all_assets = set(weights_new.index) | set(weights_old.index)
    w_new = pd.Series(0.0, index=all_assets)
    w_old = pd.Series(0.0, index=all_assets)
    
    w_new.update(weights_new)
    w_old.update(weights_old)
    
    # Turnover is sum of absolute weight changes
    return np.sum(np.abs(w_new - w_old)) / 2

# ================================
# ENHANCED ASSET SCREENING
# ================================

def screen_assets(metrics: pd.DataFrame, top_n: int = CONFIG.TOP_N, 
                 method: str = "enhanced_composite") -> pd.DataFrame:
    """Enhanced asset screening with momentum and reduced low-vol bias."""
    m = metrics.copy()
    
    def normalize_metric(series):
        """Robust normalization handling edge cases."""
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if series.max() == series.min() or series.std() == 0:
            return pd.Series(0.5, index=series.index)
        return (series - series.min()) / (series.max() - series.min())
    
    if method == "enhanced_composite":
        # Enhanced multi-factor scoring with reduced low-vol bias
        sharpe_norm = normalize_metric(m["Sharpe"])
        vol_norm = normalize_metric(m["AnnualizedVol"])
        dd_norm = normalize_metric(m["MaxDrawdown"])
        momentum_norm = normalize_metric(m["Momentum12M"])
        
        # New weights: 70% Sharpe, 20% Low Vol, 10% Momentum
        m["Score"] = (0.7 * sharpe_norm + 
                     0.2 * (1 - vol_norm) + 
                     0.1 * momentum_norm)
                     
    elif method == "momentum_focused":
        # Focus on momentum and risk-adjusted returns
        sharpe_norm = normalize_metric(m["Sharpe"])
        momentum_norm = normalize_metric(m["Momentum12M"])
        m["Score"] = 0.6 * sharpe_norm + 0.4 * momentum_norm
        
    elif method == "risk_adjusted":
        # Focus on risk-adjusted returns
        m["Score"] = normalize_metric(m["Sharpe"])
        
    elif method == "low_risk":
        # Conservative approach
        vol_norm = normalize_metric(m["AnnualizedVol"])
        dd_norm = normalize_metric(m["MaxDrawdown"])
        m["Score"] = 0.6 * (1 - vol_norm) + 0.4 * (1 - dd_norm)
    
    return m.sort_values("Score", ascending=False).head(top_n)

# ================================
# ENHANCED PORTFOLIO OPTIMIZATION
# ================================

def calculate_portfolio_stats(weights: np.ndarray, returns: pd.DataFrame, 
                            rf: float = CONFIG.RF) -> Tuple[float, float, float]:
    """Calculate portfolio statistics with explicit dot product."""
    # Fixed: Use explicit dot product for robust calculation
    portfolio_return = (returns.mean() * 252).dot(weights)
    portfolio_std = np.sqrt(weights.T @ (returns.cov() * 252).values @ weights)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std if portfolio_std > 0 else 0
    return portfolio_return, portfolio_std, sharpe_ratio

def optimize_max_sharpe_fixed(prices: pd.DataFrame, picks: List[str]) -> Tuple[pd.Series, Tuple[float, float, float]]:
    """✅ FIXED: Optimize for maximum Sharpe ratio with corrected weight cap logic."""
    sub = prices[picks].dropna()
    returns = sub.pct_change().dropna()
    returns = clip_outliers(returns)  # Consistent outlier treatment
    
    n = len(picks)
    
    # ✅ FIXED: Proper cap calculation considering MIN_POSITIONS
    cap_target = 1.0 / max(CONFIG.MIN_POSITIONS, n)
    cap = min(CONFIG.WEIGHT_CAP, cap_target)
    cap = enforce_dynamic_cap_fixed(n, cap, CONFIG.MIN_POSITIONS)
    
    logger.debug(f"Optimizing Max Sharpe with {n} assets, cap: {cap:.2%}")

    def objective(weights):
        _, _, sharpe = calculate_portfolio_stats(weights, returns, rf=CONFIG.RF)
        return -sharpe

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, cap) for _ in range(n))
    x0 = np.full(n, 1/n)

    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'ftol': 1e-9, 'maxiter': 500})
        
        if result.success:
            weights = result.x
        else:
            logger.warning(f"Max Sharpe optimization failed: {result.message}")
            weights = x0
            
    except Exception as e:
        logger.warning(f"Max Sharpe optimization error: {e}")
        weights = x0

    weights_series = pd.Series(weights, index=picks).sort_values(ascending=False)
    performance = calculate_portfolio_stats(weights, returns, rf=CONFIG.RF)
    
    # ✅ FIXED: Better position count check
    active_positions = (weights_series > 1e-4).sum()
    if active_positions < CONFIG.MIN_POSITIONS:
        logger.debug(f"Only {active_positions} active positions, below minimum {CONFIG.MIN_POSITIONS}")
    
    return weights_series[weights_series > 1e-4], performance

def optimize_min_variance_fixed(prices: pd.DataFrame, picks: List[str]) -> Tuple[pd.Series, Tuple[float, float, float]]:
    """✅ FIXED: Optimize for minimum variance with corrected weight cap logic."""
    sub = prices[picks].dropna()
    returns = sub.pct_change().dropna()
    returns = clip_outliers(returns)
    
    n = len(picks)
    
    # ✅ FIXED: Proper cap calculation considering MIN_POSITIONS
    cap_target = 1.0 / max(CONFIG.MIN_POSITIONS, n)
    cap = min(CONFIG.WEIGHT_CAP, cap_target)
    cap = enforce_dynamic_cap_fixed(n, cap, CONFIG.MIN_POSITIONS)
    
    logger.debug(f"Optimizing Min Variance with {n} assets, cap: {cap:.2%}")

    def objective(weights):
        return weights.T @ (returns.cov() * 252).values @ weights

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, cap) for _ in range(n))
    x0 = np.full(n, 1/n)

    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'ftol': 1e-9, 'maxiter': 500})
        
        if result.success:
            weights = result.x
        else:
            logger.warning(f"Min Variance optimization failed: {result.message}")
            weights = x0
            
    except Exception as e:
        logger.warning(f"Min Variance optimization error: {e}")
        weights = x0

    weights_series = pd.Series(weights, index=picks).sort_values(ascending=False)
    performance = calculate_portfolio_stats(weights, returns, rf=CONFIG.RF)
    
    return weights_series[weights_series > 1e-4], performance

def optimize_risk_parity_fixed(prices: pd.DataFrame, picks: List[str]) -> Tuple[pd.Series, Tuple[float, float, float]]:
    """✅ FIXED: Risk Parity optimization with enhanced regularization."""
    sub = prices[picks].dropna()
    returns = sub.pct_change().dropna()
    returns = clip_outliers(returns)
    
    n = len(picks)
    logger.debug(f"Optimizing Risk Parity with {n} assets")
    
    # Enhanced covariance with simple shrinkage
    cov_sample = returns.cov() * 252
    cov_target = np.eye(n) * np.trace(cov_sample) / n  # Identity scaled
    shrinkage = 0.1  # 10% shrinkage
    cov_matrix = (1 - shrinkage) * cov_sample + shrinkage * cov_target

    def risk_parity_objective(weights):
        portfolio_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
        marginal_contrib = (cov_matrix.values @ weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        target_risk = portfolio_vol / n
        return np.sum((risk_contrib - target_risk) ** 2)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.01, 0.4) for _ in range(n))  # Adjusted bounds
    x0 = np.full(n, 1/n)

    try:
        result = minimize(risk_parity_objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints,
                         options={'ftol': 1e-9, 'maxiter': 500})
        
        if result.success:
            weights = result.x
        else:
            logger.warning(f"Risk Parity optimization failed: {result.message}")
            weights = x0
            
    except Exception as e:
        logger.warning(f"Risk Parity optimization error: {e}")
        weights = x0

    weights_series = pd.Series(weights, index=picks).sort_values(ascending=False)
    performance = calculate_portfolio_stats(weights, returns, rf=CONFIG.RF)
    
    return weights_series[weights_series > 1e-4], performance

# ================================
# ✅ FIXED BACKTESTING FRAMEWORK
# ================================

def rolling_backtest_fixed(prices: pd.DataFrame, universe: List[str], 
                          optimization_method: str = "max_sharpe",
                          screening_method: str = "enhanced_composite") -> Tuple[pd.Series, pd.DataFrame]:
    """✅ FIXED: Enhanced rolling backtest with all critical fixes applied."""
    
    logger.info(f"Starting FIXED rolling backtest - {optimization_method.upper()}")
    logger.info(f"Window: {CONFIG.WINDOW_YEARS}Y, Rebalancing: {CONFIG.REBALANCE}, Method: {screening_method}")
    
    px = prices[universe].dropna(how="all")
    returns = px.pct_change()  # ✅ FIX 1: NO clip_outliers here (was data leakage!)
    dates = returns.index
    
    # Generate rebalancing dates
    rebalance_dates = pd.Series(index=dates, data=1).resample(CONFIG.REBALANCE).last().index
    portfolio_returns = pd.Series(index=dates, dtype=float)
    
    # Track weights and turnover
    weights_history = []
    turnover_history = []
    previous_weights = pd.Series(dtype=float)  # ✅ FIX: Proper initialization
    
    rebalance_count = 0
    
    for t in rebalance_dates:
        # Define historical training window
        start_date = t - pd.DateOffset(years=CONFIG.WINDOW_YEARS)
        hist_data = px.loc[(px.index > start_date) & (px.index <= t)].dropna(how="all", axis=1)
        
        # ✅ FIX 3: Early warm-up check
        if len(hist_data) < CONFIG.MIN_WARMUP_SESSIONS:
            logger.debug(f"Skipping {t.date()}: only {len(hist_data)} sessions < {CONFIG.MIN_WARMUP_SESSIONS} required")
            continue
            
        # Apply enhanced quality filter with dynamic window (reduced logging)
        hist_data = quality_filter_fixed(hist_data, window_years=CONFIG.WINDOW_YEARS, log_level="DEBUG")
        
        if hist_data.shape[1] < CONFIG.MIN_POSITIONS:
            logger.debug(f"Skipping {t.date()}: only {hist_data.shape[1]} assets < {CONFIG.MIN_POSITIONS} required")
            continue
            
        try:
            # Calculate metrics and screen assets
            metrics = compute_metrics(hist_data, benchmark=CONFIG.BENCHMARK, rf=CONFIG.RF)
            selected_df = screen_assets(metrics, top_n=CONFIG.TOP_N, method=screening_method)
            selected_assets = selected_df.index.tolist()
            
            if len(selected_assets) < CONFIG.MIN_POSITIONS:
                continue
            
            # ✅ FIX 2: Use FIXED optimizers with proper cap logic
            if optimization_method == "max_sharpe":
                weights, _ = optimize_max_sharpe_fixed(hist_data, selected_assets)
            elif optimization_method == "min_variance":
                weights, _ = optimize_min_variance_fixed(hist_data, selected_assets)
            elif optimization_method == "risk_parity":
                weights, _ = optimize_risk_parity_fixed(hist_data, selected_assets)
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            # Calculate turnover
            turnover = calculate_turnover(weights, previous_weights)
            turnover_history.append({'date': t, 'turnover': turnover})
            weights_history.append({'date': t, 'weights': weights.copy()})
            previous_weights = weights.copy()
            
            # Apply weights to next period
            next_rebalance = rebalance_dates[rebalance_dates > t]
            end_date = next_rebalance[0] if len(next_rebalance) > 0 else dates[-1]
            
            period_returns = returns.loc[(returns.index > t) & (returns.index <= end_date), weights.index]
            if not period_returns.empty:
                # Calculate portfolio returns
                gross_returns = period_returns @ weights.values
                
                # Apply transaction costs (only on rebalancing day)
                if not gross_returns.empty:
                    net_returns = gross_returns.copy()
                    # Subtract transaction cost from first day of period
                    net_returns.iloc[0] -= CONFIG.TRANSACTION_COST * turnover
                    portfolio_returns.loc[period_returns.index] = net_returns
                
                rebalance_count += 1
                
        except Exception as e:
            logger.warning(f"Skipping rebalance at {t.date()}: {e}")
            continue
    
    logger.info(f"Completed {rebalance_count} rebalancing periods")
    
    # Create turnover DataFrame
    turnover_df = pd.DataFrame(turnover_history)
    
    return portfolio_returns.dropna(), turnover_df

# ================================
# ✅ FIXED PERFORMANCE ANALYSIS
# ================================

def comprehensive_analysis_fixed(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                                turnover_df: pd.DataFrame = None) -> Dict:
    """✅ FIXED: Enhanced comprehensive performance analysis with corrected turnover calculation."""
    
    # Align series
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return {}
        
    port_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]
    
    # Calculate metrics
    port_cumret = (1 + port_ret).cumprod().iloc[-1] - 1
    bench_cumret = (1 + bench_ret).cumprod().iloc[-1] - 1
    
    port_ann_ret = port_ret.mean() * 252
    bench_ann_ret = bench_ret.mean() * 252
    
    port_ann_vol = port_ret.std() * np.sqrt(252)
    bench_ann_vol = bench_ret.std() * np.sqrt(252)
    
    # FIXED: Correct Sharpe calculation for benchmark
    port_sharpe = (port_ann_ret - CONFIG.RF) / port_ann_vol
    bench_sharpe = (bench_ann_ret - CONFIG.RF) / bench_ann_vol  # Fixed bug here
    
    # Risk metrics
    te, ir = tracking_error_ir(port_ret, bench_ret)
    
    # Drawdown analysis
    port_cumulative = (1 + port_ret).cumprod()
    port_dd = (port_cumulative / port_cumulative.expanding().max() - 1)
    max_dd = port_dd.min()
    
    # Additional metrics
    excess_ret = port_ann_ret - bench_ann_ret
    hit_rate = (port_ret > bench_ret).mean()
    
    # Enhanced metrics
    years_analyzed = len(port_ret) / 252
    port_cagr = (1 + port_cumret) ** (1/years_analyzed) - 1
    bench_cagr = (1 + bench_cumret) ** (1/years_analyzed) - 1
    
    # ✅ FIX 4: Corrected turnover annualization
    avg_turnover = turnover_df['turnover'].mean() if turnover_df is not None and not turnover_df.empty else np.nan
    
    # Use actual rebalancing frequency instead of approximation
    freq_to_per_year = {"M": 12, "Q": 4, "A": 1, "W": 52}
    k = freq_to_per_year.get(CONFIG.REBALANCE, 12)
    annual_turnover = avg_turnover * k if not np.isnan(avg_turnover) else np.nan
    
    # Calculate portfolio beta
    beta = np.cov(port_ret, bench_ret)[0, 1] / np.var(bench_ret) if np.var(bench_ret) > 0 else np.nan
    
    return {
        "Portfolio Total Return": port_cumret,
        "Benchmark Total Return": bench_cumret,
        "Portfolio CAGR": port_cagr,
        "Benchmark CAGR": bench_cagr,
        "Portfolio Annual Return": port_ann_ret,
        "Benchmark Annual Return": bench_ann_ret,
        "Portfolio Annual Vol": port_ann_vol,
        "Benchmark Annual Vol": bench_ann_vol,
        "Portfolio Sharpe": port_sharpe,
        "Benchmark Sharpe": bench_sharpe,
        "Excess Return": excess_ret,
        "Tracking Error": te,
        "Information Ratio": ir,
        "Hit Rate": hit_rate,
        "Max Drawdown": max_dd,
        "Portfolio Beta": beta,
        "Average Turnover": avg_turnover,
        "Annual Turnover": annual_turnover,
        "Rebalances": len(turnover_df) if turnover_df is not None else 0,
        "Periods": len(port_ret)
    }

# ================================
# ENHANCED REPORTING
# ================================

def generate_enhanced_report(results: Dict, strategy_name: str) -> str:
    """Generate enhanced professional performance report."""
    
    report = f"""
    
{'='*70}
FIXED PORTFOLIO PERFORMANCE REPORT - {strategy_name.upper()}
{'='*70}

PERFORMANCE SUMMARY
{'-'*35}
Portfolio Total Return:       {results.get('Portfolio Total Return', 0):.2%}
Portfolio CAGR:               {results.get('Portfolio CAGR', 0):.2%}
Benchmark Total Return:       {results.get('Benchmark Total Return', 0):.2%}
Benchmark CAGR:               {results.get('Benchmark CAGR', 0):.2%}
Excess Return (Annual):       {results.get('Excess Return', 0):.2%}

RISK-ADJUSTED METRICS
{'-'*35}
Portfolio Sharpe Ratio:       {results.get('Portfolio Sharpe', 0):.2f}
Benchmark Sharpe Ratio:       {results.get('Benchmark Sharpe', 0):.2f}
Information Ratio:            {results.get('Information Ratio', 0):.2f}
Portfolio Beta:               {results.get('Portfolio Beta', 0):.2f}

RISK METRICS
{'-'*35}
Portfolio Volatility:         {results.get('Portfolio Annual Vol', 0):.2%}
Benchmark Volatility:         {results.get('Benchmark Annual Vol', 0):.2%}
Tracking Error:               {results.get('Tracking Error', 0):.2%}
Maximum Drawdown:             {results.get('Max Drawdown', 0):.2%}

TRANSACTION ANALYSIS (FIXED)
{'-'*35}
Average Turnover:             {results.get('Average Turnover', 0):.2%}
Annualized Turnover:          {results.get('Annual Turnover', 0):.2%}
Number of Rebalances:         {results.get('Rebalances', 0):,}

ADDITIONAL STATISTICS
{'-'*35}
Hit Rate:                     {results.get('Hit Rate', 0):.2%}
Observation Periods:          {results.get('Periods', 0):,}

{'='*70}
    """
    
    return report

# ================================
# MAIN EXECUTION - FIXED VERSION
# ================================

def main():
    """✅ FIXED: Enhanced main execution function with all critical fixes + VISUALIZATIONS."""
    logger.info("PROFESSIONAL EQUITY PORTFOLIO OPTIMIZER - FIXED VERSION 2.2")
    logger.info("=" * 70)
    
    # 1. Fetch Data
    logger.info("\n[STEP 1] Fetching market data...")
    prices = fetch_prices(CONFIG.TICKERS, benchmark=CONFIG.BENCHMARK, 
                         years=CONFIG.YEARS, interval=CONFIG.INTERVAL)
    
    # 2. Apply enhanced quality filter
    logger.info("\n[STEP 2] Applying FIXED data quality filters...")
    clean_prices = quality_filter_fixed(prices)
    
    # 3. Calculate current metrics
    logger.info("\n[STEP 3] Computing enhanced asset metrics...")
    current_metrics = compute_metrics(clean_prices, benchmark=CONFIG.BENCHMARK, rf=CONFIG.RF)
    
    print("\nTOP ASSETS BY SHARPE RATIO:")
    print("-" * 40)
    display_cols = ["CAGR", "AnnualizedVol", "Sharpe", "Momentum12M", "MaxDrawdown"]
    print(current_metrics[display_cols].head(15).round(4))
    
    # 4. Enhanced asset screening
    logger.info(f"\n[STEP 4] Enhanced screening top {CONFIG.TOP_N} assets...")
    selected_assets = screen_assets(current_metrics, top_n=CONFIG.TOP_N, method="enhanced_composite")
    picks = selected_assets.index.tolist()
    
    print(f"\nSELECTED ASSETS (Enhanced Screening):")
    print("-" * 35)
    print(selected_assets[["CAGR", "AnnualizedVol", "Sharpe", "Momentum12M", "Score"]].round(4))
    
    # 5. FIXED current portfolio optimization
    logger.info(f"\n[STEP 5] Optimizing FIXED current portfolios...")
    
    # Max Sharpe
    weights_sharpe, perf_sharpe = optimize_max_sharpe_fixed(clean_prices, picks)
    print(f"\nFIXED MAX SHARPE PORTFOLIO:")
    print("-" * 25)
    print(weights_sharpe.round(4))
    print(f"Expected Return: {perf_sharpe[0]:.2%}")
    print(f"Volatility: {perf_sharpe[1]:.2%}")
    print(f"Sharpe Ratio: {perf_sharpe[2]:.2f}")
    
    # Min Variance
    weights_minvar, perf_minvar = optimize_min_variance_fixed(clean_prices, picks)
    print(f"\nFIXED MIN VARIANCE PORTFOLIO:")
    print("-" * 25)
    print(weights_minvar.round(4))
    print(f"Expected Return: {perf_minvar[0]:.2%}")
    print(f"Volatility: {perf_minvar[1]:.2%}")
    print(f"Sharpe Ratio: {perf_minvar[2]:.2f}")
    
    # Risk Parity
    weights_rp, perf_rp = optimize_risk_parity_fixed(clean_prices, picks)
    print(f"\nFIXED RISK PARITY PORTFOLIO:")
    print("-" * 25)
    print(weights_rp.round(4))
    print(f"Expected Return: {perf_rp[0]:.2%}")
    print(f"Volatility: {perf_rp[1]:.2%}")
    print(f"Sharpe Ratio: {perf_rp[2]:.2f}")
    
    # 6. FIXED enhanced rolling backtests
    logger.info(f"\n[STEP 6] Running FIXED enhanced out-of-sample backtests...")
    
    universe = [c for c in clean_prices.columns if c != CONFIG.BENCHMARK]
    
    # Run backtests for different strategies
    strategies = ["max_sharpe", "min_variance", "risk_parity"]
    backtest_results = {}
    strategy_returns = {}  # 🆕 NEW: Store returns for visualizations
    
    for strategy in strategies:
        logger.info(f"\nRunning FIXED {strategy} backtest...")
        bt_returns, turnover_df = rolling_backtest_fixed(clean_prices, universe, 
                                                        optimization_method=strategy,
                                                        screening_method="enhanced_composite")
        
        if not bt_returns.empty:
            # Get aligned benchmark returns
            bench_returns = clean_prices[CONFIG.BENCHMARK].pct_change().iloc[1:]
            bench_aligned = bench_returns.reindex(bt_returns.index).dropna()
            bt_aligned = bt_returns.reindex(bench_aligned.index)
            
            # 🆕 NEW: Store returns for visualizations
            strategy_returns[strategy] = bt_aligned
            
            # FIXED comprehensive analysis
            analysis = comprehensive_analysis_fixed(bt_aligned, bench_aligned, turnover_df)
            backtest_results[strategy] = analysis
            
            # Generate and print enhanced report
            report = generate_enhanced_report(analysis, strategy)
            print(report)
    
    # 7. 🚀 NEW: CREATE PERFORMANCE VISUALIZATIONS (PRIORITY 1)
    if backtest_results and strategy_returns:
        print("\n" + "="*90)
        print("📊 GENERATING PRIORITY 1 PERFORMANCE VISUALIZATIONS")
        print("="*90)
        
        # Get benchmark returns for comparison
        bench_returns = clean_prices[CONFIG.BENCHMARK].pct_change().iloc[1:]
        
        for strategy in strategies:
            if strategy in strategy_returns:
                print(f"\n🔄 Creating dashboard for {strategy.upper()}...")
                
                portfolio_returns = strategy_returns[strategy]
                bench_aligned = bench_returns.reindex(portfolio_returns.index).dropna()
                port_aligned = portfolio_returns.reindex(bench_aligned.index)
                
                # Create comprehensive dashboard
                strategy_display_name = f"{strategy.replace('_', ' ').title()} Strategy"
                create_performance_dashboard(port_aligned, bench_aligned, strategy_display_name)
    
    # 🆕 NEW: PRIORITY 2 COMPREHENSIVE ANALYSIS
    if backtest_results and strategy_returns:
        print("\n" + "="*90)
        print("📊 GENERATING PRIORITY 2 COMPREHENSIVE ANALYSIS")
        print("="*90)
        
        # Collect current portfolio weights for composition analysis
        current_weights = {
            'max_sharpe': weights_sharpe,
            'min_variance': weights_minvar, 
            'risk_parity': weights_rp
        }
        
        # Create comprehensive analysis dashboard
        create_comprehensive_analysis_dashboard(backtest_results, strategy_returns, current_weights)
    
    # 8. Enhanced summary comparison
    if backtest_results:
        print("\n" + "="*90)
        print("FIXED STRATEGY COMPARISON SUMMARY")
        print("="*90)
        
        comparison_df = pd.DataFrame(backtest_results).T
        key_metrics = ["Portfolio CAGR", "Portfolio Sharpe", "Information Ratio", 
                      "Tracking Error", "Max Drawdown", "Portfolio Beta", 
                      "Annual Turnover", "Hit Rate"]
        
        available_metrics = [m for m in key_metrics if m in comparison_df.columns]
        if available_metrics:
            print(comparison_df[available_metrics].round(4))
        
        # Enhanced strategy recommendation
        if "Information Ratio" in comparison_df.columns:
            best_ir_strategy = comparison_df["Information Ratio"].idxmax()
            best_sharpe_strategy = comparison_df["Portfolio Sharpe"].idxmax()
            
            print(f"\n[FIXED RECOMMENDATIONS]")
            print(f"Best Information Ratio: {best_ir_strategy.upper()} ({comparison_df.loc[best_ir_strategy, 'Information Ratio']:.2f})")
            print(f"Best Sharpe Ratio: {best_sharpe_strategy.upper()} ({comparison_df.loc[best_sharpe_strategy, 'Portfolio Sharpe']:.2f})")
            
            # Beta analysis
            if "Portfolio Beta" in comparison_df.columns:
                avg_beta = comparison_df["Portfolio Beta"].mean()
                print(f"Average Portfolio Beta: {avg_beta:.2f}")
                if avg_beta < 0.8:
                    print("→ Low beta explains underperformance in bull markets")
                elif avg_beta > 1.2:
                    print("→ High beta indicates higher systematic risk")
    
    # 9. FIXED configuration summary
    logger.info(f"\n[COMPLETED] FIXED analysis with comprehensive visualizations finished successfully!")
    print(f"\nFIXED CONFIGURATION USED:")
    print(f"- Universe: {len(CONFIG.TICKERS)} assets")
    print(f"- Benchmark: {CONFIG.BENCHMARK}")
    print(f"- Training window: {CONFIG.WINDOW_YEARS} years")
    print(f"- Rebalancing: {CONFIG.REBALANCE}")
    print(f"- Risk-free rate: {CONFIG.RF:.2%}")
    print(f"- Weight cap: {CONFIG.WEIGHT_CAP:.1%} (FIXED)")
    print(f"- Transaction cost: {CONFIG.TRANSACTION_COST:.2%}")
    print(f"- Top assets selected: {CONFIG.TOP_N}")
    print(f"- Minimum positions: {CONFIG.MIN_POSITIONS}")
    print(f"- Minimum warmup sessions: {CONFIG.MIN_WARMUP_SESSIONS}")
    print(f"- 📊 Visualizations: PRIORITY 1 + PRIORITY 2 (Complete Suite)")
    
# ===================================================================
# REPLACE THE LAST PART OF YOUR FILE WITH THIS:
# ===================================================================

# Set style for professional charts (MOVE THIS TO TOP AFTER IMPORTS)


def plot_cumulative_returns(portfolio_returns: pd.Series, benchmark_returns: pd.Series, 
                          strategy_name: str = "Portfolio", benchmark_name: str = "Benchmark"):
    """
    Plot cumulative returns: Portfolio vs Benchmark
    Essential for evaluating strategy performance over time.
    """
    # Align returns
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        print("❌ No aligned data for cumulative returns plot")
        return
    
    port_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]
    
    # Calculate cumulative returns
    port_cumulative = (1 + port_ret).cumprod()
    bench_cumulative = (1 + bench_ret).cumprod()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot lines with cleaner labels
    ax.plot(port_cumulative.index, port_cumulative.values, 
           linewidth=2.5, label=f'{strategy_name}', color='#2E86AB')
    ax.plot(bench_cumulative.index, bench_cumulative.values, 
           linewidth=2, label='S&P 500', color='#A23B72', alpha=0.8)  # ✅ Simplified label
    
    # Formatting
    ax.set_title(f'Cumulative Returns: {strategy_name} vs S&P 500', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y-1)))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    
    # Add grid and CLEAN legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, 
             shadow=True, framealpha=0.9)  # ✅ Improved legend styling
    
    # Add performance summary box (moved to avoid legend clash)
    port_total = port_cumulative.iloc[-1] - 1
    bench_total = bench_cumulative.iloc[-1] - 1
    excess = port_total - bench_total
    
    textstr = f'{strategy_name}: {port_total:.1%}\nS&P 500: {bench_total:.1%}\nExcess: {excess:+.1%}'
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.9)
    # ✅ Moved to bottom right to avoid legend clash
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print(f"📈 {strategy_name} Total Return: {port_total:.2%}")
    print(f"📊 S&P 500 Total Return: {bench_total:.2%}")
    print(f"🎯 Excess Return: {excess:+.2%}")

def plot_drawdown_chart(portfolio_returns: pd.Series, strategy_name: str = "Portfolio"):
    """
    Plot underwater/drawdown chart showing peak-to-trough losses.
    Critical for understanding risk management and worst-case scenarios.
    """
    if portfolio_returns.empty:
        print("❌ No data for drawdown plot")
        return
    
    # Calculate drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Fill between zero and drawdown (underwater effect) - NO LEGEND NEEDED
    ax.fill_between(drawdown.index, 0, drawdown.values, 
                   color='#E74C3C', alpha=0.7)  # ✅ Removed label
    ax.plot(drawdown.index, drawdown.values, 
           color='#C0392B', linewidth=1.5)
    
    # Formatting
    ax.set_title(f'Drawdown Analysis: {strategy_name}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown', fontsize=12)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    # Calculate key drawdown stats
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # Add statistics box
    textstr = f'Max Drawdown: {max_dd:.2%}\nWorst Date: {max_dd_date.strftime("%Y-%m-%d")}'
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.9)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print(f"📉 Maximum Drawdown: {max_dd:.2%}")
    print(f"📅 Worst Drawdown Date: {max_dd_date.strftime('%Y-%m-%d')}")

def plot_rolling_sharpe(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                       window_months: int = 12, strategy_name: str = "Portfolio"):
    """
    Plot rolling Sharpe ratio to assess strategy stability over time.
    Critical for understanding if outperformance is consistent or sporadic.
    """
    # Align returns
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        print("❌ No aligned data for rolling Sharpe plot")
        return
    
    port_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]
    
    # Calculate rolling window in days
    window_days = window_months * 21  # Approximate trading days per month
    
    if len(port_ret) < window_days:
        print(f"❌ Insufficient data for {window_months}-month rolling Sharpe")
        return
    
    # Calculate rolling Sharpe ratios
    def rolling_sharpe(returns, window, rf=CONFIG.RF):
        rolling_excess = (returns.rolling(window).mean() * 252) - rf
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        return rolling_excess / rolling_vol
    
    port_rolling_sharpe = rolling_sharpe(port_ret, window_days)
    bench_rolling_sharpe = rolling_sharpe(bench_ret, window_days)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot rolling Sharpe ratios with clean labels
    ax.plot(port_rolling_sharpe.index, port_rolling_sharpe.values, 
           linewidth=2.5, label=f'{strategy_name}', color='#2E86AB')
    ax.plot(bench_rolling_sharpe.index, bench_rolling_sharpe.values, 
           linewidth=2, label='S&P 500', color='#A23B72', alpha=0.8)  # ✅ Simplified label
    
    # Add horizontal reference lines with minimal labels
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, linewidth=1)  # ✅ Removed label
    
    # Formatting
    ax.set_title(f'{window_months}-Month Rolling Sharpe Ratio: {strategy_name} vs S&P 500', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    
    # Add grid and CLEAN legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, 
             shadow=True, framealpha=0.9)  # ✅ Only 2 items now
    
    # Calculate statistics
    port_avg_sharpe = port_rolling_sharpe.mean()
    bench_avg_sharpe = bench_rolling_sharpe.mean()
    port_sharpe_std = port_rolling_sharpe.std()
    
    # Add statistics box (moved to avoid legend clash)
    textstr = f'{strategy_name} Avg: {port_avg_sharpe:.2f}\nS&P 500 Avg: {bench_avg_sharpe:.2f}\nStability: {port_sharpe_std:.2f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
    # ✅ Moved to bottom right
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # ✅ Add reference line annotations as text instead of legend
    ax.text(0.02, 0.95, 'Reference Lines:', transform=ax.transAxes, fontsize=9, 
           fontweight='bold', alpha=0.7)
    ax.text(0.02, 0.91, '• Red line: Sharpe = 0', transform=ax.transAxes, fontsize=8, 
           color='red', alpha=0.7)
    ax.text(0.02, 0.87, '• Green line: Sharpe = 1', transform=ax.transAxes, fontsize=8, 
           color='green', alpha=0.7)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Average Rolling Sharpe ({window_months}M): {port_avg_sharpe:.2f}")
    print(f"📈 S&P 500 Rolling Sharpe: {bench_avg_sharpe:.2f}")
    print(f"📏 Sharpe Stability (Std Dev): {port_sharpe_std:.2f}")

def create_performance_dashboard(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                               strategy_name: str = "Portfolio"):
    """
    Create a comprehensive 3-chart performance dashboard.
    This is the main function to call for complete PRIORITY 1 visualization suite.
    """
    print("\n" + "="*80)
    print(f"📊 CREATING PERFORMANCE DASHBOARD FOR: {strategy_name.upper()}")
    print("="*80)
    
    print(f"\n[1/3] 📈 Plotting Cumulative Returns...")
    plot_cumulative_returns(portfolio_returns, benchmark_returns, strategy_name)
    
    print(f"\n[2/3] 📉 Plotting Drawdown Analysis...")
    plot_drawdown_chart(portfolio_returns, strategy_name)
    
    print(f"\n[3/3] 📊 Plotting Rolling Sharpe Ratio...")
    plot_rolling_sharpe(portfolio_returns, benchmark_returns, strategy_name=strategy_name)
    
    print(f"\n✅ Performance dashboard completed for {strategy_name}!")

def plot_portfolio_composition(weights: pd.Series, strategy_name: str = "Portfolio"):
    """
    Create a professional pie chart showing current portfolio composition.
    Essential for understanding concentration and diversification.
    """
    if weights.empty:
        print("❌ No weights data for composition plot")
        return
    
    # Filter out very small weights for cleaner visualization
    min_weight = 0.01  # 1% minimum to show
    significant_weights = weights[weights >= min_weight].copy()
    other_weight = weights[weights < min_weight].sum()
    
    if other_weight > 0:
        significant_weights['Others'] = other_weight
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(significant_weights)))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(significant_weights.values, 
                                     labels=significant_weights.index,
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     colors=colors,
                                     explode=[0.02] * len(significant_weights))  # Slight separation
    
    # Styling
    ax.set_title(f'Portfolio Composition: {strategy_name}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Improve text styling
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    
    # Add statistics box
    n_positions = len(weights[weights > 0.001])
    max_weight = weights.max()
    top3_weight = weights.nlargest(3).sum()
    
    textstr = f'Positions: {n_positions}\nMax Weight: {max_weight:.1%}\nTop 3: {top3_weight:.1%}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(1.2, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Portfolio has {n_positions} active positions")
    print(f"🎯 Largest holding: {weights.index[0]} ({max_weight:.1%})")
    print(f"📈 Top 3 holdings represent {top3_weight:.1%} of portfolio")

def plot_monthly_returns_heatmap(portfolio_returns: pd.Series, strategy_name: str = "Portfolio"):
    """
    Create a monthly returns heatmap showing performance patterns.
    Critical for understanding seasonality and consistency.
    """
    if portfolio_returns.empty:
        print("❌ No returns data for heatmap")
        return
    
    # Calculate monthly returns
    monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    if len(monthly_returns) < 12:
        print("❌ Insufficient data for meaningful heatmap")
        return
    
    # Create year-month matrix
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns.name = 'Returns'
    
    # Pivot to create heatmap structure
    heatmap_data = monthly_returns.groupby([monthly_returns.index.year, 
                                           monthly_returns.index.month]).first().unstack()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=-0.15, vmax=0.15)
    
    # Set ticks and labels
    ax.set_xticks(range(12))
    ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)])
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    
    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(12):
            if not pd.isna(heatmap_data.iloc[i, j]):
                text = ax.text(j, i, f'{heatmap_data.iloc[i, j]:.1%}',
                             ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    # Formatting
    ax.set_title(f'Monthly Returns Heatmap: {strategy_name}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Monthly Return', rotation=270, labelpad=20)
    
    # Calculate statistics
    avg_monthly = monthly_returns.mean()
    best_month = monthly_returns.max()
    worst_month = monthly_returns.min()
    positive_months = (monthly_returns > 0).mean()
    
    # Add statistics box
    textstr = f'Avg Monthly: {avg_monthly:.1%}\nBest Month: {best_month:.1%}\nWorst Month: {worst_month:.1%}\nPositive: {positive_months:.0%}'
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.9)
    ax.text(1.15, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    print(f"📅 Average Monthly Return: {avg_monthly:.2%}")
    print(f"🚀 Best Month: {best_month:.2%}")
    print(f"📉 Worst Month: {worst_month:.2%}")
    print(f"✅ Positive Months: {positive_months:.0%}")

def plot_risk_return_scatter(backtest_results: Dict, current_portfolios: Dict = None):
    """
    Create a risk-return scatter plot comparing strategies.
    Essential for visualizing efficiency frontier and strategy comparison.
    """
    if not backtest_results:
        print("❌ No backtest results for scatter plot")
        return
    
    # Extract data for plotting
    strategies = []
    returns = []
    volatilities = []
    sharpes = []
    
    for strategy, results in backtest_results.items():
        strategies.append(strategy.replace('_', ' ').title())
        returns.append(results.get('Portfolio CAGR', 0))
        volatilities.append(results.get('Portfolio Annual Vol', 0))
        sharpes.append(results.get('Portfolio Sharpe', 0))
    
    # Add benchmark
    if backtest_results:
        first_result = next(iter(backtest_results.values()))
        strategies.append('S&P 500')
        returns.append(first_result.get('Benchmark CAGR', 0))
        volatilities.append(first_result.get('Benchmark Annual Vol', 0))
        sharpes.append(first_result.get('Benchmark Sharpe', 0))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color mapping based on Sharpe ratio
    colors = plt.cm.viridis(np.array(sharpes) / max(sharpes))
    
    # Create scatter plot
    scatter = ax.scatter(volatilities[:-1], returns[:-1], c=colors[:-1], s=200, 
                        alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add benchmark as different marker
    ax.scatter(volatilities[-1], returns[-1], c='red', s=150, marker='s', 
              alpha=0.8, edgecolors='black', linewidth=2, label='Benchmark')
    
    # Add strategy labels
    for i, (strategy, ret, vol) in enumerate(zip(strategies[:-1], returns[:-1], volatilities[:-1])):
        ax.annotate(strategy, (vol, ret), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # Add benchmark label
    ax.annotate('S&P 500', (volatilities[-1], returns[-1]), xytext=(5, 5), 
               textcoords='offset points', fontsize=10, fontweight='bold', color='red')
    
    # Formatting
    ax.set_title('Risk-Return Analysis: Portfolio Strategies vs Benchmark', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Annualized Volatility', fontsize=12)
    ax.set_ylabel('Annualized Return (CAGR)', fontsize=12)
    
    # Format axes as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for Sharpe ratios
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)
    
    # Add efficient frontier line (simplified)
    if len(returns) >= 3:
        # Sort by volatility for line
        sorted_indices = np.argsort(volatilities[:-1])
        sorted_vols = [volatilities[i] for i in sorted_indices]
        sorted_rets = [returns[i] for i in sorted_indices]
        ax.plot(sorted_vols, sorted_rets, '--', alpha=0.5, color='gray', label='Strategy Path')
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    # Find best strategy
    best_sharpe_idx = np.argmax(sharpes[:-1])
    best_strategy = strategies[best_sharpe_idx]
    
    print(f"🎯 Best Risk-Adjusted Strategy: {best_strategy}")
    print(f"📊 Return: {returns[best_sharpe_idx]:.2%}, Vol: {volatilities[best_sharpe_idx]:.2%}, Sharpe: {sharpes[best_sharpe_idx]:.2f}")

def plot_turnover_analysis(strategy_returns: Dict, backtest_results: Dict):
    """
    Create turnover analysis showing transaction costs impact.
    Critical for understanding strategy efficiency and implementation costs.
    """
    if not strategy_returns or not backtest_results:
        print("❌ No data for turnover analysis")
        return
    
    strategies = list(strategy_returns.keys())
    turnovers = []
    returns = []
    transaction_costs = []
    
    for strategy in strategies:
        if strategy in backtest_results:
            annual_turnover = backtest_results[strategy].get('Annual Turnover', 0)
            cagr = backtest_results[strategy].get('Portfolio CAGR', 0)
            
            turnovers.append(annual_turnover)
            returns.append(cagr)
            # Estimate transaction cost impact (turnover * transaction cost rate)
            transaction_costs.append(annual_turnover * CONFIG.TRANSACTION_COST)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Turnover vs Returns
    strategy_names = [s.replace('_', ' ').title() for s in strategies]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars1 = ax1.bar(strategy_names, turnovers, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Annual Turnover by Strategy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Annual Turnover', fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add value labels on bars
    for bar, turnover in zip(bars1, turnovers):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{turnover:.0%}', ha='center', va='bottom', fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Transaction Cost Impact
    net_returns = [ret - tc for ret, tc in zip(returns, transaction_costs)]
    
    x_pos = np.arange(len(strategies))
    width = 0.35
    
    bars2 = ax2.bar(x_pos - width/2, returns, width, label='Gross Return', 
                   color='lightgreen', alpha=0.8, edgecolor='black')
    bars3 = ax2.bar(x_pos + width/2, net_returns, width, label='Net Return (after costs)', 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    
    ax2.set_title('Impact of Transaction Costs', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Annual Return (CAGR)', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(strategy_names)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, ret in zip(bars2, returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{ret:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, net_ret in zip(bars3, net_returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{net_ret:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    avg_turnover = np.mean(turnovers)
    cost_impact = np.mean(transaction_costs)
    
    print(f"📊 Average Annual Turnover: {avg_turnover:.1%}")
    print(f"💰 Average Transaction Cost Impact: {cost_impact:.2%}")
    print(f"🎯 Most Efficient Strategy: {strategy_names[np.argmin(turnovers)]}")

def create_comprehensive_analysis_dashboard(backtest_results: Dict, strategy_returns: Dict, 
                                          current_weights: Dict = None):
    """
    Create a comprehensive analysis dashboard combining all Priority 2 visualizations.
    """
    print("\n" + "="*90)
    print("📊 CREATING COMPREHENSIVE ANALYSIS DASHBOARD - PRIORITY 2")
    print("="*90)
    
    # 1. Portfolio Composition (if current weights available)
    if current_weights:
        print(f"\n[1/4] 📊 Plotting Portfolio Compositions...")
        for strategy, weights in current_weights.items():
            if not weights.empty:
                strategy_display = strategy.replace('_', ' ').title()
                plot_portfolio_composition(weights, f"{strategy_display} Strategy")
    
    # 2. Monthly Returns Heatmaps
    print(f"\n[2/4] 📅 Plotting Monthly Returns Heatmaps...")
    for strategy in strategy_returns:
        if strategy in strategy_returns and not strategy_returns[strategy].empty:
            strategy_display = strategy.replace('_', ' ').title()
            plot_monthly_returns_heatmap(strategy_returns[strategy], f"{strategy_display} Strategy")
    
    # 3. Risk-Return Scatter
    print(f"\n[3/4] 📈 Plotting Risk-Return Analysis...")
    plot_risk_return_scatter(backtest_results)
    
    # 4. Turnover Analysis
    print(f"\n[4/4] 🔄 Plotting Turnover Analysis...")
    plot_turnover_analysis(strategy_returns, backtest_results)
    
    print(f"\n✅ Comprehensive analysis dashboard completed!")

# ===================================================================
# FINAL EXECUTION - ONLY ONE IF __NAME__ == "__MAIN__"
# ===================================================================

if __name__ == "__main__":
    main()