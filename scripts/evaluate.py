"""
Evaluation metrics for Feishu Quant Competition

Calculates:
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio (annualized)
- MDD (Maximum Drawdown)
- Composite Score = 0.45*CAGR_pct + 0.30*SR_pct + 0.25*(-MDD)_pct
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def calculate_returns(portfolio_values: pd.Series) -> np.ndarray:
    """Calculate daily returns from portfolio values"""
    return portfolio_values.pct_change().dropna().values


def calculate_cagr(portfolio_values: pd.Series, trading_days_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR)
    
    CAGR = (V_T / V_0)^(1/T) - 1
    where T is measured in years
    """
    V0 = portfolio_values.iloc[0]
    VT = portfolio_values.iloc[-1]
    T = len(portfolio_values) / trading_days_per_year
    
    if V0 <= 0 or T <= 0:
        return 0.0
    
    return (VT / V0) ** (1 / T) - 1


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe Ratio
    
    SR = (mean(R_t - R_f) / std(R_t - R_f)) * sqrt(N)
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(trading_days_per_year)


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Calculate Maximum Drawdown (MDD)
    
    MDD = max((P_t - V_t) / P_t)
    where P_t = running peak up to time t
    """
    cumulative = portfolio_values.values / portfolio_values.iloc[0]
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)  # Negative value


def percentile_rank(values: np.ndarray, current_value: float) -> float:
    """Calculate percentile rank of current_value among values"""
    return np.mean(values <= current_value)


def calculate_composite_score(
    cagr: float,
    sharpe_ratio: float,
    max_drawdown: float,
    all_cagr: np.ndarray,
    all_sharpe: np.ndarray,
    all_mdd: np.ndarray
) -> Dict[str, float]:
    """
    Calculate composite score as defined in competition rules
    
    Score = 0.45 * CAGR_pct + 0.30 * SR_pct + 0.25 * (-MDD)_pct
    
    Where _pct is percentile ranking across all submissions
    """
    # Negative MDD for percentile (lower drawdown = higher rank)
    negative_mdd = -max_drawdown
    all_negative_mdd = -all_mdd
    
    cagr_pct = percentile_rank(all_cagr, cagr)
    sr_pct = percentile_rank(all_sharpe, sharpe_ratio)
    mdd_pct = percentile_rank(all_negative_mdd, negative_mdd)
    
    composite = 0.45 * cagr_pct + 0.30 * sr_pct + 0.25 * mdd_pct
    
    return {
        'cagr': cagr,
        'cagr_percentile': cagr_pct,
        'sharpe_ratio': sharpe_ratio,
        'sharpe_percentile': sr_pct,
        'max_drawdown': max_drawdown,
        'mdd_percentile': mdd_pct,
        'composite_score': composite
    }


def evaluate_portfolio(
    daily_values: pd.DataFrame,
    all_cagr: np.ndarray = None,
    all_sharpe: np.ndarray = None,
    all_mdd: np.ndarray = None
) -> Dict[str, float]:
    """
    Evaluate portfolio performance
    
    Args:
        daily_values: DataFrame with 'portfolio_value' column
        all_cagr: Array of CAGR from all submissions (for percentile)
        all_sharpe: Array of Sharpe ratios from all submissions
        all_mdd: Array of MDD from all submissions
    
    Returns:
        Dictionary with all metrics
    """
    values = daily_values['portfolio_value']
    returns = calculate_returns(values)
    
    cagr = calculate_cagr(values)
    sharpe = calculate_sharpe_ratio(returns)
    mdd = calculate_max_drawdown(values)
    
    print("\n" + "="*60)
    print("PORTFOLIO PERFORMANCE METRICS")
    print("="*60)
    print(f"Initial Capital:    {daily_values['portfolio_value'].iloc[0]:,.2f} RMB")
    print(f"Final Capital:      {daily_values['portfolio_value'].iloc[-1]:,.2f} RMB")
    print(f"Total Return:       {((daily_values['portfolio_value'].iloc[-1] / daily_values['portfolio_value'].iloc[0]) - 1) * 100:.2f}%")
    print(f"CAGR:               {cagr * 100:.2f}%")
    print(f"Sharpe Ratio:       {sharpe:.4f}")
    print(f"Max Drawdown:       {mdd * 100:.2f}%")
    print(f"Holdings Count:     {daily_values['holdings_count'].mean():.1f} avg (min={daily_values['holdings_count'].min()})")
    print("="*60)
    
    # Calculate composite score if benchmark data available
    if all_cagr is not None and all_sharpe is not None and all_mdd is not None:
        scores = calculate_composite_score(cagr, sharpe, mdd, all_cagr, all_sharpe, all_mdd)
        print(f"\nComposite Score:    {scores['composite_score']:.4f}")
        print(f"  - CAGR percentile:   {scores['cagr_percentile']:.3f}")
        print(f"  - Sharpe percentile: {scores['sharpe_percentile']:.3f}")
        print(f"  - MDD percentile:    {scores['mdd_percentile']:.3f}")
        print("="*60)
        return scores
    
    print("="*60)
    return {
        'cagr': cagr,
        'sharpe_ratio': sharpe,
        'max_drawdown': mdd,
        'total_return': (values.iloc[-1] / values.iloc[0] - 1)
    }


def generate_submission_summary(submission_df: pd.DataFrame) -> None:
    """Generate summary statistics for submission file"""
    print("\n" + "="*60)
    print("SUBMISSION FILE SUMMARY")
    print("="*60)
    
    total_orders = len(submission_df)
    buy_orders = submission_df[submission_df['buy_percentage'] > 0]
    sell_orders = submission_df[submission_df['sell_percentage'] > 0]
    
    print(f"Total orders:           {total_orders:,}")
    print(f"Buy orders:             {len(buy_orders):,}")
    print(f"Sell orders:            {len(sell_orders):,}")
    print(f"Unique days:            {submission_df['trade_day_id'].nunique()}")
    print(f"Unique assets:          {submission_df['asset_id'].nunique()}")
    
    if len(buy_orders) > 0:
        print(f"\nBuy percentage stats:")
        print(f"  Mean: {buy_orders['buy_percentage'].mean():.6f}")
        print(f"  Std:  {buy_orders['buy_percentage'].std():.6f}")
        print(f"  Max:  {buy_orders['buy_percentage'].max():.6f}")
    
    if len(sell_orders) > 0:
        print(f"\nSell percentage stats:")
        print(f"  Mean: {sell_orders['sell_percentage'].mean():.6f}")
        print(f"  Std:  {sell_orders['sell_percentage'].std():.6f}")
        print(f"  Max:  {sell_orders['sell_percentage'].max():.6f}")
    
    print("="*60)