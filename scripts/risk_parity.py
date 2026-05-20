import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.seed import set_seed

# =========================================================
# Configuration
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_OUTPUTS = os.path.join(PROJECT_ROOT, "data", "outputs")

DAILY_FILE = os.path.join(DATA_RAW, "daily_data_in_sample.parquet")
MEMBER2_PATH = os.path.join(DATA_OUTPUTS, "alpha_signals_result", "member2_alpha.csv")
OUT_DIR = os.path.join(DATA_OUTPUTS, "alpha_signals_result")

os.makedirs(OUT_DIR, exist_ok=True)

set_seed(42)


def get_erc_weights(vols):
    """Equal Risk Contribution (Inverse volatility) weights"""
    vols = np.asarray(vols)
    inv_vols = 1.0 / (vols + 1e-9)
    weights = inv_vols / inv_vols.sum()
    return weights


def load_daily_volatility():
    """Load daily volatility data for risk parity"""
    daily = pd.read_parquet(DAILY_FILE)
    daily['close_adj'] = daily['close'] * daily['adj_factor']
    
    # Calculate daily returns and volatility
    daily = daily.sort_values(['asset_id', 'trade_day_id']).reset_index(drop=True)
    g = daily.groupby('asset_id')
    daily['daily_ret'] = g['close_adj'].pct_change()
    daily['vol_20'] = g['daily_ret'].transform(lambda x: x.rolling(20, min_periods=5).std())
    daily['vol_20'] = daily['vol_20'].fillna(daily.groupby('trade_day_id')['vol_20'].transform('median'))
    daily['vol_20'] = daily['vol_20'].fillna(0.02)  # Default 2% vol
    
    return daily[['trade_day_id', 'asset_id', 'vol_20']]


def main():
    print("\n" + "="*60)
    print("RISK PARITY PORTFOLIO (Part 2)")
    print("="*60)
    
    # Load member2 alpha
    alpha_df = pd.read_csv(MEMBER2_PATH)
    print(f"Member2 alpha shape: {alpha_df.shape}")
    
    # Load volatility
    vol_df = load_daily_volatility()
    print(f"Volatility data shape: {vol_df.shape}")
    
    # Merge alpha with volatility
    merged = alpha_df.merge(vol_df, on=['trade_day_id', 'asset_id'], how='left')
    merged['vol_20'] = merged['vol_20'].fillna(0.02)
    
    # Build risk parity portfolio
    portfolio_records = []
    TOP_K = 20  # Select top 20 stocks by alpha
    
    for day, grp in tqdm(merged.groupby('trade_day_id'), desc="Building portfolio"):
        # Select top 20 by alpha
        top_stocks = grp.nlargest(TOP_K, 'm2_score_z').copy()
        
        if len(top_stocks) < 10:
            continue
        
        # Risk parity weights (inverse volatility)
        vols = top_stocks['vol_20'].values
        vols = np.clip(vols, 1e-6, None)
        weights = get_erc_weights(vols)
        
        for i, row in enumerate(top_stocks.itertuples()):
            portfolio_records.append({
                'trade_day_id': day,
                'asset_id': row.asset_id,
                'buy_percentage': weights[i],
                'sell_percentage': 0.0,
                'alpha_score': row.m2_score_z,
                'vol_20': row.vol_20
            })
    
    # Create portfolio dataframe
    portfolio_df = pd.DataFrame(portfolio_records)
    
    # Verify weights sum to 1 each day
    weight_check = portfolio_df.groupby('trade_day_id')['buy_percentage'].sum()
    print(f"Mean daily weight sum: {weight_check.mean():.6f}")
    
    # Save portfolio
    portfolio_path = os.path.join(OUT_DIR, 'risk_parity_portfolio.csv')
    portfolio_df.to_csv(portfolio_path, index=False)
    print(f"\nPortfolio saved to: {portfolio_path}")
    print(f"Portfolio shape: {portfolio_df.shape}")
    print(f"Number of days: {portfolio_df['trade_day_id'].nunique()}")
    
    # Also save in submission format for Part 2
    submission = portfolio_df[['trade_day_id', 'asset_id', 'buy_percentage', 'sell_percentage']]
    submission_path = os.path.join(OUT_DIR, 'member2_submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()