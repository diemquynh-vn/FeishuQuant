"""
Out-of-Sample (OOS) Inference for Feishu Quant Competition

This script generates submission for OOS period (D485 to D726)
IMPORTANT: OOS data is used ONLY for inference, NOT for training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config_oos import (
    LOB_OOS_FILE, DAILY_OOS_FILE, OOS_OUTPUT_DIR, MODEL_DIR, ALPHA_DIR,
    OOS_DAYS, SELL_MODE, TEAM_ID, SUBMISSION_FILENAME, TOP_K, BACKTEST_PARAMS
)
from src.utils.seed import set_seed
from src.utils.memory import get_device, clean_memory
from src.utils.io import load_pickle
from src.models.cnn_lstm_model import CNNLSTMModel
from src.features.lob_features import (
    lob_features, order_flow_features, PRICE_COLS, VOLUME_COLS, LOB_FEAT
)
from src.features.market_features import add_market_features, MARKET_FEATS
from src.features.daily_features import DAILY_FEAT, build_daily_features

set_seed(42)
DEVICE = get_device()


def get_all_lob_columns():
    cols = []
    for side in ['ask', 'bid']:
        for i in range(1, 11):
            cols.append(f'{side}_price_{i}')
    for side in ['ask', 'bid']:
        for i in range(1, 11):
            cols.append(f'{side}_volume_{i}')
    return cols


_all_lob_cols = get_all_lob_columns()
FEATURE_COLS = PRICE_COLS + VOLUME_COLS + LOB_FEAT + DAILY_FEAT + MARKET_FEATS


def load_oos_daily():
    """Load OOS daily data (D485-D726)"""
    print("\n" + "="*60)
    print("Loading OOS Daily Data")
    print("="*60)
    
    if not os.path.exists(DAILY_OOS_FILE):
        raise FileNotFoundError(f"OOS daily file not found: {DAILY_OOS_FILE}")
    
    daily = pd.read_parquet(DAILY_OOS_FILE)
    daily = build_daily_features(daily)
    
    # Filter only OOS days
    daily = daily[daily['trade_day_id'].isin(OOS_DAYS)]
    
    daily_merge = daily[['asset_id', 'trade_day_id'] + DAILY_FEAT].copy()
    print(f"OOS daily shape: {daily_merge.shape}")
    print(f"OOS days: {daily['trade_day_id'].nunique()}")
    
    return daily, daily_merge


def load_oos_lob_data():
    """Load OOS LOB data day by day"""
    print("\n" + "="*60)
    print("Processing OOS LOB Data")
    print("="*60)
    
    if not os.path.exists(LOB_OOS_FILE):
        raise FileNotFoundError(f"OOS LOB file not found: {LOB_OOS_FILE}")
    
    # Scan available days
    day_scan = pd.read_parquet(LOB_OOS_FILE, columns=['trade_day_id'])
    available_days = sorted(day_scan['trade_day_id'].unique())
    days_to_use = [d for d in available_days if d in set(OOS_DAYS)]
    del day_scan
    clean_memory()
    
    print(f"Processing {len(days_to_use)} OOS days")
    return days_to_use


def generate_oos_alpha(daily_merge, days_to_use):
    """Generate alpha scores for OOS period using trained model"""
    print("\n" + "="*60)
    print("Generating OOS Alpha Signals")
    print("="*60)
    
    # Load scaler and model
    scaler = load_pickle(os.path.join(ALPHA_DIR, 'scaler.pkl'))
    checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_model.pt'), map_location=DEVICE)
    
    model = CNNLSTMModel(
        n_feat=len(FEATURE_COLS),
        hidden=64,
        n_layer=2,
        c_chan=32,
        k=3,
        drop=0.4,
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['state'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}, IC={checkpoint['val_ic']:.5f}")
    
    records = []
    
    for day in tqdm(days_to_use, desc='OOS Alpha'):
        chunk = pd.read_parquet(LOB_OOS_FILE, filters=[('trade_day_id', '==', day)])
        
        if len(chunk) == 0:
            continue
        
        chunk = chunk[~chunk[_all_lob_cols].isna().all(axis=1)]
        chunk[_all_lob_cols] = chunk[_all_lob_cols].fillna(0)
        chunk = chunk[chunk['bid_price_1'] < chunk['ask_price_1']]
        
        slot_cnt = chunk.groupby('asset_id')['time'].transform('count')
        chunk = chunk[slot_cnt == 24].reset_index(drop=True)
        
        if len(chunk) == 0:
            continue
        
        chunk = lob_features(chunk)
        chunk = chunk.groupby('asset_id', group_keys=False).apply(order_flow_features).reset_index(drop=True)
        chunk = add_market_features(chunk)
        
        daily_day = daily_merge[daily_merge['trade_day_id'] == day]
        feat_map = daily_day.set_index('asset_id')[DAILY_FEAT]
        
        for asset, grp in chunk.groupby('asset_id'):
            grp = grp.sort_values('time').reset_index(drop=True)
            
            if len(grp) != 24 or asset not in feat_map.index:
                continue
            
            for col in DAILY_FEAT:
                grp[col] = feat_map.loc[asset, col]
            
            # Normalize price and volume
            mid_price = grp['mid_price'].values + 1e-9
            for col in PRICE_COLS:
                grp[col] = grp[col] / mid_price
            
            vol_base = grp[VOLUME_COLS].sum(axis=1).rolling(5, min_periods=1).mean().values + 1e-9
            for col in VOLUME_COLS:
                grp[col] = grp[col] / vol_base
            
            X = grp[FEATURE_COLS].values.astype(np.float32)
            X = np.nan_to_num(X)
            X = scaler.transform(X)
            X = torch.from_numpy(X).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                score = model(X).item()
            
            records.append({
                'trade_day_id': day,
                'asset_id': asset,
                'alpha_score': score,
            })
        
        del chunk
        clean_memory()
    
    alpha_df = pd.DataFrame(records)
    
    # Cross-sectional normalization per day
    alpha_df['alpha_z'] = alpha_df.groupby('trade_day_id')['alpha_score'].transform(
        lambda x: (x - x.median()) / ((x - x.median()).abs().median() + 1e-9)).clip(-3, 3)
    alpha_df['alpha_rank'] = alpha_df.groupby('trade_day_id')['alpha_score'].rank(pct=True)
    
    print(f"Generated {len(alpha_df):,} alpha signals for OOS")
    return alpha_df


def build_oos_portfolio(alpha_df):
    """Build portfolio for OOS period"""
    print("\n" + "="*60)
    print("Building OOS Portfolio")
    print("="*60)
    
    submission = []
    
    for day, grp in alpha_df.groupby('trade_day_id'):
        grp = grp.sort_values('alpha_rank')
        
        # Select top K stocks
        top_stocks = grp.tail(TOP_K).copy()
        
        if len(top_stocks) < 10:
            print(f"Warning: Day {day} only has {len(top_stocks)} stocks (min required: 10)")
            continue
        
        # Equal weight or rank-based weight
        weights = top_stocks['alpha_rank'].values
        weights = weights / weights.sum()
        
        for i, row in enumerate(top_stocks.itertuples()):
            submission.append({
                'trade_day_id': day,
                'asset_id': row.asset_id,
                'buy_percentage': weights[i],
                'sell_percentage': 0.0  # No sell for OOS (sell logic handled by backtest)
            })
    
    submission_df = pd.DataFrame(submission)
    print(f"Portfolio built: {len(submission_df):,} orders across {submission_df['trade_day_id'].nunique()} days")
    
    return submission_df


def save_oos_submission(submission_df):
    """Save OOS submission file"""
    # Sort by trade_day_id
    submission_df = submission_df.sort_values('trade_day_id')
    
    # Remove zero rows
    submission_df = submission_df[(submission_df['buy_percentage'] != 0) | (submission_df['sell_percentage'] != 0)]
    
    # Ensure buy_percentage and sell_percentage have at most 6 decimal places
    submission_df['buy_percentage'] = submission_df['buy_percentage'].round(6)
    submission_df['sell_percentage'] = submission_df['sell_percentage'].round(6)
    
    output_path = os.path.join(OOS_OUTPUT_DIR, SUBMISSION_FILENAME)
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n" + "="*60)
    print("OOS SUBMISSION SAVED")
    print("="*60)
    print(f"File: {output_path}")
    print(f"Rows: {len(submission_df):,}")
    print(f"Days: {submission_df['trade_day_id'].nunique()}")
    print(f"Assets: {submission_df['asset_id'].nunique()}")
    print(f"Sell Mode: {SELL_MODE}")
    print("="*60)
    
    return output_path


def main():
    print("\n" + "="*70)
    print("FEISHU QUANT COMPETITION - OOS INFERENCE")
    print("="*70)
    print(f"OOS Period: {OOS_DAYS[0]} to {OOS_DAYS[-1]}")
    print(f"Sell Mode: {SELL_MODE}")
    print(f"Team ID: {TEAM_ID}")
    print("="*70)
    print("\nIMPORTANT: OOS data is used ONLY for inference, NOT for training")
    print("="*70)
    
    # Check if OOS files exist
    if not os.path.exists(LOB_OOS_FILE):
        print(f"\nERROR: OOS LOB file not found: {LOB_OOS_FILE}")
        print("Please wait for the OOS data release on May 28, 2026")
        return
    
    if not os.path.exists(DAILY_OOS_FILE):
        print(f"\nERROR: OOS daily file not found: {DAILY_OOS_FILE}")
        print("Please wait for the OOS data release on May 28, 2026")
        return
    
    # Load OOS data
    daily, daily_merge = load_oos_daily()
    days_to_use = load_oos_lob_data()
    
    # Generate alpha
    alpha_df = generate_oos_alpha(daily_merge, days_to_use)
    
    # Build portfolio
    submission_df = build_oos_portfolio(alpha_df)
    
    # Validate minimum holdings
    daily_counts = submission_df.groupby('trade_day_id')['asset_id'].nunique()
    if (daily_counts < 10).any():
        print(f"\nWARNING: Some days have < 10 stocks:")
        print(daily_counts[daily_counts < 10])
    
    # Save submission
    output_path = save_oos_submission(submission_df)
    
    print("\n" + "="*70)
    print("OOS INFERENCE COMPLETE")
    print("="*70)
    print(f"Submit file: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()