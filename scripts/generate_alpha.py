import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import spearmanr

from src.utils.seed import set_seed
from src.utils.memory import get_device, ram_usage
from src.utils.io import clean_memory, load_pickle
from src.models.cnn_lstm_model import CNNLSTMModel
from src.features.lob_features import (
    lob_features, order_flow_features, PRICE_COLS, VOLUME_COLS, LOB_FEAT
)
from src.features.market_features import add_market_features, MARKET_FEATS
from src.features.daily_features import DAILY_FEAT

# =========================================================
# Configuration
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_OUTPUTS = os.path.join(PROJECT_ROOT, "data", "outputs")

LOB_FILE = os.path.join(DATA_RAW, "lob_data_in_sample.parquet")
DAILY_FILE = os.path.join(DATA_RAW, "daily_data_in_sample.parquet")

MODEL_DIR = os.path.join(DATA_OUTPUTS, "models")
ALPHA_DIR = os.path.join(DATA_OUTPUTS, "alpha_signals")
os.makedirs(ALPHA_DIR, exist_ok=True)

# Load config from training
TRAIN_DAYS = [f'D{i:03d}' for i in range(1, 388)]
VAL_DAYS = [f'D{i:03d}' for i in range(388, 484)]
ALL_DAYS = TRAIN_DAYS + VAL_DAYS

CFG = {
    'seq_len': 24,
    'hidden_size': 64,
    'num_layers': 2,
    'cnn_channels': 32,
    'kernel_size': 3,
    'dropout': 0.4,
}

set_seed(42)
DEVICE = get_device()


def get_all_lob_columns():
    _all_lob = []
    for side in ['ask', 'bid']:
        for i in range(1, 11):
            _all_lob.append(f'{side}_price_{i}')
    for side in ['ask', 'bid']:
        for i in range(1, 11):
            _all_lob.append(f'{side}_volume_{i}')
    return _all_lob


_all_lob_cols = get_all_lob_columns()
FEATURE_COLS = PRICE_COLS + VOLUME_COLS + LOB_FEAT + DAILY_FEAT + MARKET_FEATS


def load_daily_data():
    """Load daily data with features and target"""
    daily = pd.read_parquet(DAILY_FILE)
    daily['close_adj'] = daily['close'] * daily['adj_factor']
    daily = daily.sort_values(['asset_id', 'trade_day_id']).reset_index(drop=True)
    
    g = daily.groupby('asset_id')
    daily['log_ret'] = np.log(daily['close_adj'] / (g['close_adj'].shift(1) + 1e-9))
    daily['mom_5'] = g['close_adj'].transform(lambda x: x.pct_change(5))
    daily['vol_10'] = g['log_ret'].transform(lambda x: x.rolling(10, min_periods=3).std())
    daily['rsi_14'] = daily.groupby('asset_id')['close_adj'].transform(
        lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).rolling(14, min_periods=1).mean() / 
                             (-x.diff().clip(upper=0)).rolling(14, min_periods=1).mean() + 1e-9))))
    
    prev_c = g['close_adj'].shift(1)
    tr = pd.concat([daily['high'] - daily['low'],
                    (daily['high'] - prev_c).abs(),
                    (daily['low'] - prev_c).abs()], axis=1).max(axis=1)
    daily['atr_ratio'] = (tr.groupby(daily['asset_id']).transform(
        lambda x: x.rolling(14, min_periods=1).mean()) / (daily['close_adj'] + 1e-9))
    
    daily['vol_z20'] = g['volume'].transform(
        lambda x: (x - x.rolling(20, min_periods=5).mean()) / (x.rolling(20, min_periods=5).std() + 1e-9))
    
    daily['next_close'] = g['close_adj'].shift(-1)
    daily['daily_ret'] = (daily['next_close'] - daily['close_adj']) / (daily['close_adj'] + 1e-9)
    daily.fillna(0, inplace=True)
    
    DAILY_MERGE = daily[['asset_id', 'trade_day_id'] + DAILY_FEAT].copy()
    return daily, DAILY_MERGE


def main():
    print("\n" + "="*60)
    print("GENERATING ALPHA SIGNALS (CNN-LSTM)")
    print("="*60)
    
    # Load scaler and model
    scaler_path = os.path.join(ALPHA_DIR, 'scaler.pkl')
    model_path = os.path.join(MODEL_DIR, 'best_model.pt')
    
    scaler = load_pickle(scaler_path)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    model = CNNLSTMModel(
        n_feat=len(FEATURE_COLS),
        hidden=CFG['hidden_size'],
        n_layer=CFG['num_layers'],
        c_chan=CFG['cnn_channels'],
        k=CFG['kernel_size'],
        drop=CFG['dropout'],
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['state'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}, IC={checkpoint['val_ic']:.5f}")
    
    # Load daily data
    daily, daily_merge = load_daily_data()
    
    # Scan available days
    day_scan = pd.read_parquet(LOB_FILE, columns=['trade_day_id'])
    unique_days = sorted(day_scan['trade_day_id'].unique())
    days_to_use = [d for d in unique_days if d in set(ALL_DAYS)]
    del day_scan
    clean_memory()
    
    records = []
    
    for day in tqdm(days_to_use, desc='Generating alpha'):
        chunk = pd.read_parquet(LOB_FILE, filters=[('trade_day_id', '==', day)])
        
        if len(chunk) == 0:
            continue
        
        chunk = chunk[~chunk[_all_lob_cols].isna().all(axis=1)]
        chunk[_all_lob_cols] = chunk[_all_lob_cols].fillna(0)
        chunk = chunk[chunk['bid_price_1'] < chunk['ask_price_1']]
        
        slot_cnt = chunk.groupby('asset_id')['time'].transform('count')
        chunk = chunk[slot_cnt == CFG['seq_len']].reset_index(drop=True)
        
        if len(chunk) == 0:
            continue
        
        chunk = lob_features(chunk)
        chunk = chunk.groupby('asset_id', group_keys=False).apply(order_flow_features).reset_index(drop=True)
        chunk = add_market_features(chunk)
        
        daily_day = daily_merge[daily_merge['trade_day_id'] == day]
        feat_map = daily_day.set_index('asset_id')[DAILY_FEAT]
        actual_map = daily[daily['trade_day_id'] == day].set_index('asset_id')['daily_ret']
        
        for asset, grp in chunk.groupby('asset_id'):
            grp = grp.sort_values('time').reset_index(drop=True)
            
            if len(grp) != CFG['seq_len'] or asset not in feat_map.index:
                continue
            
            daily_feat = feat_map.loc[asset]
            for col in DAILY_FEAT:
                grp[col] = daily_feat[col]
            
            # Normalize price and volume
            mid_price = grp['mid_price'].values + 1e-9
            for col in PRICE_COLS:
                grp[col] = grp[col] / mid_price
            
            vol_base = grp[VOLUME_COLS].sum(axis=1).rolling(5, min_periods=1).mean().values + 1e-9
            for col in VOLUME_COLS:
                grp[col] = grp[col] / vol_base
            
            X = grp[FEATURE_COLS].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = scaler.transform(X)
            X = torch.from_numpy(X).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                score = model(X).item()
            
            actual = actual_map.get(asset, np.nan)
            records.append({
                'trade_day_id': day,
                'asset_id': asset,
                'alpha_score': score,
                'actual_return': actual
            })
        
        del chunk
        clean_memory()
    
    # Build alpha dataframe
    alpha_df = pd.DataFrame(records)
    
    # Cross-sectional normalization
    alpha_df['alpha_z'] = alpha_df.groupby('trade_day_id')['alpha_score'].transform(
        lambda x: (x - x.median()) / ((x - x.median()).abs().median() + 1e-9))
    alpha_df['alpha_z'] = alpha_df['alpha_z'].clip(-3, 3)
    alpha_df['alpha_rank'] = alpha_df.groupby('trade_day_id')['alpha_score'].rank(pct=True)
    
    # Save
    output_path = os.path.join(ALPHA_DIR, 'member1_alpha.csv')
    alpha_df.to_csv(output_path, index=False)
    
    print(f"\nAlpha signals generated: {len(alpha_df):,}")
    print(f"Saved to: {output_path}")
    
    # Evaluate validation IC
    val_alpha = alpha_df[alpha_df['trade_day_id'].isin(VAL_DAYS)].dropna(subset=['actual_return'])
    daily_ics = []
    for day, grp in val_alpha.groupby('trade_day_id'):
        if len(grp) >= 5:
            ic, _ = spearmanr(grp['alpha_score'], grp['actual_return'])
            if not np.isnan(ic):
                daily_ics.append(ic)
    
    if daily_ics:
        mean_ic = np.mean(daily_ics)
        std_ic = np.std(daily_ics)
        icir = mean_ic / (std_ic + 1e-9)
        print(f"\nValidation IC: {mean_ic:+.4f} (std={std_ic:.4f}, IR={icir:.4f})")


if __name__ == "__main__":
    main()