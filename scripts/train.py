import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.memory import get_device, ram_usage
from src.utils.io import save_pickle, clean_memory
from src.datasets.lob_dataset import LOBDataset
from src.models.cnn_lstm_model import CNNLSTMModel
from src.features.lob_features import (
    lob_features, order_flow_features, PRICE_COLS, VOLUME_COLS, 
    LOB_FEAT
)
from src.features.market_features import add_market_features, MARKET_FEATS
from src.features.daily_features import DAILY_FEAT, build_daily_features

# =========================================================
# Configuration - UPDATED PATHS
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_OUTPUTS = os.path.join(PROJECT_ROOT, "data", "outputs")

LOB_FILE = os.path.join(DATA_RAW, "lob_data_in_sample.parquet")
DAILY_FILE = os.path.join(DATA_RAW, "daily_data_in_sample.parquet")

# Output directories
MODEL_DIR = os.path.join(DATA_OUTPUTS, "models")
ALPHA_DIR = os.path.join(DATA_OUTPUTS, "alpha_signals")
PLOT_DIR = os.path.join(DATA_OUTPUTS, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ALPHA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================================================
# Time split
# =========================================================
TRAIN_DAYS = [f'D{i:03d}' for i in range(1, 388)]
VAL_DAYS = [f'D{i:03d}' for i in range(388, 484)]
ALL_DAYS = TRAIN_DAYS + VAL_DAYS

# =========================================================
# Model configuration
# =========================================================
CFG = {
    'seq_len': 24,
    'batch_size': 256,
    'hidden_size': 64,
    'num_layers': 2,
    'cnn_channels': 32,
    'kernel_size': 3,
    'dropout': 0.4,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'n_epochs': 80,
    'patience': 10,
    'clip_norm': 2.0,
}

set_seed(42)
DEVICE = get_device()
ram_usage()

# =========================================================
# Get all LOB column names
# =========================================================
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

# =========================================================
# Feature columns
# =========================================================
FEATURE_COLS = PRICE_COLS + VOLUME_COLS + LOB_FEAT + DAILY_FEAT + MARKET_FEATS
print(f'Total features: {len(FEATURE_COLS)}')


# =========================================================
# Load and process daily data
# =========================================================
def load_and_process_daily():
    print("\n" + "="*60)
    print("Loading daily data...")
    print("="*60)
    
    daily = pd.read_parquet(DAILY_FILE)
    print(f'Raw daily shape: {daily.shape}')
    
    daily = build_daily_features(daily)
    
    DAILY_MERGE = daily[['asset_id', 'trade_day_id'] + DAILY_FEAT + ['target']].copy()
    print(f'DAILY_MERGE shape: {DAILY_MERGE.shape}')
    ram_usage()
    
    return daily, DAILY_MERGE


# =========================================================
# Process LOB data day by day
# =========================================================
def process_lob_data(daily_merge):
    print("\n" + "="*60)
    print("Processing LOB data...")
    print("="*60)
    
    # Scan available days
    day_scan = pd.read_parquet(LOB_FILE, columns=['trade_day_id'])
    unique_days = sorted(day_scan['trade_day_id'].unique())
    days_to_use = [d for d in unique_days if d in set(ALL_DAYS)]
    del day_scan
    clean_memory()
    
    print(f'Processing {len(days_to_use)} days...')
    
    train_X, train_y = [], []
    val_X, val_y = [], []
    train_set, val_set = set(TRAIN_DAYS), set(VAL_DAYS)
    
    for day in tqdm(days_to_use, desc='Processing LOB day-by-day'):
        chunk = pd.read_parquet(LOB_FILE, filters=[('trade_day_id', '==', day)])
        
        if len(chunk) == 0:
            continue
        
        # Remove rows with all missing LOB values
        chunk = chunk[~chunk[_all_lob_cols].isna().all(axis=1)]
        
        if len(chunk) == 0:
            continue
        
        # Fill NaN with 0 for numeric columns only
        numeric_cols = chunk.select_dtypes(include=['float64', 'int64']).columns
        chunk[numeric_cols] = chunk[numeric_cols].fillna(0)
        
        # Convert asset_id to string
        chunk['asset_id'] = chunk['asset_id'].astype(str)
        
        # Fill time column if exists
        if 'time' in chunk.columns:
            chunk['time'] = chunk['time'].fillna('09:40:00')
        
        # Filter valid bid-ask spread
        chunk = chunk[chunk['bid_price_1'] < chunk['ask_price_1']]
        
        if len(chunk) == 0:
            continue
        
        # Keep assets with full sequence
        slot_cnt = chunk.groupby('asset_id')['time'].transform('count')
        chunk = chunk[slot_cnt == CFG['seq_len']].reset_index(drop=True)
        
        if len(chunk) == 0:
            continue
        
        # Feature engineering
                # Feature engineering
        chunk = lob_features(chunk)
        
        # Apply order_flow_features while preserving asset_id
        def apply_order_flow_safe(df_group):
            result = order_flow_features(df_group)
            # Ensure asset_id is preserved
            if 'asset_id' in df_group.columns and 'asset_id' not in result.columns:
                result['asset_id'] = df_group['asset_id'].iloc[0]
            return result
        
        chunk = chunk.groupby('asset_id', group_keys=False).apply(apply_order_flow_safe).reset_index(drop=True)
        chunk = add_market_features(chunk)
        
        # Debug: check if asset_id exists
        if 'asset_id' not in chunk.columns:
            print(f"ERROR: Day {day} - asset_id lost after feature engineering!")
            print(f"Columns: {chunk.columns.tolist()}")
            continue
        
        # Merge daily features
        daily_day = daily_merge[daily_merge['trade_day_id'] == day]
        feat_map = daily_day.set_index('asset_id')[DAILY_FEAT]
        label_map = daily_day.set_index('asset_id')['target']
        
        # Build sequences
        for asset, grp in chunk.groupby('asset_id'):
            grp = grp.sort_values('time').copy()
            
            if len(grp) != CFG['seq_len'] or asset not in feat_map.index:
                continue
            
            # Add daily features
            daily_feat = feat_map.loc[asset]
            for col in DAILY_FEAT:
                grp[col] = daily_feat[col]
            
            y = label_map.get(asset, np.nan)
            if pd.isna(y):
                continue
            
            X = grp[FEATURE_COLS].values.astype(np.float32)
            
            if day in train_set:
                train_X.append(X)
                train_y.append(y)
            else:
                val_X.append(X)
                val_y.append(y)
        
        del chunk
        clean_memory()
    
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    val_X = np.array(val_X)
    val_y = np.array(val_y)
    
    print(f'\nTrain X: {train_X.shape}, y: {train_y.shape}')
    print(f'Val X: {val_X.shape}, y: {val_y.shape}')
    ram_usage()
    
    return train_X, train_y, val_X, val_y


# =========================================================
# Normalize and create datasets
# =========================================================
def normalize_and_create_datasets(train_X, train_y, val_X, val_y):
    print("\n" + "="*60)
    print("Normalizing features...")
    print("="*60)
    
    # Fit scaler on training data
    scaler = RobustScaler()
    train_2d = train_X.reshape(-1, train_X.shape[-1])
    scaler.fit(train_2d)
    
    # Transform
    train_X = scaler.transform(train_2d).reshape(train_X.shape)
    val_2d = val_X.reshape(-1, val_X.shape[-1])
    val_X = scaler.transform(val_2d).reshape(val_X.shape)
    
    # Clean NaN/Inf
    train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
    val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Save scaler
    scaler_path = os.path.join(ALPHA_DIR, 'scaler.pkl')
    save_pickle(scaler, scaler_path)
    print(f'Scaler saved to {scaler_path}')
    
    # Create datasets
    train_ds = LOBDataset(train_X, train_y)
    val_ds = LOBDataset(val_X, val_y)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False,
                            num_workers=2, pin_memory=True)
    
    print(f'Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}')
    ram_usage()
    
    return train_loader, val_loader, len(train_ds), len(val_ds)


# =========================================================
# IC Loss function
# =========================================================
class IC_Loss(nn.Module):
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        pred = pred - pred.mean()
        target = target - target.mean()
        
        pred_std = pred.std(unbiased=False) + 1e-9
        target_std = target.std(unbiased=False) + 1e-9
        
        pred = pred / pred_std
        target = target / target_std
        
        ic = (pred * target).mean()
        return 1.0 - ic


# =========================================================
# Training function
# =========================================================
def train_model(train_loader, val_loader, train_size, val_size):
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)
    
    model = CNNLSTMModel(
        n_feat=len(FEATURE_COLS),
        hidden=CFG['hidden_size'],
        n_layer=CFG['num_layers'],
        c_chan=CFG['cnn_channels'],
        k=CFG['kernel_size'],
        drop=CFG['dropout'],
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model params: {n_params:,}')
    
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=CFG['lr'], 
                                   weight_decay=CFG['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    criterion = IC_Loss()
    
    best_val_ic = -999
    wait = 0
    train_hist, val_hist = [], []
    
    print(f'\nTraining on {DEVICE} | {train_size:,} train | {val_size:,} val')
    print('-' * 65)
    
    for epoch in range(1, CFG['n_epochs'] + 1):
        # Training
        model.train()
        t_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_norm'])
            optimizer.step()
            t_loss += loss.item() * len(y)
        
        t_loss /= train_size
        
        # Validation
        model.eval()
        v_loss = 0.0
        all_pred, all_y = [], []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                all_pred.append(pred.cpu().numpy())
                all_y.append(y.cpu().numpy())
                v_loss += criterion(pred, y).item() * len(y)
        
        v_loss /= val_size
        
        all_pred = np.concatenate(all_pred)
        all_y = np.concatenate(all_y)
        val_ic, _ = spearmanr(all_pred, all_y)
        val_ic = 0 if np.isnan(val_ic) else val_ic
        
        scheduler.step()
        train_hist.append(t_loss)
        val_hist.append(v_loss)
        lr_now = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f'Epoch {epoch:3d}/{CFG["n_epochs"]}', 
              f'train={t_loss:.6f}',
              f'val={v_loss:.6f}',
              f'IC={val_ic:.5f}',
              f'lr={lr_now:.2e}', end='')
        
        # Save best model
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            wait = 0
            model_path = os.path.join(MODEL_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'state': model.state_dict(),
                'val_ic': best_val_ic,
                'cfg': CFG
            }, model_path)
            print('  [SAVED]')
        else:
            wait += 1
            print(f'  (patience {wait}/{CFG["patience"]})')
            if wait >= CFG['patience']:
                print(f'\nEarly stopping at epoch {epoch}')
                break
    
    # Plot learning curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_hist, label='Train', color='steelblue', lw=1.5)
    ax.plot(val_hist, label='Val', color='tomato', lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IC Loss')
    ax.set_title('CNN-LSTM Learning Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'learning_curve.png'), dpi=150)
    plt.show()
    
    print(f'\nBest validation IC: {best_val_ic:.6f}')
    return model, best_val_ic


# =========================================================
# Main function
# =========================================================
def main():
    print("\n" + "="*60)
    print("CNN-LSTM TRAINING PIPELINE")
    print("="*60)
    print(f"Data raw path: {DATA_RAW}")
    print(f"Output path: {DATA_OUTPUTS}")
    
    # Step 1: Load daily data
    daily, daily_merge = load_and_process_daily()
    
    # Step 2: Process LOB data
    train_X, train_y, val_X, val_y = process_lob_data(daily_merge)
    
    # Step 3: Normalize and create datasets
    train_loader, val_loader, train_size, val_size = normalize_and_create_datasets(
        train_X, train_y, val_X, val_y
    )
    
    # Step 4: Train model
    model, best_ic = train_model(train_loader, val_loader, train_size, val_size)
    
    # Step 5: Save final model info
    info_path = os.path.join(MODEL_DIR, 'training_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Best validation IC: {best_ic:.6f}\n")
        f.write(f"Config: {CFG}\n")
        f.write(f"Features: {len(FEATURE_COLS)}\n")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")
    print(f"Alpha signals dir: {ALPHA_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()