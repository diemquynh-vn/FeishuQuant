import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import gc
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr

from src.utils.seed import set_seed
from src.features.daily_features import build_daily_features

# =========================================================
# Configuration
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_OUTPUTS = os.path.join(PROJECT_ROOT, "data", "outputs")

DAILY_FILE = os.path.join(DATA_RAW, "daily_data_in_sample.parquet")
MODEL_DIR = os.path.join(DATA_OUTPUTS, "models")
ALPHA_RESULT_DIR = os.path.join(DATA_OUTPUTS, "alpha_signals_result")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ALPHA_RESULT_DIR, exist_ok=True)

TRAIN_DAYS = [f'D{i:03d}' for i in range(1, 388)]
VAL_DAYS = [f'D{i:03d}' for i in range(388, 484)]

set_seed(42)

# Features (same as LightGBM)
FEATURES = [
    "log_ret", "mom_3", "mom_5", "mom_10", "mom_20",
    "vol_5", "vol_10", "vol_20", "vol_ratio",
    "rsi_14", "atr_ratio", "vol_z20",
    "bb_pct_b", "amihud", "overnight_gap", "hl_range",
    "close_to_high", "macd_signal"
]


def prepare_data(daily_feats):
    """Prepare features with cross-sectional neutralization"""
    # Add cross-sectional features
    for col in FEATURES:
        daily_feats[f'{col}_cs'] = (daily_feats[col] - 
                                     daily_feats.groupby('trade_day_id')[col].transform('mean'))
    
    features_cs = FEATURES + [f'{c}_cs' for c in FEATURES]
    
    # Clip outliers
    for col in ['amihud', 'vol_z20', 'vol_ratio']:
        lo = daily_feats[col].quantile(0.01)
        hi = daily_feats[col].quantile(0.99)
        daily_feats[col] = daily_feats[col].clip(lo, hi)
    
    # Split
    train_mask = daily_feats['trade_day_id'].isin(TRAIN_DAYS)
    val_mask = daily_feats['trade_day_id'].isin(VAL_DAYS)
    
    X_train = daily_feats.loc[train_mask, features_cs].astype(np.float32)
    y_train = daily_feats.loc[train_mask, 'target'].astype(np.float32)
    X_val = daily_feats.loc[val_mask, features_cs].astype(np.float32)
    y_val = daily_feats.loc[val_mask, 'target'].astype(np.float32)
    
    return X_train, y_train, X_val, y_val, features_cs, train_mask, val_mask


def load_lgb_predictions(daily_feats):
    """Load LightGBM predictions if available"""
    lgb_path = os.path.join(ALPHA_RESULT_DIR, 'lightgbm_alpha.csv')
    
    if os.path.exists(lgb_path):
        lgb_df = pd.read_csv(lgb_path)
        lgb_pred = lgb_df.set_index(['trade_day_id', 'asset_id'])['lgb_pred']
        daily_feats = daily_feats.merge(
            lgb_df[['trade_day_id', 'asset_id', 'lgb_pred']], 
            on=['trade_day_id', 'asset_id'], 
            how='left'
        )
        print("Loaded LightGBM predictions")
        return daily_feats, True
    else:
        print("LightGBM predictions not found, training from scratch")
        return daily_feats, False


def main():
    print("\n" + "="*60)
    print("TRAINING RIDGE + ENSEMBLE")
    print("="*60)
    
    # Load data
    daily = pd.read_parquet(DAILY_FILE)
    daily_feats = build_daily_features(daily)
    
    # Load LightGBM predictions
    daily_feats, has_lgb = load_lgb_predictions(daily_feats)
    
    # Prepare data
    X_train, y_train, X_val, y_val, features, train_mask, val_mask = prepare_data(daily_feats)
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    # Train Ridge
    print("\nTraining Ridge...")
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_train, y_train)
    print(f"Best alpha: {ridge.alpha_}")
    
    # Ridge predictions
    daily_feats.loc[train_mask, 'ridge_pred'] = ridge.predict(X_train)
    daily_feats.loc[val_mask, 'ridge_pred'] = ridge.predict(X_val)
    
    # Evaluate Ridge
    ridge_val = daily_feats.loc[val_mask]
    ridge_ic, _ = spearmanr(ridge_val['ridge_pred'], ridge_val['target'])
    print(f"Ridge validation IC: {ridge_ic:.5f}")
    
    # Evaluate LightGBM if available
    if has_lgb:
        lgb_val = daily_feats.loc[val_mask].dropna(subset=['lgb_pred'])
        lgb_ic, _ = spearmanr(lgb_val['lgb_pred'], lgb_val['target'])
        print(f"LightGBM validation IC: {lgb_ic:.5f}")
        
        # Weighted ensemble based on IC
        total_ic = max(ridge_ic, 0) + max(lgb_ic, 0) + 1e-9
        w_ridge = max(ridge_ic, 0) / total_ic
        w_lgb = max(lgb_ic, 0) / total_ic
        
        print(f"\nEnsemble weights: Ridge={w_ridge:.3f}, LGB={w_lgb:.3f}")
        
        # Ensemble prediction
        daily_feats['m2_score'] = (w_ridge * daily_feats['ridge_pred'].fillna(0) + 
                                    w_lgb * daily_feats['lgb_pred'].fillna(0))
    else:
        daily_feats['m2_score'] = daily_feats['ridge_pred']
    
    # Cross-sectional normalization
    daily_feats['m2_score_z'] = daily_feats.groupby('trade_day_id')['m2_score'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9))
    daily_feats['m2_score_z'] = daily_feats['m2_score_z'].clip(-3, 3)
    
    # Save alpha
    output = daily_feats[['trade_day_id', 'asset_id', 'm2_score', 'm2_score_z']]
    output_path = os.path.join(ALPHA_RESULT_DIR, 'member2_alpha.csv')
    output.to_csv(output_path, index=False)
    print(f"\nalpha saved to: {output_path}")
    
    # Final evaluation on validation
    val_final = daily_feats.loc[val_mask]
    final_ic, _ = spearmanr(val_final['m2_score'], val_final['target'])
    print(f"Final ensemble IC on validation: {final_ic:.5f}")
    
    # Save Ridge model
    import pickle
    ridge_path = os.path.join(MODEL_DIR, 'ridge_model.pkl')
    with open(ridge_path, 'wb') as f:
        pickle.dump(ridge, f)
    print(f"Ridge model saved to: {ridge_path}")
    
    # Cleanup
    gc.collect()


if __name__ == "__main__":
    main()