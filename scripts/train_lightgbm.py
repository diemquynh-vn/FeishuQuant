import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import lightgbm as lgb
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
ALPHA_DIR = os.path.join(DATA_OUTPUTS, "alpha_signals_result")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ALPHA_DIR, exist_ok=True)

TRAIN_DAYS = [f'D{i:03d}' for i in range(1, 388)]
VAL_DAYS = [f'D{i:03d}' for i in range(388, 484)]

set_seed(42)

# Features
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
    
    return X_train, y_train, X_val, y_val, features_cs


def ic_eval(y_pred, data):
    """Custom evaluation metric for LightGBM"""
    y_true = data.get_label()
    tmp = pd.DataFrame({'pred': y_pred, 'y': y_true})
    ic, _ = spearmanr(tmp['pred'], tmp['y'])
    return 'spearman_ic', ic, True


def main():
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM")
    print("="*60)
    
    # Load data
    daily = pd.read_parquet(DAILY_FILE)
    daily_feats = build_daily_features(daily)
    
    # Prepare data
    X_train, y_train, X_val, y_val, features = prepare_data(daily_feats)
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    # LightGBM params
    params = {
        'objective': 'regression',
        'metric': 'None',
        'learning_rate': 0.02,
        'num_leaves': 63,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'min_data_in_leaf': 200,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'seed': 42,
        'verbose': -1,
        'num_threads': -1,
    }
    
    # Train
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        feval=ic_eval,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    # Save model
    model_path = os.path.join(MODEL_DIR, 'lightgbm_model.txt')
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    # Predict
    daily_feats['lgb_pred'] = model.predict(daily_feats[features].astype(np.float32))
    
    # Normalize
    daily_feats['lgb_alpha_z'] = daily_feats.groupby('trade_day_id')['lgb_pred'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9))
    daily_feats['lgb_alpha_z'] = daily_feats['lgb_alpha_z'].clip(-3, 3)
    
    # Save alpha
    output = daily_feats[['trade_day_id', 'asset_id', 'lgb_pred', 'lgb_alpha_z']]
    output_path = os.path.join(ALPHA_DIR, 'lightgbm_alpha.csv')
    output.to_csv(output_path, index=False)
    print(f"Alpha saved to: {output_path}")
    
    # Evaluate
    val_data = daily_feats[daily_feats['trade_day_id'].isin(VAL_DAYS)]
    ic, _ = spearmanr(val_data['lgb_pred'], val_data['target'])
    print(f"Validation IC: {ic:.5f}")


if __name__ == "__main__":
    main()