import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr, zscore
from tqdm import tqdm

from config import DAILY_FILE, MODEL_DIR, ALPHA_RESULT_DIR, TRAIN_DAYS, VAL_DAYS
from src.utils.seed import set_seed
from src.features.daily_features import build_daily_features

set_seed(42)

# Features
FEATURES = [
    "log_ret", "mom_5", "mom_20", "vol_10", "vol_20",
    "rsi_14", "atr_ratio", "vol_z20"
]


def prepare_features(df):
    # Cross-sectional neutralization
    for col in FEATURES:
        df[f'{col}_cs'] = df[col] - df.groupby('trade_day_id')[col].transform('mean')
    
    features_cs = FEATURES + [f'{c}_cs' for c in FEATURES]
    
    train_mask = df['trade_day_id'].isin(TRAIN_DAYS)
    val_mask = df['trade_day_id'].isin(VAL_DAYS)
    
    X_train = df.loc[train_mask, features_cs].astype(np.float32)
    y_train = df.loc[train_mask, 'target'].astype(np.float32)
    X_val = df.loc[val_mask, features_cs].astype(np.float32)
    y_val = df.loc[val_mask, 'target'].astype(np.float32)
    
    return X_train, y_train, X_val, y_val, features_cs, train_mask, val_mask


def ic_eval(y_pred, data):
    y_true = data.get_label()
    ic = spearmanr(y_pred, y_true)[0]
    return 'ic', ic, True


def train_lightgbm(X_train, y_train, X_val, y_val):
    print("\nTraining LightGBM...")
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    params = {
        'objective': 'regression', 'metric': 'None',
        'learning_rate': 0.02, 'num_leaves': 63,
        'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 1,
        'min_data_in_leaf': 200, 'lambda_l1': 1.0, 'lambda_l2': 1.0,
        'seed': 42, 'verbose': -1, 'num_threads': -1,
    }
    
    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dval], feval=ic_eval,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    model.save_model(os.path.join(MODEL_DIR, 'lightgbm_model.txt'))
    return model


def train_ridge(X_train, y_train):
    print("\nTraining Ridge...")
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_train, y_train)
    print(f"Best alpha: {ridge.alpha_}")
    return ridge


def get_erc_weights(vols):
    vols = np.asarray(vols)
    inv_vols = 1.0 / (vols + 1e-9)
    return inv_vols / inv_vols.sum()


def build_risk_parity_portfolio(alpha_df, vol_df, top_k=20):
    merged = alpha_df.merge(vol_df, on=['trade_day_id', 'asset_id'], how='left')
    merged['vol_20'] = merged['vol_20'].fillna(0.02)
    
    records = []
    for day, grp in tqdm(merged.groupby('trade_day_id'), desc='Risk Parity'):
        top_stocks = grp.nlargest(top_k, 'alpha_z')
        if len(top_stocks) < 10:
            continue
        
        weights = get_erc_weights(top_stocks['vol_20'].values)
        
        for i, row in enumerate(top_stocks.itertuples()):
            records.append({
                'trade_day_id': day,
                'asset_id': row.asset_id,
                'buy_percentage': weights[i],
                'sell_percentage': 0.0,
                'alpha_score': row.alpha_z,
            })
    
    return pd.DataFrame(records)


def main():
    print("\n" + "="*60)
    print("PART 2: DAILY MODELS (LightGBM + Ridge + Risk Parity)")
    print("="*60)
    
    # Load data
    daily = pd.read_parquet(DAILY_FILE)
    daily = build_daily_features(daily)
    
    # Prepare features
    X_train, y_train, X_val, y_val, features, train_mask, val_mask = prepare_features(daily)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Train LightGBM
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    
    # Train Ridge
    ridge_model = train_ridge(X_train, y_train)
    
    # Predict
    daily.loc[train_mask, 'lgb_pred'] = lgb_model.predict(X_train)
    daily.loc[val_mask, 'lgb_pred'] = lgb_model.predict(X_val)
    daily.loc[train_mask, 'ridge_pred'] = ridge_model.predict(X_train)
    daily.loc[val_mask, 'ridge_pred'] = ridge_model.predict(X_val)
    
    # Calculate IC weights
    val_data = daily.loc[val_mask]
    lgb_ic = max(spearmanr(val_data['lgb_pred'], val_data['target'])[0], 0)
    ridge_ic = max(spearmanr(val_data['ridge_pred'], val_data['target'])[0], 0)
    
    total_ic = lgb_ic + ridge_ic + 1e-9
    w_lgb, w_ridge = lgb_ic / total_ic, ridge_ic / total_ic
    
    print(f"\nLGB IC: {lgb_ic:.5f} | Ridge IC: {ridge_ic:.5f}")
    print(f"Weights: LGB={w_lgb:.3f}, Ridge={w_ridge:.3f}")
    
    # Ensemble
    daily['m2_score'] = w_lgb * daily['lgb_pred'] + w_ridge * daily['ridge_pred']
    daily['m2_score_z'] = daily.groupby('trade_day_id')['m2_score'].transform(
        lambda x: zscore(x, ddof=0)).clip(-3, 3)
    
    # Save member2 alpha
    alpha_df = daily[['trade_day_id', 'asset_id', 'm2_score', 'm2_score_z']]
    alpha_df.to_csv(os.path.join(ALPHA_RESULT_DIR, 'member2_alpha.csv'), index=False)
    print(f"\nSaved member2 alpha to {ALPHA_RESULT_DIR}")
    
    # Build risk parity portfolio
    vol_df = daily[['trade_day_id', 'asset_id', 'vol_20']]
    portfolio_df = build_risk_parity_portfolio(alpha_df.rename(columns={'m2_score_z': 'alpha_z'}), vol_df)
    
    # Save submission
    portfolio_df[['trade_day_id', 'asset_id', 'buy_percentage', 'sell_percentage']].to_csv(
        os.path.join(ALPHA_RESULT_DIR, 'member2_submission.csv'), index=False)
    
    print(f"Portfolio saved: {len(portfolio_df):,} rows")
    print("="*60)


if __name__ == "__main__":
    main()