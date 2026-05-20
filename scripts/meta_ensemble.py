import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, zscore
from sklearn.linear_model import RidgeCV

from src.utils.seed import set_seed

# =========================================================
# Configuration
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUTPUTS = os.path.join(PROJECT_ROOT, "data", "outputs")

MEMBER1_PATH = os.path.join(DATA_OUTPUTS, "alpha_signals", "member1_alpha.csv")
MEMBER2_PATH = os.path.join(DATA_OUTPUTS, "alpha_signals_result", "member2_alpha.csv")
OUT_DIR = os.path.join(DATA_OUTPUTS, "member3_result")

os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = [f'D{i:03d}' for i in range(1, 388)]
VAL_DAYS = [f'D{i:03d}' for i in range(388, 484)]

set_seed(42)


def load_and_merge():
    """Load member1 and member2 alpha"""
    m1 = pd.read_csv(MEMBER1_PATH)
    m2 = pd.read_csv(MEMBER2_PATH)
    
    print(f"Member1 shape: {m1.shape}")
    print(f"Member2 shape: {m2.shape}")
    
    m1 = m1[['trade_day_id', 'asset_id', 'alpha_score', 'alpha_z']].copy()
    m1.columns = ['trade_day_id', 'asset_id', 'm1_alpha', 'm1_alpha_z']
    
    m2 = m2[['trade_day_id', 'asset_id', 'm2_score', 'm2_score_z']].copy()
    
    merged = m1.merge(m2, on=['trade_day_id', 'asset_id'], how='inner')
    print(f"Merged shape: {merged.shape}")
    
    return merged


def create_meta_features(df):
    """Create meta features for ensemble"""
    df['alpha_diff'] = df['m1_alpha_z'] - df['m2_score_z']
    df['alpha_mean'] = (df['m1_alpha_z'] + df['m2_score_z']) / 2.0
    df['alpha_product'] = df['m1_alpha_z'] * df['m2_score_z']
    df['alpha_abs_gap'] = df['alpha_diff'].abs()
    
    # Cross-sectional normalization
    meta_cols = ['m1_alpha_z', 'm2_score_z', 'alpha_diff', 'alpha_mean', 'alpha_product', 'alpha_abs_gap']
    
    for col in meta_cols:
        df[f'{col}_cs'] = df.groupby('trade_day_id')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9))
    
    features = [f'{col}_cs' for col in meta_cols]
    return df, features


def train_meta_model(df, features):
    """Train Ridge meta model if target available"""
    # Try to load target from daily data
    try:
        daily = pd.read_parquet(os.path.join(
            os.path.dirname(PROJECT_ROOT), "data", "raw", "daily_data_in_sample.parquet"))
        daily['close_adj'] = daily['close'] * daily['adj_factor']
        g = daily.groupby('asset_id')
        daily['target'] = (g['close_adj'].shift(-1) - daily['close_adj']) / (daily['close_adj'] + 1e-9)
        daily['target'] = daily.groupby('trade_day_id')['target'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9))
        daily['target'] = daily['target'].fillna(0)
        
        df = df.merge(daily[['trade_day_id', 'asset_id', 'target']], 
                      on=['trade_day_id', 'asset_id'], how='left')
        has_target = True
        print("Target found in daily data")
    except:
        has_target = False
        print("Target not found, using equal-weight ensemble")
    
    if has_target:
        train_mask = df['trade_day_id'].isin(TRAIN_DAYS)
        val_mask = df['trade_day_id'].isin(VAL_DAYS)
        
        X_train = df.loc[train_mask, features].fillna(0).astype(np.float32)
        y_train = df.loc[train_mask, 'target'].fillna(0).astype(np.float32)
        X_val = df.loc[val_mask, features].fillna(0).astype(np.float32)
        y_val = df.loc[val_mask, 'target'].fillna(0).astype(np.float32)
        
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        ic, _ = spearmanr(val_pred, y_val)
        print(f"Meta model validation IC: {ic:.5f}")
        
        # Predict all
        df['meta_alpha'] = model.predict(df[features].fillna(0).astype(np.float32))
        
    else:
        # Equal weight ensemble
        df['meta_alpha'] = (df['m1_alpha_z'] * 0.5 + df['m2_score_z'] * 0.5)
    
    return df, has_target


def build_portfolio(df, top_k=10):
    """Build long/short portfolio"""
    df['meta_alpha_z'] = df.groupby('trade_day_id')['meta_alpha'].transform(
        lambda x: zscore(x, ddof=0)).fillna(0).clip(-3, 3)
    df['meta_rank'] = df.groupby('trade_day_id')['meta_alpha_z'].rank(pct=True)
    
    df['buy_percentage'] = 0.0
    df['sell_percentage'] = 0.0
    
    for day, grp in df.groupby('trade_day_id'):
        grp = grp.sort_values('meta_rank')
        
        # Long
        long_idx = grp.tail(top_k).index
        long_weight = grp.loc[long_idx, 'meta_rank']
        long_weight = long_weight / long_weight.sum()
        df.loc[long_idx, 'buy_percentage'] = long_weight.values
        
        # Short
        short_idx = grp.head(top_k).index
        short_weight = 1 - grp.loc[short_idx, 'meta_rank']
        short_weight = short_weight / short_weight.sum()
        df.loc[short_idx, 'sell_percentage'] = short_weight.values
    
    return df


def visualize(df):
    """Create visualization plots"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Alpha distribution
    axes[0].hist(df['meta_alpha_z'], bins=80, color='steelblue', alpha=0.8)
    axes[0].axvline(0, color='red', lw=1, ls='--')
    axes[0].set_title('Meta Alpha Distribution')
    axes[0].set_xlabel('alpha_z')
    axes[0].set_ylabel('Count')
    
    # Daily mean
    daily_mean = df.groupby('trade_day_id')['meta_alpha_z'].mean()
    axes[1].plot(daily_mean.values, lw=1.5)
    axes[1].axhline(0, color='grey', ls='--', lw=0.8)
    axes[1].set_title('Daily Mean Meta Alpha')
    axes[1].set_xlabel('Day Index')
    
    # Signal correlation
    sample_day = df['trade_day_id'].iloc[0]
    sample = df[df['trade_day_id'] == sample_day]
    axes[2].scatter(sample['m1_alpha_z'], sample['m2_score_z'], alpha=0.5)
    axes[2].set_xlabel('Member1 Alpha')
    axes[2].set_ylabel('Member2 Alpha')
    axes[2].set_title('Signal Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'member3_diagnostics.png'), dpi=150)
    plt.show()


def main():
    print("\n" + "="*60)
    print("MEMBER 3 - META ENSEMBLE")
    print("="*60)
    
    # Load and merge
    df = load_and_merge()
    
    # Create meta features
    df, features = create_meta_features(df)
    
    # Train meta model
    df, has_target = train_meta_model(df, features)
    
    # Build portfolio
    df = build_portfolio(df, top_k=10)
    
    # Save submission
    submission_cols = ['trade_day_id', 'asset_id', 'buy_percentage', 'sell_percentage']
    submission_path = os.path.join(OUT_DIR, 'member3_submission.csv')
    df[submission_cols].to_csv(submission_path, index=False)
    print(f"Saved submission: {submission_path}")
    
    # Save diagnostics
    diag_cols = ['trade_day_id', 'asset_id', 'm1_alpha_z', 'm2_score_z', 
                 'meta_alpha', 'meta_alpha_z', 'meta_rank']
    diag_path = os.path.join(OUT_DIR, 'member3_alpha.csv')
    df[diag_cols].to_csv(diag_path, index=False)
    print(f"Saved diagnostics: {diag_path}")
    
    # Visualize
    visualize(df)
    
    # Summary
    print("\n" + "="*60)
    print("MEMBER 3 SUMMARY")
    print("="*60)
    print(f"Fusion shape: {df.shape}")
    print(f"Submission shape: {df[submission_cols].shape}")
    print(f"Top-K holdings: 10")
    print("="*60)


if __name__ == "__main__":
    main()