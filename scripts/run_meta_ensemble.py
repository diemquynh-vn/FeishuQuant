"""
PART 3: Meta Ensemble (FULL VERSION từ Colab)
Combine member1 (LOB alpha) + member2 (daily alpha)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, zscore
from sklearn.linear_model import RidgeCV

from config import ALPHA_DIR, ALPHA_RESULT_DIR, MEMBER3_DIR, PLOT_DIR, TRAIN_DAYS, VAL_DAYS
from src.utils.seed import set_seed

set_seed(42)


def main():
    print("\n" + "="*60)
    print("PART 3: META ENSEMBLE")
    print("="*60)
    
    # ======================================================
    # Load alpha files
    # ======================================================
    m1 = pd.read_csv(os.path.join(ALPHA_DIR, 'member1_alpha.csv'))
    m2 = pd.read_csv(os.path.join(ALPHA_RESULT_DIR, 'member2_alpha.csv'))
    
    print(f'Member1 shape: {m1.shape}')
    print(f'Member2 shape: {m2.shape}')
    
    # Select columns
    m1 = m1[['trade_day_id', 'asset_id', 'alpha_score', 'alpha_z']].copy()
    m1.columns = ['trade_day_id', 'asset_id', 'm1_alpha', 'm1_alpha_z']
    
    m2 = m2[['trade_day_id', 'asset_id', 'm2_score', 'm2_score_z']].copy()
    
    # Merge
    fusion_df = m1.merge(m2, on=['trade_day_id', 'asset_id'], how='inner')
    print(f'Fusion shape: {fusion_df.shape}')
    
    # ======================================================
    # Load target if available
    # ======================================================
    try:
        daily = pd.read_parquet(os.path.join(
            os.path.dirname(ALPHA_DIR), 'raw', 'daily_data_in_sample.parquet'))
        daily['close_adj'] = daily['close'] * daily['adj_factor']
        g = daily.groupby('asset_id')
        daily['target'] = (g['close_adj'].shift(-1) - daily['close_adj']) / (daily['close_adj'] + 1e-9)
        daily['target'] = daily.groupby('trade_day_id')['target'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0)
        
        fusion_df = fusion_df.merge(
            daily[['trade_day_id', 'asset_id', 'target']],
            on=['trade_day_id', 'asset_id'],
            how='left'
        )
        has_target = True
        print("Target loaded")
    except Exception as e:
        has_target = False
        print(f"Target not found: {e}")
    
    # ======================================================
    # Feature engineering (đầy đủ như Colab)
    # ======================================================
    fusion_df['alpha_diff'] = fusion_df['m1_alpha_z'] - fusion_df['m2_score_z']
    fusion_df['alpha_mean'] = (fusion_df['m1_alpha_z'] + fusion_df['m2_score_z']) / 2.0
    fusion_df['alpha_product'] = fusion_df['m1_alpha_z'] * fusion_df['m2_score_z']
    fusion_df['alpha_abs_gap'] = fusion_df['alpha_diff'].abs()
    
    # Cross-sectional normalization
    meta_cols = ['m1_alpha_z', 'm2_score_z', 'alpha_diff', 'alpha_mean', 'alpha_product', 'alpha_abs_gap']
    
    for col in meta_cols:
        fusion_df[f'{col}_cs'] = fusion_df.groupby('trade_day_id')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9))
    
    FEATURES = [f'{col}_cs' for col in meta_cols]
    print(f'Total features: {len(FEATURES)}')
    
    # ======================================================
    # Train / validation split
    # ======================================================
    train_mask = fusion_df['trade_day_id'].isin(TRAIN_DAYS)
    val_mask = fusion_df['trade_day_id'].isin(VAL_DAYS)
    
    train_df = fusion_df.loc[train_mask].copy()
    val_df = fusion_df.loc[val_mask].copy()
    
    X_train = train_df[FEATURES].fillna(0).astype(np.float32)
    X_val = val_df[FEATURES].fillna(0).astype(np.float32)
    
    # ======================================================
    # Meta model
    # ======================================================
    if has_target:
        y_train = train_df['target'].fillna(0).astype(np.float32)
        y_val = val_df['target'].fillna(0).astype(np.float32)
        
        print("\nTraining meta ensemble model...")
        meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        meta_model.fit(X_train, y_train)
        
        val_pred = meta_model.predict(X_val)
        ic, _ = spearmanr(val_pred, y_val)
        print(f'Validation IC: {ic:+.5f}')
        
        # Predict all
        fusion_df['meta_alpha'] = meta_model.predict(
            fusion_df[FEATURES].fillna(0).astype(np.float32))
    else:
        print("Target not found → using equal-weight ensemble")
        fusion_df['meta_alpha'] = (fusion_df['m1_alpha_z'] * 0.5 + fusion_df['m2_score_z'] * 0.5)
    
    # ======================================================
    # Normalize final alpha
    # ======================================================
    fusion_df['meta_alpha_z'] = fusion_df.groupby('trade_day_id')['meta_alpha'].transform(
        lambda x: zscore(x, ddof=0)).fillna(0).clip(-3, 3)
    fusion_df['meta_rank'] = fusion_df.groupby('trade_day_id')['meta_alpha_z'].rank(pct=True)
    
    # ======================================================
    # Portfolio construction
    # ======================================================
    TOP_K = 10
    fusion_df['buy_percentage'] = 0.0
    fusion_df['sell_percentage'] = 0.0
    
    for day, grp in fusion_df.groupby('trade_day_id'):
        grp = grp.sort_values('meta_rank')
        
        # Long
        long_idx = grp.tail(TOP_K).index
        long_weight = grp.loc[long_idx, 'meta_rank']
        long_weight = long_weight / long_weight.sum()
        fusion_df.loc[long_idx, 'buy_percentage'] = long_weight.values
        
        # Short
        short_idx = grp.head(TOP_K).index
        short_weight = 1 - grp.loc[short_idx, 'meta_rank']
        short_weight = short_weight / short_weight.sum()
        fusion_df.loc[short_idx, 'sell_percentage'] = short_weight.values
    
    # ======================================================
    # Save submission
    # ======================================================
    submission_cols = ['trade_day_id', 'asset_id', 'buy_percentage', 'sell_percentage']
    submission_path = os.path.join(MEMBER3_DIR, 'member3_submission.csv')
    fusion_df[submission_cols].to_csv(submission_path, index=False)
    print(f'\nSaved submission: {submission_path}')
    
    # ======================================================
    # Save diagnostics
    # ======================================================
    diag_cols = [
        'trade_day_id', 'asset_id',
        'm1_alpha_z', 'm2_score_z',
        'meta_alpha', 'meta_alpha_z', 'meta_rank'
    ]
    diag_path = os.path.join(MEMBER3_DIR, 'member3_alpha.csv')
    fusion_df[diag_cols].to_csv(diag_path, index=False)
    print(f'Saved diagnostics: {diag_path}')
    
    # ======================================================
    # Visualization
    # ======================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Alpha distribution
    axes[0].hist(fusion_df['meta_alpha_z'], bins=80, color='steelblue', alpha=0.8)
    axes[0].axvline(0, color='red', lw=1, ls='--')
    axes[0].set_title('Meta Alpha Distribution')
    axes[0].set_xlabel('alpha_z')
    axes[0].set_ylabel('Count')
    
    # Daily mean alpha
    daily_mean = fusion_df.groupby('trade_day_id')['meta_alpha_z'].mean()
    axes[1].plot(daily_mean.values, lw=1.5, color='tomato')
    axes[1].axhline(0, color='grey', ls='--', lw=0.8)
    axes[1].set_title('Daily Mean Meta Alpha')
    axes[1].set_xlabel('Day Index')
    
    # Signal correlation
    sample_day = fusion_df['trade_day_id'].iloc[0]
    sample = fusion_df[fusion_df['trade_day_id'] == sample_day]
    axes[2].scatter(sample['m1_alpha_z'], sample['m2_score_z'], alpha=0.5)
    axes[2].set_xlabel('Member1 Alpha')
    axes[2].set_ylabel('Member2 Alpha')
    axes[2].set_title('Signal Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'member3_diagnostics.png'), dpi=150)
    plt.show()
    
    # ======================================================
    # Summary
    # ======================================================
    print('\n' + '='*60)
    print('MEMBER 3 — META ENSEMBLE SUMMARY')
    print('='*60)
    print(f'Fusion shape      : {fusion_df.shape}')
    print(f'Submission shape  : {fusion_df[submission_cols].shape}')
    print(f'Top-K holdings    : {TOP_K}')
    if has_target:
        print(f'Validation IC     : {ic:+.5f}')
    print('='*60)


if __name__ == "__main__":
    main()