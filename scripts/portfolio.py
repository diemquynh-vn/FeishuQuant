import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from config import ALPHA_DIR, PLOT_DIR, VAL_DAYS

def build_portfolio(alpha_df, top_k=10):
    alpha_df = alpha_df.copy()
    alpha_df['buy_percentage'] = 0.0
    alpha_df['sell_percentage'] = 0.0
    
    for day, grp in alpha_df.groupby('trade_day_id'):
        grp = grp.sort_values('alpha_rank')
        
        long_idx = grp.tail(top_k).index
        long_weight = grp.loc[long_idx, 'alpha_rank']
        long_weight = long_weight / long_weight.sum()
        alpha_df.loc[long_idx, 'buy_percentage'] = long_weight.values
        
        short_idx = grp.head(top_k).index
        short_weight = 1 - grp.loc[short_idx, 'alpha_rank']
        short_weight = short_weight / short_weight.sum()
        alpha_df.loc[short_idx, 'sell_percentage'] = short_weight.values
    
    return alpha_df


def visualize(alpha_df, daily_ics):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Alpha distribution
    axes[0].hist(alpha_df['alpha_z'], bins=80, color='steelblue', alpha=0.8)
    axes[0].axvline(0, color='red', lw=1, ls='--')
    axes[0].set_title('Alpha Z Distribution')
    axes[0].set_xlabel('alpha_z')
    
    # Daily mean alpha
    daily_mean = alpha_df.groupby('trade_day_id')['alpha_z'].mean()
    axes[1].plot(daily_mean.values, color='tomato', lw=1)
    axes[1].axhline(0, ls='--', color='grey', lw=0.8)
    axes[1].set_title('Daily Mean Alpha')
    axes[1].set_xlabel('Day Index')
    
    # Rolling IC
    if len(daily_ics) > 0:
        rolling_ic = pd.Series(daily_ics).rolling(20, min_periods=1).mean()
        axes[2].plot(rolling_ic.values, color='green', lw=1.5, label='20D Rolling IC')
        axes[2].axhline(0, ls='--', color='grey', lw=0.8)
        axes[2].axhline(0.03, ls='--', color='orange', lw=0.8, label='IC=0.03')
        axes[2].set_title('Rolling IC')
        axes[2].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'part1_diagnostics.png'), dpi=150)
    plt.show()


def main():
    print("\n" + "="*60)
    print("PART 1: PORTFOLIO & VISUALIZATION")
    print("="*60)
    
    alpha_path = os.path.join(ALPHA_DIR, 'member1_alpha.csv')
    if not os.path.exists(alpha_path):
        print(f"Error: {alpha_path} not found. Run generate_alpha.py first.")
        return
    
    alpha_df = pd.read_csv(alpha_path)
    print(f"Loaded {len(alpha_df):,} signals")
    
    # Build portfolio
    alpha_df = build_portfolio(alpha_df, top_k=10)
    
    # Save submission
    submission_cols = ['trade_day_id', 'asset_id', 'buy_percentage', 'sell_percentage']
    alpha_df[submission_cols].to_csv(os.path.join(ALPHA_DIR, 'member1_submission.csv'), index=False)
    print(f"Saved submission to {ALPHA_DIR}")
    
    # Evaluate
    val_alpha = alpha_df[alpha_df['trade_day_id'].isin(VAL_DAYS)].dropna(subset=['actual_return'])
    daily_ics = []
    for day, grp in val_alpha.groupby('trade_day_id'):
        if len(grp) >= 5:
            ic = spearmanr(grp['alpha_score'], grp['actual_return'])[0]
            if not np.isnan(ic):
                daily_ics.append(ic)
    
    if daily_ics:
        print(f"\nValidation IC: {np.mean(daily_ics):+.4f} (std={np.std(daily_ics):.4f})")
        visualize(alpha_df, daily_ics)
    
    print("="*60)


if __name__ == "__main__":
    main()