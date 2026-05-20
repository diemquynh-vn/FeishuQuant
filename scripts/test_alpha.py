"""Test alpha quality with IC and ICIR"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from config import ALPHA_DIR, ALPHA_RESULT_DIR, MEMBER3_DIR, VAL_DAYS

def test_alpha(alpha_df, name):
    val_df = alpha_df[alpha_df['trade_day_id'].isin(VAL_DAYS)].dropna(subset=['actual_return'])
    
    if 'actual_return' not in val_df.columns:
        print(f"{name}: No actual_return column")
        return None
    
    ics = []
    for day, grp in val_df.groupby('trade_day_id'):
        if len(grp) >= 5:
            ic = spearmanr(grp['alpha_score'], grp['actual_return'])[0]
            if not np.isnan(ic):
                ics.append(ic)
    
    if len(ics) == 0:
        print(f"{name}: No valid ICs")
        return None
    
    mean_ic = np.mean(ics)
    std_ic = np.std(ics)
    icir = mean_ic / (std_ic + 1e-9)
    
    print(f"\n{name}:")
    print(f"  Mean IC: {mean_ic:+.4f}")
    print(f"  Std IC:  {std_ic:.4f}")
    print(f"  ICIR:    {icir:+.4f}")
    
    return mean_ic, std_ic, icir

def main():
    print("\n" + "="*60)
    print("ALPHA QUALITY TEST")
    print("="*60)
    
    
    m1_path = os.path.join(ALPHA_DIR, 'member1_alpha.csv')
    if os.path.exists(m1_path):
        m1 = pd.read_csv(m1_path)
        test_alpha(m1, "Member 1 (CNN-LSTM)")
    
    
    m2_path = os.path.join(ALPHA_RESULT_DIR, 'member2_alpha.csv')
    if os.path.exists(m2_path):
        m2 = pd.read_csv(m2_path)
        m2 = m2.rename(columns={'m2_score': 'alpha_score', 'm2_score_z': 'alpha_z'})
        test_alpha(m2, "Member 2 (LGB + Ridge)")
    
    
    m3_path = os.path.join(MEMBER3_DIR, 'member3_alpha.csv')
    if os.path.exists(m3_path):
        m3 = pd.read_csv(m3_path)
        m3['alpha_score'] = m3['meta_alpha']
        m3['actual_return'] = None  # No actual return
        print("\nMember 3 (Meta Ensemble): No actual_return in file")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
    