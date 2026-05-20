import numpy as np
import pandas as pd

DAILY_FEAT = [
    'log_ret', 'mom_3', 'mom_5', 'mom_10', 'mom_20',
    'vol_5', 'vol_10', 'vol_20', 'vol_ratio',
    'rsi_14', 'atr_ratio', 'vol_z20',
    'bb_pct_b', 'amihud', 'overnight_gap', 'hl_range', 'close_to_high',
    'macd_signal'
]

def build_daily_features(df):
    """
    Build daily features and target
    Dựa trên code gốc từ CELL 4 và PART 2 của Colab
    """
    # Sort
    df = df.sort_values(['asset_id', 'trade_day_id']).reset_index(drop=True)
    
    # Adjusted prices
    df['close_adj'] = df['close'] * df['adj_factor']
    df['high_adj'] = df['high'] * df['adj_factor']
    df['low_adj'] = df['low'] * df['adj_factor']
    df['open_adj'] = df['open'] * df['adj_factor']
    
    # Group
    gb = df.groupby('asset_id')
    prev_c = gb['close_adj'].shift(1)
    
    # Log return
    df['log_ret'] = np.log((df['close_adj'] + 1e-9) / (prev_c + 1e-9))
    
    # Momentum
    for w in [3, 5, 10, 20]:
        df[f'mom_{w}'] = gb['close_adj'].pct_change(w)
    
    # Volatility
    for w in [5, 10, 20]:
        df[f'vol_{w}'] = gb['log_ret'].transform(lambda x: x.rolling(w, min_periods=2).std())
    df['vol_ratio'] = df['vol_5'] / (df['vol_20'] + 1e-9)
    
    # RSI 14
    def _rsi(s, w=14):
        d = s.diff()
        gain = d.clip(lower=0).rolling(w, min_periods=1).mean()
        loss = (-d.clip(upper=0)).rolling(w, min_periods=1).mean()
        return 100 - 100 / (1 + gain / (loss + 1e-9))
    df['rsi_14'] = gb['close_adj'].transform(_rsi)
    
    # ATR ratio
    tr = pd.concat([
        df['high_adj'] - df['low_adj'],
        (df['high_adj'] - prev_c).abs(),
        (df['low_adj'] - prev_c).abs()
    ], axis=1).max(axis=1)
    df['atr_ratio'] = tr.groupby(df['asset_id']).transform(
        lambda x: x.rolling(14, min_periods=1).mean()) / (df['close_adj'] + 1e-9)
    
    # Volume Z-score
    df['vol_z20'] = gb['volume'].transform(
        lambda x: (x - x.rolling(20, min_periods=5).mean()) / 
                  (x.rolling(20, min_periods=5).std() + 1e-9))
    
    # Bollinger Bands %B
    sma20 = gb['close_adj'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    std20 = gb['close_adj'].transform(lambda x: x.rolling(20, min_periods=5).std()).fillna(1e-9) + 1e-9
    df['bb_pct_b'] = (df['close_adj'] - (sma20 - 2 * std20)) / (4 * std20)
    
    # Amihud illiquidity
    amt = 'amount' if 'amount' in df.columns else 'volume'
    df['amihud'] = (df['log_ret'].abs() / (df[amt] + 1e-9) * 1e6).groupby(df['asset_id']).transform(
        lambda x: x.rolling(20, min_periods=5).mean())
    
    # Overnight gap
    df['overnight_gap'] = (df['open_adj'] - prev_c) / (prev_c + 1e-9)
    
    # HL range
    df['hl_range'] = (df['high_adj'] - df['low_adj']) / (df['close_adj'] + 1e-9)
    df['close_to_high'] = (df['close_adj'] - df['low_adj']) / (
        (df['high_adj'] - df['low_adj']).replace(0, np.nan).fillna(1e-9))
    
    # MACD signal
    ema12 = gb['close_adj'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = gb['close_adj'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd_signal'] = (ema12 - ema26).groupby(df['asset_id']).transform(
        lambda x: x.ewm(span=9, adjust=False).mean())
    
    # Target: next-day return (cross-sectionally normalized)
    df['next_ret'] = gb['close_adj'].shift(-1) / (df['close_adj'] + 1e-9) - 1
    df['next_ret'] = df['next_ret'].clip(-0.1, 0.1)
    
    df['target'] = df.groupby('trade_day_id')['next_ret'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9))
    df['target'] = df['target'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Fill missing
    df.fillna(0, inplace=True)
    
    return df