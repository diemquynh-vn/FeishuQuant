import numpy as np
import pandas as pd

PRICE_COLS = [f'bid_price_{i}' for i in range(1, 11)] + [f'ask_price_{i}' for i in range(1, 11)]
VOLUME_COLS = [f'bid_volume_{i}' for i in range(1, 11)] + [f'ask_volume_{i}' for i in range(1, 11)]

LOB_FEAT = [
    'obi_total', 'obi_top3', 'obi_deep3', 'obi_divergence',
    *[f'obi_L{i}' for i in range(1, 11)],
    'slope_bid', 'slope_ask', 'spread_rel', 'vwap_bias',
    'qp_bid', 'qp_ask', 'mid_ret', 'obi_delta',
    'obi_ma3', 'obi_ma5', 'obi_std5', 'price_accel',
]

MARKET_FEATS = ['market_obi', 'market_spread', 'obi_vs_market', 'spread_vs_market']

def lob_features(df):
    """Calculate order book imbalance features"""
    bv = {i: df[f'bid_volume_{i}'] for i in range(1, 11)}
    av = {i: df[f'ask_volume_{i}'] for i in range(1, 11)}
    bp = {i: df[f'bid_price_{i}'] for i in range(1, 11)}
    ap = {i: df[f'ask_price_{i}'] for i in range(1, 11)}
    
    tot_bv = sum(bv[i] for i in range(1, 11))
    tot_av = sum(av[i] for i in range(1, 11))
    
    # Overall OBI
    df['obi_total'] = (tot_bv - tot_av) / (tot_bv + tot_av + 1e-9)
    
    # Per-level OBI
    for i in range(1, 11):
        df[f'obi_L{i}'] = (bv[i] - av[i]) / (bv[i] + av[i] + 1e-9)
    
    # Top vs deep
    t3bv = sum(bv[i] for i in range(1, 4))
    t3av = sum(av[i] for i in range(1, 4))
    d3bv = sum(bv[i] for i in range(8, 11))
    d3av = sum(av[i] for i in range(8, 11))
    
    df['obi_top3'] = (t3bv - t3av) / (t3bv + t3av + 1e-9)
    df['obi_deep3'] = (d3bv - d3av) / (d3bv + d3av + 1e-9)
    df['obi_divergence'] = df['obi_top3'] - df['obi_deep3']
    
    # Mid price and spread
    df['mid_price'] = (bp[1] + ap[1]) / 2.0
    df['spread_rel'] = (ap[1] - bp[1]) / (df['mid_price'] + 1e-9)
    
    # Slope
    df['slope_bid'] = (bp[1] - bp[10]) / (tot_bv + 1e-9)
    df['slope_ask'] = (ap[10] - ap[1]) / (tot_av + 1e-9)
    
    # VWAP bias
    vwap_b = sum(bp[i] * bv[i] for i in range(1, 11)) / (tot_bv + 1e-9)
    vwap_a = sum(ap[i] * av[i] for i in range(1, 11)) / (tot_av + 1e-9)
    df['vwap_bias'] = (vwap_a - vwap_b) / (df['mid_price'] + 1e-9)
    
    # Queue position
    df['qp_bid'] = bv[1] / (tot_bv + 1e-9)
    df['qp_ask'] = av[1] / (tot_av + 1e-9)
    
    return df


def order_flow_features(df):
    """Calculate order flow dynamics - FIXED: preserves asset_id"""
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure time column is sorted
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)
    
    # Calculate features
    df['mid_ret'] = df['mid_price'].pct_change().fillna(0)
    df['obi_delta'] = df['obi_total'].diff().fillna(0)
    df['obi_ma3'] = df['obi_total'].rolling(3, min_periods=1).mean()
    df['obi_ma5'] = df['obi_total'].rolling(5, min_periods=1).mean()
    df['obi_std5'] = df['obi_total'].rolling(5, min_periods=1).std().fillna(0)
    df['price_accel'] = df['mid_ret'].diff().fillna(0)
    
    return df


def add_market_features(df):
    """Add market-level features"""
    df['market_obi'] = df.groupby('time')['obi_total'].transform('mean')
    df['market_spread'] = df.groupby('time')['spread_rel'].transform('mean')
    df['obi_vs_market'] = df['obi_total'] - df['market_obi']
    df['spread_vs_market'] = df['spread_rel'] - df['market_spread']
    return df