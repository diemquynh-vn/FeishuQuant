def add_market_features(df):
    df['market_obi'] = df.groupby('time')['obi_total'].transform('mean')
    df['market_spread'] = df.groupby('time')['spread_rel'].transform('mean')
    df['obi_vs_market'] = df['obi_total'] - df['market_obi']
    df['spread_vs_market'] = df['spread_rel'] - df['market_spread']
    return df

MARKET_FEATS = ['market_obi', 'market_spread', 'obi_vs_market', 'spread_vs_market']