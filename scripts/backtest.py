"""
Simulates portfolio execution with:
- Initial capital: 50,000,000 RMB
- Buy execution: vwap_0930_0935 (09:30-09:35 VWAP)
- Sell execution: open OR close (user selects one mode for entire OOS)
- T+1 rule: shares bought on day t cannot be sold until day t+1
- Lot size: 100 shares (A-share market convention)
- Sequential buy order execution (no normalization)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Literal

SellMode = Literal['open', 'close']


class FeishuBacktest:
    
    def __init__(
        self,
        initial_capital: float = 50_000_000,
        sell_mode: SellMode = 'open',
        lot_size: int = 100,
        transaction_cost_rate: float = 0.0003  # 3bps transaction cost (standard A-share)
    ):
        self.initial_capital = initial_capital
        self.sell_mode = sell_mode
        self.lot_size = lot_size
        self.transaction_cost_rate = transaction_cost_rate
        
        self.capital = initial_capital
        self.holdings: Dict[str, float] = {}  # {asset_id: shares_held}
        self.daily_records = []
        
    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.holdings = {}
        self.daily_records = []
    
    def _get_buy_price(self, row: pd.Series) -> float:
        """Get buy execution price (vwap_0930_0935)"""
        return row['vwap_0930_0935']
    
    def _get_sell_price(self, row: pd.Series) -> float:
        """Get sell execution price based on selected mode"""
        if self.sell_mode == 'open':
            return row['open']
        else:
            return row['close']
    
    def _round_to_lot(self, shares: float) -> int:
        """Round shares to lot size (e.g., 100 shares)"""
        return int(np.floor(shares / self.lot_size) * self.lot_size)
    
    def _apply_transaction_cost(self, amount: float) -> float:
        """Apply transaction cost (buy and sell)"""
        return amount * self.transaction_cost_rate
    
    def execute_sell_orders(self, day_orders: pd.DataFrame, daily_data: pd.DataFrame) -> None:
        """
        Execute sell orders before buy orders (T+1 rule)
        Only sell shares that were bought on previous days
        """
        for _, order in day_orders.iterrows():
            asset = order['asset_id']
            sell_pct = order['sell_percentage']
            
            if sell_pct <= 0 or asset not in self.holdings:
                continue
            
            # Find price for this asset on this day
            price_row = daily_data[daily_data['asset_id'] == asset]
            if len(price_row) == 0:
                continue
            
            sell_price = self._get_sell_price(price_row.iloc[0])
            shares_available = self.holdings[asset]
            shares_to_sell = shares_available * sell_pct
            
            # Round to lot size
            shares_to_sell = self._round_to_lot(shares_to_sell)
            
            if shares_to_sell > 0:
                proceeds = shares_to_sell * sell_price
                cost = self._apply_transaction_cost(proceeds)
                self.capital += (proceeds - cost)
                self.holdings[asset] -= shares_to_sell
                
                # Remove if no shares left
                if self.holdings[asset] < 1e-6:
                    del self.holdings[asset]
    
    def execute_buy_orders(self, day_orders: pd.DataFrame, daily_data: pd.DataFrame) -> None:
        """
        Execute buy orders sequentially in CSV order
        Each order allocates a percentage of REMAINING cash
        """
        for _, order in day_orders.iterrows():
            asset = order['asset_id']
            buy_pct = order['buy_percentage']
            
            if buy_pct <= 0:
                continue
            
            # Find price for this asset on this day
            price_row = daily_data[daily_data['asset_id'] == asset]
            if len(price_row) == 0:
                continue
            
            buy_price = self._get_buy_price(price_row.iloc[0])
            allocated_cash = self.capital * buy_pct
            
            # Calculate shares (round to lot size)
            shares_to_buy = self._round_to_lot(allocated_cash / buy_price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * buy_price
                total_cost = cost + self._apply_transaction_cost(cost)
                
                if total_cost <= self.capital:
                    self.capital -= total_cost
                    self.holdings[asset] = self.holdings.get(asset, 0) + shares_to_buy
    
    def calculate_portfolio_value(self, day: str, daily_data: pd.DataFrame) -> float:
        """Calculate end-of-day portfolio value"""
        portfolio_value = self.capital
        
        for asset, shares in self.holdings.items():
            price_row = daily_data[daily_data['asset_id'] == asset]
            if len(price_row) > 0:
                # Use closing price for end-of-day valuation
                portfolio_value += shares * price_row.iloc[0]['close']
        
        return portfolio_value
    
    def run(self, submission_df: pd.DataFrame, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        Run full backtest simulation
        
        Args:
            submission_df: DataFrame with columns [trade_day_id, asset_id, buy_percentage, sell_percentage]
            daily_data: Daily OHLCV data
        
        Returns:
            DataFrame with daily portfolio values and metrics
        """
        self.reset()
        
        # Sort by day
        days = sorted(submission_df['trade_day_id'].unique())
        
        for day in days:
            day_orders = submission_df[submission_df['trade_day_id'] == day]
            day_daily = daily_data[daily_data['trade_day_id'] == day]
            
            # Process sell orders first (T+1 rule)
            self.execute_sell_orders(day_orders, day_daily)
            
            # Process buy orders
            self.execute_buy_orders(day_orders, day_daily)
            
            # Calculate portfolio value at end of day
            portfolio_value = self.calculate_portfolio_value(day, day_daily)
            holdings_count = len(self.holdings)
            
            self.daily_records.append({
                'trade_day_id': day,
                'portfolio_value': portfolio_value,
                'cash': self.capital,
                'holdings_count': holdings_count,
                'holdings': str(list(self.holdings.keys()))[:100]  # Truncated for display
            })
            
            # Validate minimum holdings (at least 10 stocks)
            if holdings_count < 10:
                print(f"Warning: Day {day} has only {holdings_count} stocks (minimum required: 10)")
        
        return pd.DataFrame(self.daily_records)