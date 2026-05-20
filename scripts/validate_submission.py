"""
Validate submission file format for Feishu Quant Competition

Checks:
- Correct column names
- No missing values
- buy_percentage and sell_percentage in [0, 1]
- No duplicate (trade_day_id, asset_id)
- Minimum 10 stocks per day
- Valid asset_id format (A + 6 digits)
- Valid trade_day_id format (D + 3 digits)
- buy_percentage sum check (optional warning)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


class SubmissionValidator:
    """Validate submission file against competition rules"""
    
    REQUIRED_COLUMNS = ['trade_day_id', 'asset_id', 'buy_percentage', 'sell_percentage']
    
    def __init__(self, submission_path: str):
        self.submission_path = submission_path
        self.df = None
        self.errors = []
        self.warnings = []
    
    def load(self) -> bool:
        """Load submission file"""
        try:
            self.df = pd.read_csv(self.submission_path)
            return True
        except Exception as e:
            self.errors.append(f"Cannot load file: {e}")
            return False
    
    def validate_columns(self) -> bool:
        """Check required columns exist"""
        missing = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing:
            self.errors.append(f"Missing columns: {missing}")
            return False
        return True
    
    def validate_no_missing_values(self) -> bool:
        """Check no missing values in required columns"""
        for col in self.REQUIRED_COLUMNS:
            if self.df[col].isna().any():
                self.errors.append(f"Missing values in column: {col}")
                return False
        return True
    
    def validate_percentage_ranges(self) -> bool:
        """Check buy_percentage and sell_percentage are in [0, 1]"""
        valid = True
        
        if (self.df['buy_percentage'] < 0).any() or (self.df['buy_percentage'] > 1).any():
            self.errors.append("buy_percentage must be in [0, 1]")
            valid = False
        
        if (self.df['sell_percentage'] < 0).any() or (self.df['sell_percentage'] > 1).any():
            self.errors.append("sell_percentage must be in [0, 1]")
            valid = False
        
        return valid
    
    def validate_asset_id_format(self) -> bool:
        """Check asset_id format: A followed by 6 digits"""
        pattern = r'^A\d{6}$'
        invalid = self.df[~self.df['asset_id'].str.match(pattern, na=False)]
        
        if len(invalid) > 0:
            self.errors.append(f"Invalid asset_id format: {invalid['asset_id'].unique()[:5].tolist()}")
            return False
        return True
    
    def validate_trade_day_id_format(self) -> bool:
        """Check trade_day_id format: D followed by 3 digits"""
        pattern = r'^D\d{3}$'
        invalid = self.df[~self.df['trade_day_id'].str.match(pattern, na=False)]
        
        if len(invalid) > 0:
            self.errors.append(f"Invalid trade_day_id format: {invalid['trade_day_id'].unique()[:5].tolist()}")
            return False
        return True
    
    def validate_no_duplicates(self) -> bool:
        """Check no duplicate (trade_day_id, asset_id) pairs"""
        duplicates = self.df.duplicated(subset=['trade_day_id', 'asset_id']).sum()
        
        if duplicates > 0:
            self.errors.append(f"Found {duplicates} duplicate (trade_day_id, asset_id) pairs")
            return False
        return True
    
    def validate_daily_holdings(self, min_stocks: int = 10) -> bool:
        """Check each day has at least min_stocks unique assets"""
        daily_counts = self.df.groupby('trade_day_id')['asset_id'].nunique()
        invalid_days = daily_counts[daily_counts < min_stocks]
        
        if len(invalid_days) > 0:
            self.errors.append(f"Days with < {min_stocks} stocks: {invalid_days.index.tolist()}")
            return False
        
        self.warnings.append(f"Minimum holdings per day: {daily_counts.min()} stocks")
        return True
    
    def validate_row_ordering(self) -> bool:
        """Check rows are sorted by trade_day_id"""
        if not self.df['trade_day_id'].is_monotonic_increasing:
            self.warnings.append("Rows are not sorted by trade_day_id")
            # Auto-sort
            self.df = self.df.sort_values('trade_day_id')
            self.warnings.append("Auto-sorted by trade_day_id")
        return True
    
    def validate_decimal_precision(self, max_decimals: int = 6) -> bool:
        """Check decimal precision of percentages"""
        for col in ['buy_percentage', 'sell_percentage']:
            # Check if any value exceeds max_decimals
            decimals = self.df[col].astype(str).str.split('.').str[-1].str.len()
            if (decimals > max_decimals).any():
                self.warnings.append(f"Some {col} values have > {max_decimals} decimals (will be truncated)")
        return True
    
    def check_buy_percentage_sum(self) -> None:
        """Check daily sum of buy_percentage (warning only, no rejection)"""
        daily_sum = self.df.groupby('trade_day_id')['buy_percentage'].sum()
        if (daily_sum > 1.0).any():
            self.warnings.append(f"Daily buy_percentage sum exceeds 1.0 on some days (max={daily_sum.max():.4f})")
    
    def remove_zero_rows(self) -> None:
        """Remove rows where both buy and sell percentages are zero"""
        before = len(self.df)
        self.df = self.df[(self.df['buy_percentage'] != 0) | (self.df['sell_percentage'] != 0)]
        removed = before - len(self.df)
        if removed > 0:
            self.warnings.append(f"Removed {removed} rows with both percentages = 0")
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validations"""
        if not self.load():
            return False, self.errors, self.warnings
        
        self.remove_zero_rows()
        
        validations = [
            self.validate_columns,
            self.validate_no_missing_values,
            self.validate_percentage_ranges,
            self.validate_asset_id_format,
            self.validate_trade_day_id_format,
            self.validate_no_duplicates,
            self.validate_daily_holdings,
            self.validate_row_ordering,
            self.validate_decimal_precision,
        ]
        
        for validate in validations:
            validate()
        
        self.check_buy_percentage_sum()
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def get_validated_dataframe(self) -> pd.DataFrame:
        """Return validated and cleaned dataframe"""
        if self.df is not None:
            # Sort by trade_day_id
            self.df = self.df.sort_values('trade_day_id')
            # Remove zero rows
            self.df = self.df[(self.df['buy_percentage'] != 0) | (self.df['sell_percentage'] != 0)]
        return self.df


def print_validation_report(submission_path: str) -> None:
    """Print validation report for submission file"""
    print("\n" + "="*60)
    print("SUBMISSION VALIDATION REPORT")
    print("="*60)
    print(f"File: {submission_path}")
    
    validator = SubmissionValidator(submission_path)
    is_valid, errors, warnings = validator.validate_all()
    
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nNo errors found")
    
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if is_valid:
        print("\nSUBMISSION IS VALID")
    else:
        print("\nSUBMISSION IS INVALID")
    
    print("="*60)
    
    # Print summary
    if validator.df is not None:
        df = validator.get_validated_dataframe()
        print(f"\nFinal submission stats:")
        print(f"  - Total orders: {len(df):,}")
        print(f"  - Unique days: {df['trade_day_id'].nunique()}")
        print(f"  - Unique assets: {df['asset_id'].nunique()}")
        print("="*60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print_validation_report(sys.argv[1])
    else:
        print("Usage: python validate_submission.py <submission_file.csv>")