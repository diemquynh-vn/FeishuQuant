"""
Run full backtest on validation period to evaluate strategy performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from config import ALPHA_DIR, DATA_RAW, VAL_DAYS
from config_oos import BACKTEST_PARAMS, INITIAL_CAPITAL, SELL_MODE, TRANSACTION_COST_RATE
from scripts.backtest import FeishuBacktest
from scripts.evaluate import evaluate_portfolio, generate_submission_summary
from scripts.validate_submission import SubmissionValidator


def load_validation_submission():
    """Load validation submission (from member1 or member3)"""
    
    # Try member3 submission first
    member3_path = os.path.join(ALPHA_DIR, '../member3_result/member3_submission.csv')
    if os.path.exists(member3_path):
        submission = pd.read_csv(member3_path)
        print(f"Loaded member3 submission: {len(submission):,} rows")
        return submission
    
    # Try member1 submission
    member1_path = os.path.join(ALPHA_DIR, 'member1_submission.csv')
    if os.path.exists(member1_path):
        submission = pd.read_csv(member1_path)
        print(f"Loaded member1 submission: {len(submission):,} rows")
        return submission
    
    raise FileNotFoundError("No submission file found")


def load_daily_data_for_days(days):
    """Load daily data for specific days"""
    daily = pd.read_parquet(os.path.join(DATA_RAW, "daily_data_in_sample.parquet"))
    daily = daily[daily['trade_day_id'].isin(days)]
    return daily


def main():
    print("\n" + "="*70)
    print("BACKTEST SIMULATION FOR FEISHU COMPETITION")
    print("="*70)
    print(f"Initial Capital: {INITIAL_CAPITAL:,.0f} RMB")
    print(f"Sell Mode: {SELL_MODE}")
    print(f"Transaction Cost: {TRANSACTION_COST_RATE*100:.3f}%")
    print("="*70)
    
    # Load submission
    submission_df = load_validation_submission()
    
    # Validate submission
    validator = SubmissionValidator(submission_df)
    # Note: For validation, we need to pass dataframe directly
    validator.df = submission_df
    is_valid, errors, warnings = validator.validate_all()
    
    if not is_valid:
        print("\nSubmission has errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    print("\nSubmission validation passed")
    
    # Load daily data for validation days
    days_in_submission = submission_df['trade_day_id'].unique()
    daily_data = load_daily_data_for_days(days_in_submission)
    
    # Run backtest
    backtest = FeishuBacktest(
        initial_capital=INITIAL_CAPITAL,
        sell_mode=SELL_MODE,
        lot_size=100,
        transaction_cost_rate=TRANSACTION_COST_RATE
    )
    
    results = backtest.run(submission_df, daily_data)
    
    # Generate submission summary
    generate_submission_summary(submission_df)
    
    # Evaluate performance
    # For validation, we don't have all submissions for percentile
    metrics = evaluate_portfolio(results)
    
    # Print daily results summary
    print("\n" + "="*60)
    print("DAILY RESULTS SAMPLE")
    print("="*60)
    print(results[['trade_day_id', 'portfolio_value', 'cash', 'holdings_count']].head(10).to_string(index=False))
    
    # Check minimum holdings requirement
    min_holdings = results['holdings_count'].min()
    if min_holdings < 10:
        print(f"\nWARNING: Minimum holdings = {min_holdings} (< 10 required)")
    else:
        print(f"\nMinimum holdings = {min_holdings} (>= 10 required)")
    
    # Save backtest results
    output_path = os.path.join(os.path.dirname(ALPHA_DIR), 'outputs', 'backtest_results.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nBacktest results saved to: {output_path}")
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()