"""
RUN FULL PIPELINE - Part 1 + Part 2 + Part 3
Chạy toàn bộ quy trình từ A đến Z
"""

import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def run_script(script_name, description):
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70)
    
    script_path = os.path.join(PROJECT_ROOT, "scripts", script_name)
    result = subprocess.run([sys.executable, script_path], cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"Failed: {description}")
        return False
    print(f"Completed: {description}")
    return True

def main():
    print("\n" + "="*70)
    print("FEISHU - FULL ALPHA GENERATION PIPELINE")
    print("="*70)
    
    steps = [
        ("run_lob.py", "Part 1: CNN-LSTM + LOB Alpha"),
        ("run_daily.py", "Part 2: LightGBM + Ridge + Risk Parity"),
        ("run_meta_ensemble.py", "Part 3: Meta Ensemble"),
    ]
    
    for script, desc in steps:
        if not run_script(script, desc):
            print(f"\nPipeline stopped at: {desc}")
            return 1
    
    print("\n" + "="*70)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nOutput files:")
    print("  - data/outputs/alpha_signals/member1_alpha.csv")
    print("  - data/outputs/alpha_signals_result/member2_alpha.csv")
    print("  - data/outputs/member3_result/member3_submission.csv")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    exit(main())