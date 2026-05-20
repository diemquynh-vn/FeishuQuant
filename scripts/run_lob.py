"""
PART 1: Full LOB Pipeline
Run training -> generate alpha -> portfolio
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import main as train_main
from scripts.generate_alpha import main as generate_main
from scripts.portfolio import main as portfolio_main

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PART 1: FULL LOB PIPELINE")
    print("="*70)
    
    print("\n[1/3] Training model...")
    train_main()
    
    print("\n[2/3] Generating alpha...")
    generate_main()
    
    print("\n[3/3] Building portfolio...")
    portfolio_main()
    
    print("\n" + "="*70)
    print("PART 1 COMPLETE!")
    print("="*70)