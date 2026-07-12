"""
Live Empirical HMM Validation Suite
Evaluates the persistence and volatility separation directly from the generated CSV dataset.

NOTE: The upstream pipeline now produces MONTHLY data end-to-end (one row per
month), not daily data with regimes mapped back onto a daily calendar. Every
calculation below has been adjusted for monthly-frequency observations.
"""

import os
import pandas as pd
import numpy as np

def run_validation_suite(data_path="data/processed/regime_labeled_dataset_v1.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing labeled dataset. Run hmm_regime_engine.py first.")

    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)

    print("="*60)
    print("           EMPIRICAL QUANTITATIVE VALIDATION REPORT          ")
    print("="*60)

    # --- TEST 1: LIVE TRANSITIONS ---
    print("\n[TEST 1] Checking Monthly Regime Persistence...")
    # Calculate transitions by checking shifts month-over-month
    df['next_regime'] = df['inferred_regime'].shift(-1)

    # Monthly regimes are expected to flip more often than daily-mapped ones
    # (a regime lasting ~10 months implies ~0.90 self-persistence, not ~0.99),
    # so the pass bar is lower than the old daily threshold.
    PERSISTENCE_THRESHOLD = 0.80

    for r in sorted(df['inferred_regime'].unique()):
        state_data = df[df['inferred_regime'] == r]
        same_state = state_data[state_data['next_regime'] == r]
        persistence = len(same_state) / len(state_data) if len(state_data) > 0 else 0
        status = "PASSED" if persistence >= PERSISTENCE_THRESHOLD else "WARNING"
        print(f"  * Regime {r} Monthly Self-Persistence: {persistence:.4f} -> {status}")

    # --- TEST 2: VOLATILITY SEPARATION ---
    print("\n[TEST 2] Checking Volatility Separation Across Regimes...")
    # Monthly returns annualize with sqrt(12), not sqrt(252) (that factor is for daily returns)
    vol_profiles = df.groupby('inferred_regime')['spy_equivalent_return'].std() * np.sqrt(12)
    print("  * Annualized Equity Volatility Profile:")
    for regime, vol in vol_profiles.items():
        print(f"    - Regime {regime}: {vol*100:.2f}%")

    max_vol_ratio = vol_profiles.max() / vol_profiles.min()
    if max_vol_ratio > 1.3:
        print(f"  * Status: PASSED (Strong risk variance division discovered. Ratio: {max_vol_ratio:.2f}x)")
    else:
        print("  * Status: WARNING (Overlapping volatility footprint.)")
    print("="*60)

if __name__ == "__main__":
    run_validation_suite()