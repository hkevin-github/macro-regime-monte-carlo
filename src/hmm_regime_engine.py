"""
Regime Inference Engine - Hidden Markov Model Training
Trains a 3-State Gaussian Hidden Markov Model directly on the monthly macro
observation matrix produced by the ingestion pipeline.

The upstream pipeline (data_pipeline.py) now emits data that is ALREADY monthly
(Growth, Inflation, Labor, Yield Curve, Credit Risk, Equity Returns are all
monthly-frequency and aligned on a Period[M] -> month-end timestamp index), so
there is no more daily matrix to downsample and no more monthly-to-daily label
mapping step required. The model trains directly on every row of the input file.
"""

import os
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def load_processed_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data matrix not discovered at {file_path}")
    print(f"[*] Ingesting historical master data from: {file_path}")
    return pd.read_csv(file_path, index_col='Date', parse_dates=True)


def train_3state_hmm(df: pd.DataFrame) -> tuple[GaussianHMM, pd.DataFrame]:
    """
    Trains a 3-State Gaussian HMM by initializing explicit prior structural means,
    guaranteeing a clean split into Expansion, Inflation, and Recession regimes.

    Feature set now spans macro concepts from the ingestion pipeline:
    Inflation, Labor, and Yield Curve.
    """
    feature_cols = [
        'cpi_yoy_zscore',
        'unemployment_delta_zscore',
        'yield_spread_delta_zscore',
    ]

    # 1. Data is already monthly (one row per month, month-end indexed) -- no
    #    resampling/downsampling needed anymore. Just select the feature matrix.
    monthly_df = df[feature_cols].dropna()
    X_monthly = monthly_df.values

    # 2. Instantiate with an explicit initialization configuration
    # init_params="tcip" means initialize transitions (t), covars (c), and initials (p) randomly,
    # but let us handle the means (m) manually.
    model = GaussianHMM(
        n_components=3,
        covariance_type='full',
        n_iter=1000,
        init_params="tcip",
        random_state=42
    )

    # Define explicit centers for our Z-Scores:
    # [cpi_yoy, unemployment_delta, yield_spread_delta]
    model.means_ = np.array([
        [-0.5, -0.5,  0.2],  # Regime 0: Normal / Expansion (low CPI, falling unemp, tight spreads)
        [ 1.5,  0.0, -0.2],  # Regime 1: High Inflation / Stagflation (high CPI, low spread)
        [-0.2,  2.0, -1.0],  # Regime 2: Severe Recession / Structural Shock (spiking unemployment, wide spreads)
    ])

    print("[*] Calibration Phase: Training Anchored Monthly HMM...")
    model.fit(X_monthly)

    # 3. Decode monthly paths directly -- this IS the full-resolution output now.
    monthly_states = model.predict(X_monthly)
    labeled_df = monthly_df.copy()
    labeled_df['inferred_regime'] = monthly_states

    # 4. Reattach the remaining (non-feature) columns, e.g. spy_equivalent_return and
    #    the raw Fama-French/macro columns, aligned on the same monthly index.
    other_cols = [c for c in df.columns if c not in feature_cols]
    labeled_df = labeled_df.join(df[other_cols], how='left')

    return model, labeled_df


def output_model_diagnostics(model: GaussianHMM, labeled_df: pd.DataFrame):
    print("\n" + "="*50)
    print("             CORRECTED HMM ENGINE DIAGNOSTICS             ")
    print("="*50)

    print("\n[+] Monthly Observations Allotted per Regime:")
    print(labeled_df['inferred_regime'].value_counts().sort_index())

    print("\n[+] Monthly Transition Matrix (A):")
    transition_matrix = pd.DataFrame(
        model.transmat_,
        index=[f"Regime_{i}" for i in range(3)],
        columns=[f"To_Regime_{i}" for i in range(3)]
    )
    print(transition_matrix.round(4))

    raw_feature_cols = ['cpi_yoy', 'unemployment_delta', 'yield_spread_delta']
    available_raw_cols = [c for c in raw_feature_cols if c in labeled_df.columns]
    if available_raw_cols:
        print("\n[+] Empirical Means of Raw Macro Variables per Regime Vector:")
        print(labeled_df.groupby('inferred_regime')[available_raw_cols].mean().round(4))

    print("\n[+] Market Returns Profile per Regime Vector:")
    print(labeled_df.groupby('inferred_regime')[['spy_equivalent_return']].agg(['mean', 'std', 'count']).round(5))
    print("="*50)


if __name__ == "__main__":
    DATA_INPUT = "data/processed/aligned_macro_dataset_v1.csv"
    DATA_OUTPUT = "data/processed/regime_labeled_dataset_v1.csv"

    try:
        master_df = load_processed_data(DATA_INPUT)
        hmm_model, output_df = train_3state_hmm(master_df)
        output_model_diagnostics(hmm_model, output_df)
        output_df.to_csv(DATA_OUTPUT)
        print(f"\n[+] Success: Clean regime dataset written to: {DATA_OUTPUT}")
    except Exception as e:
        print(f"[-] Training Engine Failure: {str(e)}")