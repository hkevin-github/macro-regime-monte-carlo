"""
Regime Inference Engine - Hidden Markov Model Training
Trains a Gaussian Hidden Markov Model directly on the monthly macro
observation matrix produced by the ingestion pipeline.

State count: n_components=4. BIC model selection empirically shows 4 states
fit meaningfully better than 2, 3, or 5+ states. Three states produced two
near-duplicate regimes with ~0% persistence that swapped labels every month.
Regimes are initialized unsupervised (k-means-based) and labeled after fitting
from their empirical feature means, not assumed in advance.

The upstream pipeline (data_pipeline.py) emits data that is already monthly
(Growth, Inflation, Labor, Yield Curve, Credit Risk, Equity Returns are all
monthly-frequency and aligned on a Period[M] -> month-end timestamp index), so
there is no daily matrix to downsample. The model trains directly on every row.
"""

import os
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def load_processed_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data matrix not discovered at {file_path}")
    print(f"[*] Ingesting historical master data from: {file_path}")
    return pd.read_csv(file_path, index_col='date', parse_dates=True)


def train_regime_hmm(df: pd.DataFrame, n_components: int = 4) -> tuple[GaussianHMM, pd.DataFrame]:
    """
    Trains a Gaussian HMM on the monthly macro observation matrix.
    
    Feature set spans all five macro concepts from the ingestion pipeline:
    Growth, Inflation, Labor, Yield Curve, and Credit Risk.
    """
    feature_cols = [
        'growth_yoy_zscore',
        'cpi_yoy_zscore',
        'unemployment_delta_zscore',
        'yield_spread_delta_zscore',
        'credit_spread_delta_zscore',
    ]

    # Data is already monthly (one row per month, month-end indexed)
    monthly_df = df[feature_cols].dropna()
    X_monthly = monthly_df.values

    # Instantiate with fully unsupervised initialization
    # init_params="stmc" initializes start probs (s), transitions (t), 
    # means (m), and covars (c) from the data (k-means for means),
    # letting natural cluster structure emerge
    model = GaussianHMM(
        n_components=n_components,
        covariance_type='diag',
        n_iter=1000,
        init_params="stmc",
        random_state=42
    )

    print(f"[*] Training {n_components}-state HMM (unsupervised initialization)...")
    model.fit(X_monthly)

    # Decode regime sequence
    monthly_states = model.predict(X_monthly)
    labeled_df = monthly_df.copy()
    labeled_df['inferred_regime'] = monthly_states

    # Reattach non-feature columns (equity returns, Fama-French factors, raw macro)
    other_cols = [c for c in df.columns if c not in feature_cols]
    labeled_df = labeled_df.join(df[other_cols], how='left')

    return model, labeled_df


def check_regime_quality(labeled_df: pd.DataFrame, model: GaussianHMM, feature_cols: list):
    """
    Diagnose whether regimes are well-separated and stable.
    Flags potential issues: small regimes, low persistence, near-duplicate states.
    """
    print("\n" + "="*60)
    print("REGIME QUALITY DIAGNOSTICS")
    print("="*60)
    
    # 1. Regime sizes
    regime_counts = labeled_df['inferred_regime'].value_counts().sort_index()
    print("\n[*] Regime Frequency:")
    for regime, count in regime_counts.items():
        pct = 100 * count / len(labeled_df)
        print(f"Regime {regime}: {count:4d} months ({pct:5.1f}%)")
    
    smallest = regime_counts.min()
    print(f"\nSmallest regime: {smallest} months ({100*smallest/len(labeled_df):.1f}%)")
    if smallest < 50:
        print("⚠️  WARNING: Regime with < 50 months may be unstable")
    
    # 2. Regime persistence (diagonal of transition matrix)
    persistence = np.diag(model.transmat_)
    print("\n[*] Regime Persistence (P(stay in same regime next month)):")
    for i, p in enumerate(persistence):
        status = "✓" if p >= 0.75 else "⚠️" if p >= 0.65 else "❌"
        print(f"Regime {i}: {p:.3f} {status}")
    
    if any(persistence < 0.70):
        print("\n⚠️  WARNING: Some regimes have persistence < 0.70 (frequent switching)")
    
    # 3. Regime separation (Euclidean distance between regime means in z-score space)
    regime_means = labeled_df.groupby('inferred_regime')[feature_cols].mean().values
    n_regimes = len(regime_means)
    
    print("\n[*] Pairwise Distance Between Regime Means (z-score space):")
    min_dist = float('inf')
    close_pairs = []
    
    for i in range(n_regimes):
        for j in range(i+1, n_regimes):
            dist = np.linalg.norm(regime_means[i] - regime_means[j])
            status = "✓" if dist >= 1.5 else "⚠️" if dist >= 1.0 else "❌"
            print(f"Regime {i} ↔ Regime {j}: {dist:.3f} {status}")
            
            if dist < min_dist:
                min_dist = dist
            if dist < 1.0:
                close_pairs.append((i, j, dist))
    
    if close_pairs:
        print(f"\n⚠️  WARNING: {len(close_pairs)} regime pair(s) very close (distance < 1.0):")
        for i, j, d in close_pairs:
            print(f"   Regimes {i} and {j} (distance: {d:.3f}) may be near-duplicates")
    
    print("="*60)


def output_model_diagnostics(model: GaussianHMM, labeled_df: pd.DataFrame, feature_cols: list):
    n_components = model.n_components
    print("\n" + "="*60)
    print("             HMM REGIME MODEL DIAGNOSTICS             ")
    print("="*60)

    print(f"\n[+] Observations per Regime (n_components={n_components}):")
    regime_counts = labeled_df['inferred_regime'].value_counts().sort_index()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(labeled_df)
        print(f"Regime {regime}: {count:4d} months ({pct:5.1f}%)")

    print("\n[+] Transition Matrix (probability of moving from row regime to column regime):")
    transition_matrix = pd.DataFrame(
        model.transmat_,
        index=[f"Regime_{i}" for i in range(n_components)],
        columns=[f"To_{i}" for i in range(n_components)]
    )
    print(transition_matrix.round(3))

    print("\n[+] Z-Scored Feature Means per Regime (use this to interpret regimes):")
    regime_feature_means = labeled_df.groupby('inferred_regime')[feature_cols].mean()
    print(regime_feature_means.round(3))

    raw_feature_cols = ['growth_yoy', 'cpi_yoy', 'unemployment_delta', 'yield_spread_delta', 'credit_spread_delta']
    available_raw_cols = [c for c in raw_feature_cols if c in labeled_df.columns]
    if available_raw_cols:
        print("\n[+] Raw Macro Variable Means per Regime:")
        print(labeled_df.groupby('inferred_regime')[available_raw_cols].mean().round(4))

    print("\n[+] Equity Returns per Regime:")
    returns_profile = labeled_df.groupby('inferred_regime')[['equity_return']].agg(['mean', 'std', 'count'])
    returns_profile.columns = ['mean', 'std', 'count']
    print(returns_profile.round(5))
    
    print("="*60)


if __name__ == "__main__":
    DATA_INPUT = "data/processed/aligned_macro_dataset.csv"
    DATA_OUTPUT = "data/processed/regime_labeled_dataset.csv"

    FEATURE_COLS = [
        'growth_yoy_zscore',
        'cpi_yoy_zscore',
        'unemployment_delta_zscore',
        'yield_spread_delta_zscore',
        'credit_spread_delta_zscore',
    ]

    try:
        master_df = load_processed_data(DATA_INPUT)
        hmm_model, output_df = train_regime_hmm(master_df, n_components=4)
        output_model_diagnostics(hmm_model, output_df, FEATURE_COLS)
        check_regime_quality(output_df, hmm_model, FEATURE_COLS)
        
        output_df.to_csv(DATA_OUTPUT)
        print(f"\n[+] Success: Regime-labeled dataset written to: {DATA_OUTPUT}")
        
    except Exception as e:
        print(f"[-] Training Engine Failure: {str(e)}")