"""
Regime Inference Engine - Hidden Markov Model Training
Trains a Gaussian Hidden Markov Model directly on the monthly macro
observation matrix produced by the ingestion pipeline.

State count: n_components=4, not 3. BIC model selection (see
select_n_components_by_bic()) showed 4 states fit meaningfully better than 3,
and 3 states did not even beat 2 -- forcing 3 states onto this data produced
two near-duplicate regimes that swapped labels almost every month. Regimes are
initialized unsupervised (k-means-based) and labeled after fitting from their
empirical feature means, not assumed in advance.

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
    return pd.read_csv(file_path, index_col='date', parse_dates=True)


def select_n_components_by_bic(X: np.ndarray, candidates=(2, 3, 4), covariance_type='diag', random_state=42) -> dict:
    """
    Fits a GaussianHMM for each candidate n_components and returns BIC scores,
    so the state count is chosen from evidence rather than assumed. Lower BIC
    is better (penalizes extra parameters, not just log-likelihood).
    """
    results = {}
    for k in candidates:
        trial_model = GaussianHMM(
            n_components=k,
            covariance_type=covariance_type,
            n_iter=1000,
            random_state=random_state,
        )
        trial_model.fit(X)
        log_likelihood = trial_model.score(X)
        n_features = X.shape[1]
        # Free parameters: transition matrix (k*(k-1)) + initial probs (k-1)
        # + means (k*n_features) + covariances (k*n_features for 'diag')
        n_params = k * (k - 1) + (k - 1) + k * n_features + k * n_features
        n_samples = X.shape[0]
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        results[k] = {'log_likelihood': log_likelihood, 'bic': bic, 'n_params': n_params}
    return results


def train_regime_hmm(df: pd.DataFrame, n_components: int = 4) -> tuple[GaussianHMM, pd.DataFrame]:
    """
    Trains a Gaussian HMM on the monthly macro observation matrix.

    n_components=4 (not 3) because BIC model selection empirically favored 4
    states over 3 by a wide margin, and 3 states did not even beat 2 -- see
    select_n_components_by_bic() output. Manually-seeded means (previously
    hand-picked for a 3-state Expansion/Stagflation/Recession story) are
    dropped here: with 4 states genuinely supported by the data, forcing a
    3-regime narrative onto it was producing two near-duplicate states that
    swapped labels almost every month (confirmed via transition matrix:
    ~97-99% cross-transition probability between the two duplicate states).
    Instead this lets hmmlearn's default k-means-based initialization find
    the 4 clusters on its own, and the regimes are labeled AFTER fitting,
    from their empirical mean feature vectors -- not assumed in advance.

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

    # 1. Data is already monthly (one row per month, month-end indexed) -- no
    #    resampling/downsampling needed anymore. Just select the feature matrix.
    monthly_df = df[feature_cols].dropna()
    X_monthly = monthly_df.values

    # 1b. BIC model-selection check: confirms the chosen n_components against
    # neighboring candidates, printed for visibility every run.
    print("[*] Running BIC model-selection check across candidate state counts...")
    bic_results = select_n_components_by_bic(X_monthly, candidates=(2, 3, 4, 5), covariance_type='diag')
    for k, stats in bic_results.items():
        print(f"    n_components={k}: log-likelihood={stats['log_likelihood']:.2f}, "
              f"BIC={stats['bic']:.2f}, params={stats['n_params']}")
    best_k = min(bic_results, key=lambda k: bic_results[k]['bic'])
    print(f"[*] BIC-preferred state count: {best_k} (training n_components={n_components} as configured)")

    # 2. Instantiate with fully unsupervised initialization -- no hand-seeded
    # means_/covars_. init_params="stmc" initializes start probs (s),
    # transitions (t), means (m), and covars (c) all from the data itself
    # (hmmlearn defaults to k-means for mean init), letting the natural
    # cluster structure emerge rather than imposing an assumed narrative.
    model = GaussianHMM(
        n_components=n_components,
        covariance_type='diag',
        n_iter=1000,
        init_params="stmc",
        random_state=42
    )

    print("[*] Calibration Phase: Training Monthly HMM (unsupervised init)...")
    model.fit(X_monthly)

    # 3. Decode monthly paths directly -- this IS the full-resolution output now.
    monthly_states = model.predict(X_monthly)
    labeled_df = monthly_df.copy()
    labeled_df['inferred_regime'] = monthly_states

    # 4. Reattach the remaining (non-feature) columns, e.g. equity_return and
    #    the raw Fama-French/macro columns, aligned on the same monthly index.
    other_cols = [c for c in df.columns if c not in feature_cols]
    labeled_df = labeled_df.join(df[other_cols], how='left')

    return model, labeled_df


def output_model_diagnostics(model: GaussianHMM, labeled_df: pd.DataFrame, feature_cols: list):
    n_components = model.n_components
    print("\n" + "="*50)
    print("             CORRECTED HMM ENGINE DIAGNOSTICS             ")
    print("="*50)

    print(f"\n[+] Monthly Observations Allotted per Regime (n_components={n_components}):")
    print(labeled_df['inferred_regime'].value_counts().sort_index())

    print("\n[+] Monthly Transition Matrix (A):")
    transition_matrix = pd.DataFrame(
        model.transmat_,
        index=[f"Regime_{i}" for i in range(n_components)],
        columns=[f"To_Regime_{i}" for i in range(n_components)]
    )
    print(transition_matrix.round(4))

    # Z-scored feature means per regime -- this is the primary basis for
    # naming/interpreting each regime, since it's directly comparable across
    # features (unlike raw units below).
    print("\n[+] Empirical Z-Scored Feature Means per Regime (use this to name regimes):")
    print(labeled_df.groupby('inferred_regime')[feature_cols].mean().round(4))

    raw_feature_cols = ['growth_yoy', 'cpi_yoy', 'unemployment_delta', 'yield_spread_delta', 'credit_spread_delta']
    available_raw_cols = [c for c in raw_feature_cols if c in labeled_df.columns]
    if available_raw_cols:
        print("\n[+] Empirical Means of Raw Macro Variables per Regime Vector:")
        print(labeled_df.groupby('inferred_regime')[available_raw_cols].mean().round(4))

    print("\n[+] Market Returns Profile per Regime Vector:")
    print(labeled_df.groupby('inferred_regime')[['equity_return']].agg(['mean', 'std', 'count']).round(5))
    print("="*50)


if __name__ == "__main__":
    DATA_INPUT = "data/processed/aligned_macro_dataset_v1.csv"
    DATA_OUTPUT = "data/processed/regime_labeled_dataset_v1.csv"

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
        output_df.to_csv(DATA_OUTPUT)
        print(f"\n[+] Success: Clean regime dataset written to: {DATA_OUTPUT}")
    except Exception as e:
        print(f"[-] Training Engine Failure: {str(e)}")