"""
HMM Model Selection - BIC/AIC Comparison
Fits Gaussian HMMs with varying numbers of states and compares them using
information criteria to identify the optimal model complexity.
"""

import os
import json
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt


FEATURE_COLS = [
    'growth_yoy_zscore',
    'cpi_yoy_zscore',
    'unemployment_delta_zscore',
    'yield_spread_delta_zscore',
    'credit_spread_delta_zscore',
]


def load_processed_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data matrix not discovered at {file_path}")
    print(f"[*] Loading data from: {file_path}")
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df = df.sort_index()
    if df.index.duplicated().any():
        print("⚠️  WARNING: Duplicate dates found. Keeping last occurrence.")
        df = df[~df.index.duplicated(keep='last')]
    return df


def _num_free_params_hmm(n_states: int, n_features: int, covariance_type: str = "diag") -> int:
    """
    Rough count of free parameters for Gaussian HMM.
    """
    # start probabilities: n_states - 1
    # transition matrix: n_states * (n_states - 1)
    # means: n_states * n_features
    # covars: diag => n_states * n_features
    if covariance_type != "diag":
        raise ValueError("This helper currently assumes diag covariance.")
    return (n_states - 1) + n_states * (n_states - 1) + n_states * n_features + n_states * n_features


def fit_and_score_models(df: pd.DataFrame, feature_cols: list, min_states: int = 2, max_states: int = 6) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("HMM MODEL SELECTION: BIC/AIC COMPARISON")
    print("=" * 70)

    X = df[feature_cols].dropna().values
    n_samples, n_features = X.shape
    results = []

    for n in range(min_states, max_states + 1):
        print(f"[*] Fitting {n}-state model...")

        model = GaussianHMM(
            n_components=n,
            covariance_type='diag',
            n_iter=1000,
            init_params="stmc",
            random_state=42
        )

        try:
            model.fit(X)
            log_likelihood = model.score(X)
            k = _num_free_params_hmm(n, n_features, covariance_type="diag")
            bic = -2 * log_likelihood + k * np.log(n_samples)
            aic = -2 * log_likelihood + 2 * k

            converged = model.monitor_.converged

            results.append({
                'n_states': n,
                'BIC': bic,
                'AIC': aic,
                'log_likelihood': log_likelihood,
                'free_params': k,
                'converged': converged
            })

        except Exception as e:
            print(f"    ⚠️  Failed to fit {n}-state model: {str(e)}")
            continue

    if not results:
        raise RuntimeError("All models failed to fit. Check your data.")

    comparison = pd.DataFrame(results)
    comparison['delta_BIC'] = comparison['BIC'] - comparison['BIC'].min()
    comparison['delta_AIC'] = comparison['AIC'] - comparison['AIC'].min()

    return comparison


def interpret_results(comparison: pd.DataFrame) -> dict:
    print("\n[+] Model Selection Results:")
    print(comparison.to_string(index=False))

    best_bic = int(comparison.loc[comparison['BIC'].idxmin(), 'n_states'])
    best_aic = int(comparison.loc[comparison['AIC'].idxmin(), 'n_states'])

    print(f"\n[+] Optimal states by BIC: {best_bic}")
    print(f"[+] Optimal states by AIC: {best_aic}")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDANCE")
    print("=" * 70)

    if best_bic == best_aic:
        print(f"✓ Both criteria agree: {best_bic} states is optimal")
        recommendation = best_bic
    else:
        print(f"⚠️  Criteria disagree: BIC prefers {best_bic}, AIC prefers {best_aic}")
        print("    Using BIC recommendation for interpretability.")
        recommendation = best_bic

    bic_sorted = comparison.sort_values('BIC')
    if len(bic_sorted) >= 2:
        print(f"\n[+] Best BIC margin vs next best: {bic_sorted.iloc[1]['BIC'] - bic_sorted.iloc[0]['BIC']:.2f}")

    if 'converged' in comparison.columns and (~comparison['converged']).any():
        print("\n⚠️  WARNING: Some models did not converge:")
        print(comparison.loc[~comparison['converged'], ['n_states']].to_string(index=False))

    print("=" * 70)

    return {
        'recommended': recommendation,
        'best_bic': best_bic,
        'best_aic': best_aic,
        'comparison_table': comparison
    }


def plot_information_criteria(comparison: pd.DataFrame, output_path: str = "notebook/model_selection.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(comparison['n_states'], comparison['BIC'], 'o-', linewidth=2, markersize=8, color='#1F77B4')
    axes[0].axvline(comparison.loc[comparison['BIC'].idxmin(), 'n_states'],
                    color='red', linestyle='--', alpha=0.7, label='Optimal')
    axes[0].set_xlabel('Number of States')
    axes[0].set_ylabel('BIC (lower is better)')
    axes[0].set_title('Bayesian Information Criterion')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(comparison['n_states'], comparison['AIC'], 'o-', linewidth=2, markersize=8, color='#FF7F0E')
    axes[1].axvline(comparison.loc[comparison['AIC'].idxmin(), 'n_states'],
                    color='red', linestyle='--', alpha=0.7, label='Optimal')
    axes[1].set_xlabel('Number of States')
    axes[1].set_ylabel('AIC (lower is better)')
    axes[1].set_title('Akaike Information Criterion')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Model selection plot saved to: {output_path}")
    plt.close()


def save_results(comparison: pd.DataFrame, recommendation: dict):
    os.makedirs("data/models", exist_ok=True)

    comparison_path = "data/models/model_selection_results.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"[+] Model selection results saved to: {comparison_path}")

    rec_path = "data/models/model_selection_recommendation.json"
    with open(rec_path, "w") as f:
        json.dump({
            'recommended_states': recommendation['recommended'],
            'best_bic': recommendation['best_bic'],
            'best_aic': recommendation['best_aic']
        }, f, indent=2)
    print(f"[+] Recommendation saved to: {rec_path}")


if __name__ == "__main__":
    DATA_INPUT = "data/processed/aligned_macro_dataset.csv"

    try:
        df = load_processed_data(DATA_INPUT)
        comparison = fit_and_score_models(df, FEATURE_COLS, min_states=2, max_states=6)
        recommendation = interpret_results(comparison)
        plot_information_criteria(comparison)
        save_results(comparison, recommendation)

        print("\n" + "=" * 70)
        print("✓ MODEL SELECTION COMPLETE")
        print("=" * 70)
        print(f"\nRecommended number of states: {recommendation['recommended']}")
        print("Next step: Run hmm_regime_engine.py")
    except Exception as e:
        print(f"[-] Model Selection Failure: {str(e)}")
        raise