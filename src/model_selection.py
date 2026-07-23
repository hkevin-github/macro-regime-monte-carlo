"""
HMM Model Selection - BIC/AIC Comparison with Structural Validation
Fits Gaussian HMMs with varying numbers of states and compares them using
information criteria AND structural quality checks to identify the optimal model.
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
    if covariance_type != "diag":
        raise ValueError("This helper currently assumes diag covariance.")
    return (n_states - 1) + n_states * (n_states - 1) + n_states * n_features + n_states * n_features


def validate_model_structure(model: GaussianHMM, X: np.ndarray, n_states: int) -> dict:
    """
    Validate structural quality of fitted HMM to detect overfitting.
    
    Returns dict with:
        - is_valid: bool (no critical issues)
        - issues: list of critical problems
        - warnings: list of minor concerns
        - metrics: dict of quality metrics
    """
    issues = []
    warnings = []
    metrics = {}
    
    # 1. Check regime persistence
    persistence = np.diag(model.transmat_)
    metrics['min_persistence'] = float(persistence.min())
    metrics['mean_persistence'] = float(persistence.mean())
    
    # CRITICAL: Deterministic states (always transition out)
    if np.any(persistence < 0.05):
        deterministic_states = np.where(persistence < 0.05)[0].tolist()
        issues.append(f"Deterministic states (persistence < 0.05): {deterministic_states}")
    
    # WARNING: Low persistence states
    elif np.any(persistence < 0.65):
        unstable_states = np.where(persistence < 0.65)[0].tolist()
        warnings.append(f"Low persistence states (< 0.65): {unstable_states}")
    
    # 2. Check regime occupancy
    states = model.predict(X)
    regime_counts = pd.Series(states).value_counts().sort_index()
    metrics['min_regime_count'] = int(regime_counts.min())
    metrics['regime_counts'] = regime_counts.to_dict()
    
    n_obs = len(X)
    
    # CRITICAL: Must have at least 1% of data (allows rare crisis regimes)
    critical_min = max(10, int(0.01 * n_obs))
    
    # WARNING: Prefer at least 3% of data
    warning_min = max(25, int(0.03 * n_obs))
    
    if regime_counts.min() < critical_min:
        tiny_regimes = regime_counts[regime_counts < critical_min].index.tolist()
        issues.append(f"Critically small regimes (< {critical_min} obs): {tiny_regimes} "
                     f"with counts {regime_counts[tiny_regimes].tolist()}")
    elif regime_counts.min() < warning_min:
        small_regimes = regime_counts[regime_counts < warning_min].index.tolist()
        warnings.append(f"Small regimes (< {warning_min} obs): {small_regimes} "
                       f"with counts {regime_counts[small_regimes].tolist()}")
    
    # 3. Check regime separation
    regime_means = np.array([model.means_[state] for state in range(n_states)])
    
    min_distance = float('inf')
    close_pairs = []
    duplicate_pairs = []
    
    for i in range(n_states):
        for j in range(i + 1, n_states):
            dist = np.linalg.norm(regime_means[i] - regime_means[j])
            min_distance = min(min_distance, dist)
            
            if dist < 0.5:  # CRITICAL: Near-duplicates
                duplicate_pairs.append((i, j, float(dist)))
            elif dist < 1.0:  # WARNING: Close regimes
                close_pairs.append((i, j, float(dist)))
    
    metrics['min_regime_distance'] = float(min_distance)
    
    if duplicate_pairs:
        issues.append(f"Near-duplicate regimes (distance < 0.5): {duplicate_pairs}")
    elif close_pairs:
        warnings.append(f"Close regimes (distance < 1.0): {close_pairs}")
    
    # 4. Check for ping-pong patterns
    for i in range(n_states):
        for j in range(i + 1, n_states):
            prob_i_to_j = model.transmat_[i, j]
            prob_j_to_i = model.transmat_[j, i]
            
            if prob_i_to_j > 0.7 and prob_j_to_i > 0.7:
                issues.append(f"Ping-pong pattern between states {i} and {j} "
                            f"(prob {i}→{j}: {prob_i_to_j:.3f}, prob {j}→{i}: {prob_j_to_i:.3f})")
    
    is_valid = len(issues) == 0
    
    return {
        'is_valid': is_valid,
        'issues': issues,
        'warnings': warnings,
        'metrics': metrics
    }


def fit_and_score_models(df: pd.DataFrame, feature_cols: list, min_states: int = 2, max_states: int = 6) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 70)
    print("HMM MODEL SELECTION: BIC/AIC WITH STRUCTURAL VALIDATION")
    print("=" * 70)

    X = df[feature_cols].dropna().values
    n_samples, n_features = X.shape
    results = []
    validation_results = {}

    for n in range(min_states, max_states + 1):
        print(f"\n[*] Fitting {n}-state model...")

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
            
            # Structural validation
            validation = validate_model_structure(model, X, n)
            validation_results[n] = validation
            
            # Report validation results
            if not validation['is_valid']:
                print(f"    ⚠️  Critical structural issues:")
                for issue in validation['issues']:
                    print(f"        • {issue}")
            elif validation['warnings']:
                print(f"    ⚠️  Minor warnings:")
                for warning in validation['warnings']:
                    print(f"        • {warning}")
            else:
                print(f"    ✓ Model structure is valid")
            
            print(f"    Metrics: persistence={validation['metrics']['min_persistence']:.3f}, "
                  f"min_count={validation['metrics']['min_regime_count']}, "
                  f"min_distance={validation['metrics']['min_regime_distance']:.3f}")

            results.append({
                'n_states': n,
                'BIC': bic,
                'AIC': aic,
                'log_likelihood': log_likelihood,
                'free_params': k,
                'converged': converged,
                'structurally_valid': validation['is_valid'],
                'num_issues': len(validation['issues']),
                'num_warnings': len(validation['warnings']),
                'min_persistence': validation['metrics']['min_persistence'],
                'min_regime_count': validation['metrics']['min_regime_count'],
                'min_regime_distance': validation['metrics']['min_regime_distance'],
            })

        except Exception as e:
            print(f"    ⚠️  Failed to fit {n}-state model: {str(e)}")
            continue

    if not results:
        raise RuntimeError("All models failed to fit. Check your data.")

    comparison = pd.DataFrame(results)
    comparison['delta_BIC'] = comparison['BIC'] - comparison['BIC'].min()
    comparison['delta_AIC'] = comparison['AIC'] - comparison['AIC'].min()

    return comparison, validation_results


def interpret_results(comparison: pd.DataFrame, validation_results: dict) -> dict:
    print("\n[+] Model Selection Results:")
    print(comparison.to_string(index=False))

    valid_models = comparison[comparison['structurally_valid']]
    
    if len(valid_models) == 0:
        print("\n⚠️  WARNING: No structurally valid models found!")
        print("    Recommending best BIC model, but manual review required.")
        recommendation = int(comparison.loc[comparison['BIC'].idxmin(), 'n_states'])
        recommendation_reason = "best_bic_no_valid_models"
    else:
        best_valid_bic = int(valid_models.loc[valid_models['BIC'].idxmin(), 'n_states'])
        best_overall_bic = int(comparison.loc[comparison['BIC'].idxmin(), 'n_states'])
        
        print(f"\n[+] Best structurally valid model by BIC: {best_valid_bic} states")
        print(f"[+] Best overall BIC (ignoring structure): {best_overall_bic} states")
        
        if best_valid_bic == best_overall_bic:
            print(f"\n✓ Best BIC model is also structurally valid")
            recommendation = best_valid_bic
            recommendation_reason = "best_bic_and_valid"
        else:
            bic_diff = comparison.loc[comparison['n_states'] == best_valid_bic, 'BIC'].values[0] - \
                      comparison.loc[comparison['n_states'] == best_overall_bic, 'BIC'].values[0]
            print(f"\n⚠️  Best BIC model ({best_overall_bic} states) has structural issues")
            print(f"    BIC difference: {bic_diff:.2f}")
            print(f"    → Recommending best valid model ({best_valid_bic} states)")
            recommendation = best_valid_bic
            recommendation_reason = "best_valid_despite_bic"

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDANCE")
    print("=" * 70)
    print(f"Recommended: {recommendation} states ({recommendation_reason})")
    
    # Show validation details
    if recommendation in validation_results:
        val = validation_results[recommendation]
        if val['is_valid']:
            if val['warnings']:
                print(f"⚠️  Recommended model has minor warnings:")
                for warning in val['warnings']:
                    print(f"    • {warning}")
            else:
                print(f"✓ Recommended model has no structural issues")
        else:
            print(f"⚠️  Recommended model has critical issues:")
            for issue in val['issues']:
                print(f"    • {issue}")
    
    # Show why best BIC was rejected
    best_overall_bic = int(comparison.loc[comparison['BIC'].idxmin(), 'n_states'])
    if best_overall_bic != recommendation and best_overall_bic in validation_results:
        print(f"\nWhy {best_overall_bic}-state model was not recommended:")
        val = validation_results[best_overall_bic]
        for issue in val['issues']:
            print(f"    • {issue}")

    print("=" * 70)

    return {
        'recommended': recommendation,
        'recommendation_reason': recommendation_reason,
        'best_bic': int(comparison.loc[comparison['BIC'].idxmin(), 'n_states']),
        'best_aic': int(comparison.loc[comparison['AIC'].idxmin(), 'n_states']),
        'best_valid_bic': int(valid_models.loc[valid_models['BIC'].idxmin(), 'n_states']) if len(valid_models) > 0 else None,
        'comparison_table': comparison,
        'validation_results': validation_results,
    }


def plot_information_criteria(comparison: pd.DataFrame, output_path: str = "notebook/model_selection.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    valid = comparison[comparison['structurally_valid']]
    invalid = comparison[~comparison['structurally_valid']]
    
    # BIC plot
    axes[0].plot(comparison['n_states'], comparison['BIC'], 'o-', linewidth=2, markersize=8, color='#1F77B4')
    
    if len(valid) > 0:
        axes[0].scatter(valid['n_states'], valid['BIC'], s=150, marker='o', 
                       facecolors='none', edgecolors='green', linewidths=2, label='Valid', zorder=5)
    if len(invalid) > 0:
        axes[0].scatter(invalid['n_states'], invalid['BIC'], s=150, marker='x', 
                       color='red', linewidths=2, label='Issues', zorder=5)
    
    axes[0].axvline(comparison.loc[comparison['BIC'].idxmin(), 'n_states'],
                    color='red', linestyle='--', alpha=0.5, label='Best BIC')
    if len(valid) > 0:
        axes[0].axvline(valid.loc[valid['BIC'].idxmin(), 'n_states'],
                       color='green', linestyle='--', alpha=0.5, label='Best valid')
    
    axes[0].set_xlabel('Number of States')
    axes[0].set_ylabel('BIC (lower is better)')
    axes[0].set_title('Bayesian Information Criterion')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # AIC plot
    axes[1].plot(comparison['n_states'], comparison['AIC'], 'o-', linewidth=2, markersize=8, color='#FF7F0E')
    
    if len(valid) > 0:
        axes[1].scatter(valid['n_states'], valid['AIC'], s=150, marker='o', 
                       facecolors='none', edgecolors='green', linewidths=2, label='Valid', zorder=5)
    if len(invalid) > 0:
        axes[1].scatter(invalid['n_states'], invalid['AIC'], s=150, marker='x', 
                       color='red', linewidths=2, label='Issues', zorder=5)
    
    axes[1].axvline(comparison.loc[comparison['AIC'].idxmin(), 'n_states'],
                    color='red', linestyle='--', alpha=0.7, label='Best AIC')
    axes[1].set_xlabel('Number of States')
    axes[1].set_ylabel('AIC (lower is better)')
    axes[1].set_title('Akaike Information Criterion')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Structural quality plot
    axes[2].plot(comparison['n_states'], comparison['min_persistence'], 'o-', 
                linewidth=2, markersize=8, color='#2CA02C', label='Min persistence')
    axes[2].axhline(0.65, color='orange', linestyle='--', alpha=0.7, label='Stability threshold')
    axes[2].axhline(0.05, color='red', linestyle='--', alpha=0.7, label='Deterministic threshold')
    axes[2].set_xlabel('Number of States')
    axes[2].set_ylabel('Minimum Persistence')
    axes[2].set_title('Model Structural Quality')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_ylim(0, 1)

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
            'recommendation_reason': recommendation['recommendation_reason'],
            'best_bic': recommendation['best_bic'],
            'best_aic': recommendation['best_aic'],
            'best_valid_bic': recommendation['best_valid_bic'],
        }, f, indent=2)
    print(f"[+] Recommendation saved to: {rec_path}")
    
    val_path = "data/models/model_validation_details.json"
    validation_serializable = {}
    for n_states, val in recommendation['validation_results'].items():
        validation_serializable[str(n_states)] = {
            'is_valid': val['is_valid'],
            'issues': val['issues'],
            'warnings': val['warnings'],
            'metrics': val['metrics']
        }
    
    with open(val_path, "w") as f:
        json.dump(validation_serializable, f, indent=2)
    print(f"[+] Validation details saved to: {val_path}")


if __name__ == "__main__":
    DATA_INPUT = "data/processed/aligned_macro_dataset.csv"

    try:
        df = load_processed_data(DATA_INPUT)
        comparison, validation_results = fit_and_score_models(df, FEATURE_COLS, min_states=2, max_states=6)
        recommendation = interpret_results(comparison, validation_results)
        plot_information_criteria(comparison)
        save_results(comparison, recommendation)

        print("\n" + "=" * 70)
        print("✓ MODEL SELECTION COMPLETE")
        print("=" * 70)
        print(f"\nRecommended number of states: {recommendation['recommended']}")
        print(f"Reason: {recommendation['recommendation_reason']}")
        print("\nNext step: Run hmm_regime_engine.py")
    except Exception as e:
        print(f"[-] Model Selection Failure: {str(e)}")
        raise