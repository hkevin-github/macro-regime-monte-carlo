"""
Tests for correlation modeling.
"""

import numpy as np
import pytest
from montecarlo.regime_assumptions import RegimeAssumptions
from montecarlo.regime_return_sampler import RegimeReturnSampler


def test_correlation_matrices_valid():
    """Verify all regime correlation matrices are valid."""
    assumptions = RegimeAssumptions()
    
    for regime_id, stats in assumptions.regime_stats.items():
        if stats.get('correlation_matrix') is None:
            continue
        
        corr = np.array(stats['correlation_matrix'])
        
        # Test 1: Square matrix
        assert corr.shape[0] == corr.shape[1], f"Regime {regime_id} corr not square"
        
        # Test 2: Symmetric
        assert np.allclose(corr, corr.T), f"Regime {regime_id} corr not symmetric"
        
        # Test 3: Diagonal = 1
        assert np.allclose(np.diag(corr), 1.0), f"Regime {regime_id} diag != 1"
        
        # Test 4: Positive definite
        eigenvalues = np.linalg.eigvals(corr)
        assert np.all(eigenvalues > -1e-10), f"Regime {regime_id} not pos def"
        
        # Test 5: Correlations in [-1, 1]
        assert np.all(corr >= -1.01) and np.all(corr <= 1.01), f"Regime {regime_id} corr out of bounds"


def test_multivariate_sampling_produces_correlation():
    """Verify sampled returns have correct correlation structure."""
    assumptions = RegimeAssumptions()
    sampler = RegimeReturnSampler(assumptions.regime_stats, sampling_method='gaussian')
    
    # Find a regime with correlation matrix
    test_regime = None
    for regime_id, stats in assumptions.regime_stats.items():
        if stats.get('correlation_matrix') is not None:
            test_regime = regime_id
            break
    
    if test_regime is None:
        pytest.skip("No regime with correlation matrix found")
    
    # Sample many returns from this regime
    n_samples = 10000
    regime_path = np.full(n_samples, test_regime, dtype=int)
    
    equity_returns, inflation = sampler.sample_returns(regime_path, random_state=42)
    
    # Compute empirical correlation
    empirical_corr = np.corrcoef(equity_returns, inflation)[0, 1]
    
    # Get expected correlation
    stats = assumptions.regime_stats[test_regime]
    features = stats['correlation_features']
    
    if 'equity_return' in features and 'cpi_yoy' in features:
        equity_idx = features.index('equity_return')
        cpi_idx = features.index('cpi_yoy')
        corr_matrix = np.array(stats['correlation_matrix'])
        expected_corr = corr_matrix[equity_idx, cpi_idx]
        
        # Should be close (within Monte Carlo error)
        assert np.abs(empirical_corr - expected_corr) < 0.05, \
            f"Empirical corr {empirical_corr:.3f} != expected {expected_corr:.3f}"


def test_regime_correlations_differ():
    """Verify different regimes have different correlation structures."""
    assumptions = RegimeAssumptions()
    
    correlations_by_regime = {}
    for regime_id, stats in assumptions.regime_stats.items():
        if stats.get('correlation_matrix') is not None:
            features = stats['correlation_features']
            if 'equity_return' in features and 'cpi_yoy' in features:
                equity_idx = features.index('equity_return')
                cpi_idx = features.index('cpi_yoy')
                corr_matrix = np.array(stats['correlation_matrix'])
                correlations_by_regime[regime_id] = corr_matrix[equity_idx, cpi_idx]
    
    if len(correlations_by_regime) < 2:
        pytest.skip("Need at least 2 regimes with correlation matrices")
    
    # Verify they're not all identical
    unique_corrs = len(set(np.round(list(correlations_by_regime.values()), 2)))
    assert unique_corrs > 1, "All regimes have identical correlations"
    
    print("\nRegime-specific equity-inflation correlations:")
    for regime_id, corr in correlations_by_regime.items():
        label = assumptions.regime_labels['regime_names'][regime_id]
        print(f"  {label}: {corr:.3f}")