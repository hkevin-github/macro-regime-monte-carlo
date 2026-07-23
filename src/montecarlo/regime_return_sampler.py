"""
Regime Return Sampler
Samples market returns and inflation conditional on regime state with correlation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from .utils import yoy_to_monthly


class RegimeReturnSampler:
    def __init__(self, regime_stats: dict, sampling_method: Literal['gaussian', 'bootstrap'] = 'gaussian'):
        self.regime_stats = regime_stats
        self.sampling_method = sampling_method
        self.n_regimes = len(regime_stats)

        if sampling_method == 'bootstrap':
            self._load_historical_data()

    def _load_historical_data(self):
        df = pd.read_csv("data/processed/regime_labeled_dataset.csv", index_col='date', parse_dates=True)

        self.historical_returns = {}
        self.historical_inflation_yoy = {}

        for regime in range(self.n_regimes):
            regime_data = df[df['inferred_regime'] == regime]
            self.historical_returns[regime] = regime_data['equity_return'].values
            self.historical_inflation_yoy[regime] = regime_data['cpi_yoy'].values if 'cpi_yoy' in regime_data.columns else None

    def sample_returns(self,
                       regime_path: np.ndarray,
                       random_state: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample regime-conditioned returns with correlation.
        
        Returns:
            equity_returns: Array of monthly equity returns
            inflation_rates: Array of monthly inflation rates
        """
        rng = np.random.default_rng(random_state)

        n_periods = len(regime_path)
        equity_returns = np.zeros(n_periods)
        inflation_rates_monthly = np.zeros(n_periods)

        for t in range(n_periods):
            regime = int(regime_path[t])

            if self.sampling_method == 'gaussian':
                equity_returns[t], inflation_rates_monthly[t] = self._sample_correlated(regime, rng)
            else:
                equity_returns[t], inflation_rates_monthly[t] = self._sample_bootstrap(regime, rng)

        return equity_returns, inflation_rates_monthly

    def _sample_correlated(self, regime: int, rng: np.random.Generator) -> tuple[float, float]:
        """
        Sample equity and inflation returns with regime-specific correlation.
        Falls back to independent sampling if correlation matrix not available.
        """
        params = self.regime_stats[regime]
        
        # Check if correlation matrix available
        if params.get('correlation_matrix') is None or params.get('correlation_features') is None:
            return self._sample_gaussian_independent(regime, rng)
        
        try:
            # Build mean vector
            equity_mean = params['equity_mean']
            inflation_mean_yoy = params.get('inflation_mean', 0.02)
            
            # Convert YoY inflation to monthly for consistency
            inflation_mean_monthly = (1.0 + inflation_mean_yoy) ** (1.0 / 12.0) - 1.0
            
            mean_vector = np.array([equity_mean, inflation_mean_monthly])
            
            # Build standard deviation vector
            equity_std = params['equity_std']
            inflation_std_yoy = params.get('inflation_std', 0.01)
            # Approximate monthly std from annual (not perfect but reasonable)
            inflation_std_monthly = inflation_std_yoy / np.sqrt(12)
            
            std_vector = np.array([equity_std, inflation_std_monthly])
            
            # Get correlation matrix
            corr_matrix = np.array(params['correlation_matrix'])
            features = params['correlation_features']
            
            # Find indices for equity and inflation
            equity_idx = features.index('equity_return') if 'equity_return' in features else None
            cpi_idx = features.index('cpi_yoy') if 'cpi_yoy' in features else None
            
            if equity_idx is None or cpi_idx is None:
                return self._sample_gaussian_independent(regime, rng)
            
            # Extract 2x2 submatrix for equity and inflation
            indices = [equity_idx, cpi_idx]
            corr_2x2 = corr_matrix[np.ix_(indices, indices)]
            
            # Build covariance matrix: Σ = D @ R @ D
            D = np.diag(std_vector)
            cov_matrix = D @ corr_2x2 @ D
            
            # Sample from multivariate normal
            samples = rng.multivariate_normal(mean_vector, cov_matrix)
            
            equity_return = float(samples[0])
            inflation_monthly = float(samples[1])
            
            return equity_return, inflation_monthly
            
        except (np.linalg.LinAlgError, ValueError, IndexError) as e:
            # Fallback to independent sampling if anything goes wrong
            return self._sample_gaussian_independent(regime, rng)

    def _sample_gaussian_independent(self, regime: int, rng: np.random.Generator) -> tuple[float, float]:
        """
        Fallback: sample equity and inflation independently (no correlation).
        """
        params = self.regime_stats[regime]

        equity_return = rng.normal(params['equity_mean'], params['equity_std'])

        inflation_yoy = rng.normal(
            params.get('inflation_mean', 0.02),
            params.get('inflation_std', 0.01)
        )
        inflation_monthly = yoy_to_monthly(inflation_yoy)

        return float(equity_return), float(inflation_monthly)

    def _sample_bootstrap(self, regime: int, rng: np.random.Generator) -> tuple[float, float]:
        """
        Sample from historical returns within the regime (bootstrap).
        """
        historical_ret = self.historical_returns[regime]
        idx = rng.integers(0, len(historical_ret))
        equity_return = historical_ret[idx]

        if self.historical_inflation_yoy[regime] is not None:
            inflation_yoy = self.historical_inflation_yoy[regime][idx]
            inflation_monthly = yoy_to_monthly(inflation_yoy)
        else:
            params = self.regime_stats[regime]
            inflation_yoy = rng.normal(
                params.get('inflation_mean', 0.02),
                params.get('inflation_std', 0.01)
            )
            inflation_monthly = yoy_to_monthly(inflation_yoy)

        return float(equity_return), float(inflation_monthly)

    def sample_multiple_paths(self,
                              regime_paths: np.ndarray,
                              random_state: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample returns for multiple regime paths.
        
        Args:
            regime_paths: Array of shape (n_paths, n_periods) with regime IDs
            random_state: Random seed
            
        Returns:
            equity_returns: Array of shape (n_paths, n_periods)
            inflation_rates: Array of shape (n_paths, n_periods)
        """
        rng = np.random.default_rng(random_state)

        n_paths, n_periods = regime_paths.shape
        equity_returns = np.zeros((n_paths, n_periods))
        inflation_rates = np.zeros((n_paths, n_periods))

        for i in range(n_paths):
            seed = rng.integers(0, 2**32 - 1)
            equity_returns[i], inflation_rates[i] = self.sample_returns(
                regime_paths[i],
                random_state=int(seed)
            )

        return equity_returns, inflation_rates