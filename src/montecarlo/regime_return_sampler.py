"""
Regime Return Sampler
Samples market returns and inflation conditional on regime state.
Supports both Gaussian and historical bootstrap sampling methods.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal


def annual_to_monthly_rate(annual_rate: float) -> float:
    """Convert an annualized rate into an equivalent monthly rate."""
    if annual_rate is None:
        return None

    annual_rate = float(annual_rate)
    if annual_rate <= -1.0:
        return -1.0

    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


class RegimeReturnSampler:
    """
    Samples returns from regime-conditional distributions.
    """
    
    def __init__(self, regime_stats: dict, sampling_method: Literal['gaussian', 'bootstrap'] = 'gaussian'):
        """
        Initialize the return sampler.
        
        Args:
            regime_stats: dict mapping regime indices to their statistics
            sampling_method: 'gaussian' for parametric or 'bootstrap' for historical resampling
        """
        self.regime_stats = regime_stats
        self.sampling_method = sampling_method
        self.n_regimes = len(regime_stats)
        
        # For bootstrap method, load historical data
        if sampling_method == 'bootstrap':
            self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical returns by regime for bootstrap sampling."""
        df = pd.read_csv("data/processed/regime_labeled_dataset.csv", 
                        index_col='date', parse_dates=True)
        
        self.historical_returns = {}
        self.historical_inflation = {}
        
        for regime in range(self.n_regimes):
            regime_data = df[df['inferred_regime'] == regime]
            self.historical_returns[regime] = regime_data['equity_return'].values
            
            if 'cpi_yoy' in regime_data.columns:
                self.historical_inflation[regime] = regime_data['cpi_yoy'].values
            else:
                self.historical_inflation[regime] = None
    
    def sample_returns(self, 
                      regime_path: np.ndarray,
                      random_state: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample returns and inflation for a given regime path.
        
        Args:
            regime_path: array of regime indices for each period
            random_state: random seed for reproducibility
            
        Returns:
            Tuple of (equity_returns, inflation_rates) arrays
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_periods = len(regime_path)
        equity_returns = np.zeros(n_periods)
        inflation_rates = np.zeros(n_periods)
        
        for t in range(n_periods):
            regime = regime_path[t]
            
            if self.sampling_method == 'gaussian':
                equity_returns[t], inflation_rates[t] = self._sample_gaussian(regime)
            else:  # bootstrap
                equity_returns[t], inflation_rates[t] = self._sample_bootstrap(regime)
        
        return equity_returns, inflation_rates
    
    def _sample_gaussian(self, regime: int) -> tuple[float, float]:
        """Sample from Gaussian distribution using regime parameters."""
        params = self.regime_stats[regime]
        
        equity_return = np.random.normal(
            params['equity_mean'],
            params['equity_std']
        )

        annual_inflation_rate = np.random.normal(
            params.get('inflation_mean', 0.02),
            params.get('inflation_std', 0.01)
        )
        inflation_rate = annual_to_monthly_rate(annual_inflation_rate)
        
        return equity_return, inflation_rate
    
    def _sample_bootstrap(self, regime: int) -> tuple[float, float]:
        """Sample from historical returns within the regime."""
        historical_ret = self.historical_returns[regime]
        idx = np.random.randint(0, len(historical_ret))
        equity_return = historical_ret[idx]
        
        if self.historical_inflation[regime] is not None:
            annual_inflation_rate = self.historical_inflation[regime][idx]
            inflation_rate = annual_to_monthly_rate(annual_inflation_rate)
        else:
            # Fallback to Gaussian if no historical inflation
            params = self.regime_stats[regime]
            annual_inflation_rate = np.random.normal(
                params.get('inflation_mean', 0.02),
                params.get('inflation_std', 0.01)
            )
            inflation_rate = annual_to_monthly_rate(annual_inflation_rate)
        
        return equity_return, inflation_rate
    
    def sample_multiple_paths(self,
                             regime_paths: np.ndarray,
                             random_state: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample returns for multiple regime paths.
        
        Args:
            regime_paths: array of shape (n_paths, n_periods)
            random_state: random seed
            
        Returns:
            Tuple of (equity_returns, inflation_rates), each shape (n_paths, n_periods)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_paths, n_periods = regime_paths.shape
        equity_returns = np.zeros((n_paths, n_periods))
        inflation_rates = np.zeros((n_paths, n_periods))
        
        for i in range(n_paths):
            equity_returns[i], inflation_rates[i] = self.sample_returns(
                regime_paths[i],
                random_state=None
            )
        
        return equity_returns, inflation_rates


if __name__ == "__main__":
    # Test with dummy regime stats
    test_stats = {
        0: {'equity_mean': 0.08/12, 'equity_std': 0.15/np.sqrt(12), 
            'inflation_mean': 0.02/12, 'inflation_std': 0.01/np.sqrt(12)},
        1: {'equity_mean': -0.05/12, 'equity_std': 0.25/np.sqrt(12),
            'inflation_mean': 0.06/12, 'inflation_std': 0.02/np.sqrt(12)},
    }
    
    sampler = RegimeReturnSampler(test_stats, sampling_method='gaussian')
    
    # Sample for a test regime path
    test_path = np.array([0, 0, 0, 1, 1, 0, 0, 0])
    returns, inflation = sampler.sample_returns(test_path, random_state=42)
    
    print("Test regime path:", test_path)
    print("Sampled returns:", returns.round(4))
    print("Sampled inflation:", inflation.round(4))