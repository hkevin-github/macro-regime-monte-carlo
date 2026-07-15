"""
Regime Return Sampler
Samples market returns and inflation conditional on regime state.
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
        rng = np.random.default_rng(random_state)

        n_periods = len(regime_path)
        equity_returns = np.zeros(n_periods)
        inflation_rates_monthly = np.zeros(n_periods)

        for t in range(n_periods):
            regime = int(regime_path[t])

            if self.sampling_method == 'gaussian':
                equity_returns[t], inflation_rates_monthly[t] = self._sample_gaussian(regime, rng)
            else:
                equity_returns[t], inflation_rates_monthly[t] = self._sample_bootstrap(regime, rng)

        return equity_returns, inflation_rates_monthly

    def _sample_gaussian(self, regime: int, rng: np.random.Generator) -> tuple[float, float]:
        params = self.regime_stats[regime]

        equity_return = rng.normal(params['equity_mean'], params['equity_std'])

        inflation_yoy = rng.normal(
            params.get('inflation_mean', 0.02),
            params.get('inflation_std', 0.01)
        )
        inflation_monthly = yoy_to_monthly(inflation_yoy)

        return float(equity_return), float(inflation_monthly)

    def _sample_bootstrap(self, regime: int, rng: np.random.Generator) -> tuple[float, float]:
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