"""
Regime-Aware Multi-Asset Portfolio Monte Carlo Simulator
Handles arbitrary portfolios with multiple asset classes with correlation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from montecarlo.regime_assumptions import RegimeAssumptions
from montecarlo.regime_path_simulator import RegimePathSimulator


class RegimeMultiAssetSimulator:
    """
    Simulates portfolio returns across multiple asset classes with regime-aware dynamics
    and correlation modeling.
    """
    
    # Map portfolio asset classes to regime statistics.
    #
    # International_Equity -> its own regime-conditioned distribution, built from
    #   Ken French Developed ex-US factors (real data, ~1990+). See data_pipeline.py.
    # Emerging_Markets_Equity -> folded into International_Equity. There is no free,
    #   long-history, programmatically-fetchable EM total-return series, so rather
    #   than fabricate one, EM exposure is treated as International_Equity risk.
    # Real_Estate -> its own distribution, built from the Case-Shiller home price
    #   index (price appreciation only, NOT REIT total return -- see caveats in
    #   data_pipeline.py and hmm_regime_engine.py).
    # Commodities -> its own distribution, built from PPIACO (a wholesale price
    #   index used as a constructed proxy, NOT an investable commodities index).
    # Any of the above falls back to 'equity' stats for a given regime if that
    # regime doesn't have enough historical months for a reliable estimate
    # (see get_asset_class_params / FALLBACK_ASSET_CLASS below).
    ASSET_CLASS_MAPPING = {
        'US_Equity': 'equity',
        'International_Equity': 'intl_equity',
        'Emerging_Markets_Equity': 'intl_equity',
        'US_Bonds': 'bonds',
        'International_Bonds': 'bonds',
        'Real_Estate': 'real_estate',
        'Commodities': 'commodities',
        'Cash': 'cash',
        'Unknown': 'cash',
    }

    # Stats keys that require a reliability fallback (to 'equity') when a regime
    # lacks sufficient historical observations for that asset class.
    FALLBACK_ASSET_CLASS = {
        'intl_equity': 'equity',
        'real_estate': 'equity',
        'commodities': 'equity',
    }

    # Maps a stats_key to the corresponding column name used in the regime
    # correlation matrix's `correlation_features` list.
    FEATURE_NAME_MAP = {
        'equity': 'equity_return',
        'bonds': 'bond_return',
        'intl_equity': 'intl_equity_return',
        'real_estate': 'real_estate_return',
        'commodities': 'commodities_return',
    }
    
    def __init__(self, models_dir: str = "data/models"):
        """Initialize with regime model."""
        self.assumptions = RegimeAssumptions(models_dir=models_dir)
        self.regime_labels = self.assumptions.regime_labels['regime_names']
        self.regime_stats = self.assumptions.regime_stats
        self.transition_matrix = self.assumptions.transition_matrix
        
        self.regime_sim = RegimePathSimulator(
            transition_matrix=self.transition_matrix,
            regime_names=self.regime_labels
        )
        
        print(f"[+] Loaded {self.assumptions.n_regimes}-state regime model")
    
    def get_asset_class_params(self, asset_class: str, regime: int) -> Dict[str, float]:
        """
        Get return parameters for an asset class in a given regime.
        
        Args:
            asset_class: Portfolio asset class (e.g., 'US_Equity', 'US_Bonds')
            regime: Regime ID
            
        Returns:
            Dict with 'mean' and 'std' keys
        """
        stats_key = self.ASSET_CLASS_MAPPING.get(asset_class, 'cash')
        return self._get_params_by_stats_key(stats_key, regime)

    def _get_params_by_stats_key(self, stats_key: str, regime: int) -> Dict[str, float]:
        """
        Internal lookup by stats_key (rather than portfolio asset class), used both
        by get_asset_class_params and by the fallback path for extended asset classes
        that lack reliable regime-specific stats.
        """
        regime_stats = self.regime_stats[regime]

        if stats_key == 'equity':
            return {
                'mean': regime_stats['equity_mean'],
                'std': regime_stats['equity_std']
            }
        elif stats_key == 'bonds':
            # Use bond statistics if available, otherwise use proxy
            if regime_stats.get('bond_mean') is not None:
                return {
                    'mean': regime_stats['bond_mean'],
                    'std': regime_stats['bond_std']
                }
            else:
                # Fallback proxy model
                inflation_yoy = regime_stats.get('inflation_mean', 0.02)
                inflation_monthly = (1.0 + inflation_yoy) ** (1.0 / 12.0) - 1.0
                
                base_bond_rate = 0.04 / 12
                inflation_sensitivity = 1.5
                
                mean = base_bond_rate - inflation_sensitivity * inflation_monthly
                std = 0.01
                
                return {'mean': mean, 'std': std}
        elif stats_key == 'cash':
            return {'mean': 0.0, 'std': 0.0}
        elif stats_key in ('intl_equity', 'real_estate', 'commodities'):
            mean_key = f"{stats_key}_mean"
            std_key = f"{stats_key}_std"
            reliable_key = f"{stats_key}_is_reliable"

            if regime_stats.get(reliable_key) and regime_stats.get(mean_key) is not None:
                return {'mean': regime_stats[mean_key], 'std': regime_stats[std_key]}

            # This regime doesn't have enough historical months of this asset class's
            # proxy series to trust a regime-specific estimate (common for regimes
            # occurring before the proxy's inception -- e.g. pre-1990 for
            # international, pre-1987 for real estate). Fall back to equity.
            fallback_key = self.FALLBACK_ASSET_CLASS.get(stats_key, 'equity')
            return self._get_params_by_stats_key(fallback_key, regime)
        else:
            return {'mean': 0.0, 'std': 0.0}
    
    def sample_returns(self,
                      asset_class_weights: Dict[str, float],
                      regime_path: np.ndarray,
                      random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample portfolio returns along a regime path with correlation.
        
        Args:
            asset_class_weights: Dict of {asset_class: weight}
            regime_path: Array of regime IDs for each period
            random_state: Random seed
            
        Returns:
            Array of portfolio returns for each period
        """
        rng = np.random.default_rng(random_state)
        n_periods = len(regime_path)
        
        portfolio_returns = np.zeros(n_periods)
        
        for t in range(n_periods):
            regime = int(regime_path[t])
            regime_stats = self.regime_stats[regime]
            
            # Check if we have correlation matrix
            if regime_stats.get('correlation_matrix') is not None:
                period_return = self._sample_correlated_portfolio(
                    asset_class_weights, regime, rng
                )
            else:
                # Fallback to independent sampling
                period_return = self._sample_independent_portfolio(
                    asset_class_weights, regime, rng
                )
            
            portfolio_returns[t] = period_return
        
        return portfolio_returns

    def _sample_correlated_portfolio(self,
                                     asset_class_weights: Dict[str, float],
                                     regime: int,
                                     rng: np.random.Generator) -> float:
        """
        Sample portfolio return with correlation between asset classes.
        """
        regime_stats = self.regime_stats[regime]
        
        # Get available features from correlation matrix
        features = regime_stats.get('correlation_features', [])
        corr_matrix = np.array(regime_stats['correlation_matrix'])
        
        # Build mean and std vectors
        means = []
        stds = []
        
        for feature in features:
            if feature == 'equity_return':
                means.append(regime_stats['equity_mean'])
                stds.append(regime_stats['equity_std'])
            elif feature == 'bond_return':
                means.append(regime_stats.get('bond_mean', 0.003))
                stds.append(regime_stats.get('bond_std', 0.01))
            elif feature == 'cpi_yoy':
                inflation_yoy = regime_stats.get('inflation_mean', 0.02)
                inflation_monthly = (1.0 + inflation_yoy) ** (1.0 / 12.0) - 1.0
                means.append(inflation_monthly)
                stds.append(regime_stats.get('inflation_std', 0.01) / np.sqrt(12))
            elif feature in ('intl_equity_return', 'real_estate_return', 'commodities_return'):
                stats_key = feature.replace('_return', '')
                # Guaranteed present and reliable here, since compute_regime_correlation_matrix
                # only includes a column if it met MIN_EXTENDED_OBSERVATIONS within this regime.
                means.append(regime_stats.get(f'{stats_key}_mean', regime_stats['equity_mean']))
                stds.append(regime_stats.get(f'{stats_key}_std', regime_stats['equity_std']))
        
        mean_vector = np.array(means)
        std_vector = np.array(stds)
        
        # Build covariance matrix
        D = np.diag(std_vector)
        cov_matrix = D @ corr_matrix @ D
        
        try:
            # Sample from multivariate normal
            samples = rng.multivariate_normal(mean_vector, cov_matrix)
            
            # Map samples back to asset classes
            period_return = 0.0
            for asset_class, weight in asset_class_weights.items():
                stats_key = self.ASSET_CLASS_MAPPING.get(asset_class, 'cash')

                if stats_key == 'cash':
                    period_return += weight * 0.0
                    continue

                feature_name = self.FEATURE_NAME_MAP.get(stats_key)

                if feature_name is not None and feature_name in features:
                    # This asset class's return is part of the correlated draw --
                    # use it directly so cross-asset correlation is preserved.
                    idx = features.index(feature_name)
                    period_return += weight * samples[idx]
                else:
                    # Feature wasn't part of this regime's correlation matrix (either
                    # unsupported stats_key, or this asset class lacked enough
                    # observations in this regime). Sample independently, applying
                    # the equity-fallback logic in get_asset_class_params.
                    params = self.get_asset_class_params(asset_class, regime)
                    period_return += weight * rng.normal(params['mean'], params['std'])

            return float(period_return)
        except np.linalg.LinAlgError:
            # Fallback if covariance matrix is singular
            return self._sample_independent_portfolio(asset_class_weights, regime, rng)

    def _sample_independent_portfolio(self,
                                      asset_class_weights: Dict[str, float],
                                      regime: int,
                                      rng: np.random.Generator) -> float:
        """
        Fallback: sample asset returns independently (no correlation).
        """
        period_return = 0.0
        for asset_class, weight in asset_class_weights.items():
            params = self.get_asset_class_params(asset_class, regime)
            asset_return = rng.normal(params['mean'], params['std'])
            period_return += weight * asset_return
        
        return float(period_return)
    
    def run_simulation(self,
                      asset_class_weights: Dict[str, float],
                      initial_balance: float,
                      n_trials: int,
                      n_years: int,
                      annual_contribution: float = 0.0,
                      annual_withdrawal: float = 0.0,
                      annual_fee: float = 0.0,
                      contribution_years: int = 0,
                      failure_threshold: float = 0.0,
                      target_end_balance: float = 0.0,
                      random_state: int = 42) -> Dict:
        """
        Run Monte Carlo simulation for a multi-asset portfolio.
        
        Args:
            asset_class_weights: Dict of {asset_class: weight}
            initial_balance: Starting portfolio value
            n_trials: Number of Monte Carlo trials
            n_years: Simulation horizon in years
            annual_contribution: Annual contribution (added at start of year)
            annual_withdrawal: Annual withdrawal (taken at start of year)
            annual_fee: Annual portfolio fee rate
            contribution_years: Number of years to contribute
            failure_threshold: Portfolio value below which a trial fails
            target_end_balance: Required final balance for success
            random_state: Random seed
            
        Returns:
            Dict with simulation results
        """
        rng = np.random.default_rng(random_state)
        n_months = n_years * 12
        
        print(f"\n{'='*60}")
        print("REGIME-AWARE MONTE CARLO SIMULATION")
        print(f"{'='*60}")
        print(f"Initial balance: ${initial_balance:,.0f}")
        print(f"Trials: {n_trials:,}")
        print(f"Horizon: {n_years} years ({n_months} months)")
        print(f"Annual contribution: ${annual_contribution:,.0f}")
        print(f"Annual withdrawal: ${annual_withdrawal:,.0f}")
        print("\nAsset Allocation:")
        for ac, weight in sorted(asset_class_weights.items(), key=lambda x: -x[1]):
            print(f"  {ac}: {weight*100:.1f}%")
        
        # Storage
        all_balances = np.zeros((n_trials, n_months + 1))
        all_regimes = np.zeros((n_trials, n_months), dtype=int)
        final_balances = np.zeros(n_trials)
        
        # Get initial regime distribution
        initial_probs = self.assumptions.get_current_regime_probabilities()
        
        monthly_contribution = annual_contribution / 12
        monthly_withdrawal = annual_withdrawal / 12
        monthly_fee_rate = annual_fee / 12
        contribution_months = int(contribution_years) * 12
        all_success = np.ones(n_trials, dtype=bool)

        for trial in range(n_trials):
            if trial % 1000 == 0:
                print(f"  Trial {trial:,}/{n_trials:,}")
            
            trial_seed = random_state + trial
            
            # Simulate regime path
            regime_path = self.regime_sim.simulate_path(
                n_periods=n_months,
                initial_regime=None,
                initial_probs=initial_probs,
                random_state=trial_seed
            )
            all_regimes[trial] = regime_path
            
            # Sample returns
            returns = self.sample_returns(
                asset_class_weights,
                regime_path,
                random_state=trial_seed + 1_000_000
            )
            
            # Project balance
            balance = initial_balance
            all_balances[trial, 0] = balance
            
            trial_failed = False

            for month in range(n_months):
                if month < contribution_months:
                    balance += monthly_contribution
                
                balance -= monthly_withdrawal
                balance *= (1 + returns[month])
                balance *= (1.0 - monthly_fee_rate)

                # "Failure" is tracked as a flag once the balance first dips below
                # the threshold, but the balance itself is NOT floored at zero --
                # withdrawals and returns keep compounding through negative
                # territory (mirrors eMoney's convention) so percentile bands show
                # depth of shortfall rather than clamping everyone to $0.
                if balance < failure_threshold and not trial_failed:
                    trial_failed = True
                    all_success[trial] = False

                all_balances[trial, month + 1] = balance
            
            final_balances[trial] = balance
            if trial_failed or balance < target_end_balance:
                all_success[trial] = False
        
        # Compute statistics
        success_rate = float(np.mean(all_success))
        
        results = {
            'all_balances': all_balances,
            'all_regimes': all_regimes,
            'final_balances': final_balances,
            'success_rate': success_rate,
            'median_final': float(np.median(final_balances)),
            'mean_final': float(np.mean(final_balances)),
            'p10_final': float(np.percentile(final_balances, 10)),
            'p20_final': float(np.percentile(final_balances, 20)),
            'p25_final': float(np.percentile(final_balances, 25)),
            'p75_final': float(np.percentile(final_balances, 75)),
            'p80_final': float(np.percentile(final_balances, 80)),
            'p90_final': float(np.percentile(final_balances, 90)),
            'min_final': float(np.min(final_balances)),
            'max_final': float(np.max(final_balances)),
        }
        
        print(f"\n{'='*60}")
        print("SIMULATION RESULTS")
        print(f"{'='*60}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Median final balance: ${results['median_final']:,.0f}")
        print(f"20th percentile: ${results['p20_final']:,.0f}")
        print(f"80th percentile: ${results['p80_final']:,.0f}")
        print(f"10th percentile: ${results['p10_final']:,.0f}")
        print(f"90th percentile: ${results['p90_final']:,.0f}")
        
        return results


def main():
    """Example usage."""
    from portfolio_analyzer import PortfolioAnalyzer
    
    portfolio = {
        'SPY': 100,
        'AGG': 50,
        'VNQ': 25,
    }
    
    analyzer = PortfolioAnalyzer()
    summary = analyzer.analyze(portfolio)
    
    simulator = RegimeMultiAssetSimulator()
    results = simulator.run_simulation(
        asset_class_weights=summary['asset_class_weights'],
        initial_balance=summary['total_value'],
        n_trials=10_000,
        n_years=30,
        annual_contribution=0,
        annual_withdrawal=0,
        random_state=42
    )
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "simulation_results.json", 'w') as f:
        json_results = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in results.items()
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n[+] Results saved to output/simulation_results.json")


if __name__ == "__main__":
    main()