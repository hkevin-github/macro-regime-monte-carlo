"""
Monte Carlo Scenario Runner
Orchestrates the full regime-aware Monte Carlo simulation.
Integrates regime path generation, return sampling, and portfolio projection.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path
import json

from .regime_assumptions import RegimeAssumptions
from .regime_path_simulator import RegimePathSimulator
from .regime_return_sampler import RegimeReturnSampler, annual_to_monthly_rate
from .portfolio_engine import PortfolioEngine, PortfolioConfig


class MonteCarloRunner:
    """
    Orchestrates regime-aware Monte Carlo simulation.
    
    This is the main entry point that ties together:
    1. Regime path simulation (macro state sequences)
    2. Return sampling (regime-conditional market returns)
    3. Portfolio projection (cash flows and balance evolution)
    """
    
    def __init__(self,
                 portfolio_config: PortfolioConfig,
                 models_dir: str = "data/models",
                 sampling_method: str = 'gaussian'):
        """
        Initialize Monte Carlo runner.
        
        Args:
            portfolio_config: portfolio simulation parameters
            models_dir: directory containing trained HMM artifacts
            sampling_method: 'gaussian' or 'bootstrap' for return sampling
        """
        # Load regime assumptions
        self.assumptions = RegimeAssumptions(models_dir)
        
        # Initialize components
        self.regime_simulator = RegimePathSimulator(
            self.assumptions.transition_matrix,
            self.assumptions.regime_labels['regime_names']
        )
        
        self.return_sampler = RegimeReturnSampler(
            self.assumptions.regime_stats,
            sampling_method=sampling_method
        )
        
        self.portfolio_engine = PortfolioEngine(portfolio_config)
        
        print(f"[+] Monte Carlo runner initialized with {self.assumptions.n_regimes} regimes")
        print(f"[+] Sampling method: {sampling_method}")
    
    def run_simulation(self,
                      n_trials: int,
                      n_years: int,
                      initial_regime: Optional[int] = None,
                      use_current_regime_probs: bool = True,
                      random_state: Optional[int] = None) -> Dict:
        """
        Run full Monte Carlo simulation.
        
        Args:
            n_trials: number of scenarios to simulate
            n_years: planning horizon in years
            initial_regime: starting regime (if None, uses initial_probs)
            use_current_regime_probs: if True, initializes from latest historical posteriors
            random_state: random seed for reproducibility
            
        Returns:
            Dictionary containing all simulation results
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_periods = n_years * 12  # Convert to months
        
        print(f"\n{'='*70}")
        print(f"RUNNING MONTE CARLO SIMULATION")
        print(f"{'='*70}")
        print(f"Trials: {n_trials:,}")
        print(f"Horizon: {n_years} years ({n_periods} months)")
        print(f"Initial balance: ${self.portfolio_engine.config.initial_balance:,.0f}")
        print(f"Annual withdrawal: ${self.portfolio_engine.config.annual_withdrawal:,.0f}")
        
        # Determine initial regime distribution
        if initial_regime is not None:
            initial_probs = None
            print(f"Starting regime: {self.assumptions.regime_labels['regime_names'][initial_regime]}")
        elif use_current_regime_probs:
            initial_probs = self.assumptions.get_current_regime_probabilities()
            initial_regime = None
            print(f"Starting regime distribution: current posteriors")
        else:
            initial_probs = self.assumptions.get_stationary_distribution()
            initial_regime = None
            print(f"Starting regime distribution: stationary")
        
        print(f"\n[*] Simulating {n_trials:,} regime paths...")
        
        # Step 1: Simulate regime paths
        regime_paths = self.regime_simulator.simulate_multiple_paths(
            n_paths=n_trials,
            n_periods=n_periods,
            initial_regime=initial_regime,
            initial_probs=initial_probs,
            random_state=random_state
        )
        
        print(f"[*] Sampling returns conditional on regimes...")
        
        # Step 2: Sample returns conditional on regime paths
        returns_array, inflation_array = self.return_sampler.sample_multiple_paths(
            regime_paths,
            random_state=random_state
        )
        
        print(f"[*] Projecting portfolio outcomes...")
        
        # Step 3: Project portfolio for each scenario
        results = self.portfolio_engine.project_multiple_scenarios(
            returns_array,
            inflation_array,
            regime_paths
        )
        
        print(f"[*] Computing statistics...")
        
        # Step 4: Compute aggregate statistics
        success_rate = self.portfolio_engine.compute_success_rate(results)
        percentiles = self.portfolio_engine.compute_percentile_outcomes(results)
        failure_stats = self.portfolio_engine.compute_failure_statistics(results)
        regime_occupancy = self.regime_simulator.get_regime_occupancy(regime_paths)
        
        # Compute regime-specific outcomes
        regime_attribution = self._compute_regime_attribution(results)
        
        print(f"\n{'='*70}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Median final balance: ${percentiles[50]:,.0f}")
        print(f"5th percentile: ${percentiles[2.5]:,.0f}")
        print(f"95th percentile: ${percentiles[97.5]:,.0f}")
        
        if failure_stats['failure_rate'] > 0:
            print(f"\nFailure rate: {failure_stats['failure_rate']:.1%}")
            print(f"Median failure month: {failure_stats['median_failure_month']:.0f}")
        
        return {
            'n_trials': n_trials,
            'n_periods': n_periods,
            'success_rate': success_rate,
            'percentiles': percentiles,
            'failure_stats': failure_stats,
            'regime_occupancy': regime_occupancy,
            'regime_attribution': regime_attribution,
            'results': results,  # Individual trial results
            'regime_paths': regime_paths,
            'returns_array': returns_array,
            'inflation_array': inflation_array,
        }
    
    def _compute_regime_attribution(self, results: List[Dict]) -> Dict:
        """
        Analyze which regimes contributed to failures.
        
        Returns:
            Dictionary with regime-specific failure statistics
        """
        failures = [r for r in results if r['failed']]
        
        if not failures:
            return {}
        
        attribution = {}
        
        for regime in range(self.assumptions.n_regimes):
            regime_name = self.assumptions.regime_labels['regime_names'][regime]
            
            # Count failures where this regime was active at failure
            failures_in_regime = sum(
                1 for r in failures 
                if r['regime_path'] is not None 
                and r['failure_month'] is not None
                and r['regime_path'][int(r['failure_month'])] == regime
            )
            
            # Compute average time spent in regime before failure
            time_in_regime = []
            for r in failures:
                if r['regime_path'] is not None and r['failure_month'] is not None:
                    path_to_failure = r['regime_path'][:int(r['failure_month']) + 1]
                    time_in_regime.append((path_to_failure == regime).sum())
            
            attribution[regime_name] = {
                'failures_in_regime': failures_in_regime,
                'failure_rate_in_regime': failures_in_regime / len(failures) if failures else 0,
                'avg_time_in_regime_before_failure': np.mean(time_in_regime) if time_in_regime else 0,
            }
        
        return attribution
    
    def run_baseline_comparison(self,
                               n_trials: int,
                               n_years: int,
                               random_state: Optional[int] = None) -> Dict:
        """
        Run a traditional eMoney-style Monte Carlo (no regime switching).
        Uses average return and volatility across all regimes.
        
        This provides a baseline to compare against regime-aware simulation.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_periods = n_years * 12
        
        print(f"\n{'='*70}")
        print(f"RUNNING BASELINE (NON-REGIME) SIMULATION")
        print(f"{'='*70}")
        
        # Compute average return and volatility across all historical data
        df = pd.read_csv("data/processed/regime_labeled_dataset.csv", 
                        index_col='date', parse_dates=True)
        
        avg_return = df['equity_return'].mean()
        avg_std = df['equity_return'].std()
        avg_inflation = annual_to_monthly_rate(df['cpi_yoy'].mean()) if 'cpi_yoy' in df else 0.02 / 12
        inflation_std = annual_to_monthly_rate(df['cpi_yoy'].std()) if 'cpi_yoy' in df else 0.01 / 12
        
        print(f"Using historical average return: {avg_return:.4f} per month")
        print(f"Using historical volatility: {avg_std:.4f} per month")
        
        # Sample returns from single distribution (no regime awareness)
        returns_array = np.random.normal(avg_return, avg_std, (n_trials, n_periods))
        inflation_array = np.random.normal(avg_inflation, inflation_std, (n_trials, n_periods))
        
        # Project portfolio
        results = self.portfolio_engine.project_multiple_scenarios(
            returns_array,
            inflation_array,
            regime_paths=None
        )
        
        success_rate = self.portfolio_engine.compute_success_rate(results)
        percentiles = self.portfolio_engine.compute_percentile_outcomes(results)
        failure_stats = self.portfolio_engine.compute_failure_statistics(results)
        
        print(f"\nBaseline success rate: {success_rate:.1%}")
        print(f"Median final balance: ${percentiles[50]:,.0f}")
        
        return {
            'n_trials': n_trials,
            'n_periods': n_periods,
            'success_rate': success_rate,
            'percentiles': percentiles,
            'failure_stats': failure_stats,
            'results': results,
            'avg_return': avg_return,
            'avg_std': avg_std,
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save simulation results to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save summary statistics (JSON-serializable only)
        summary = {
            'n_trials': results['n_trials'],
            'n_periods': results['n_periods'],
            'success_rate': results['success_rate'],
            'percentiles': results['percentiles'],
            'failure_stats': results['failure_stats'],
            'regime_occupancy': results['regime_occupancy'],
            'regime_attribution': results['regime_attribution'],
        }
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[+] Results saved to: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    # Example usage
    config = PortfolioConfig(
        initial_balance=1_000_000,
        annual_withdrawal=40_000,
        withdrawal_inflation_adjust=True,
        annual_contribution=0,
        contribution_years=0,
        failure_threshold=0,
        target_end_balance=0
    )
    
    runner = MonteCarloRunner(
        portfolio_config=config,
        models_dir="data/models",
        sampling_method='gaussian'
    )
    
    # Run regime-aware simulation
    regime_results = runner.run_simulation(
        n_trials=5000,
        n_years=30,
        use_current_regime_probs=True,
        random_state=1000000
    )
    
    # Run baseline comparison
    baseline_results = runner.run_baseline_comparison(
        n_trials=5000,
        n_years=30,
        random_state=1000000
    )
    
    # Compare
    print(f"\n{'='*70}")
    print(f"REGIME-AWARE vs BASELINE COMPARISON")
    print(f"{'='*70}")
    print(f"Regime-aware success rate: {regime_results['success_rate']:.1%}")
    print(f"Baseline success rate: {baseline_results['success_rate']:.1%}")
    print(f"Difference: {(regime_results['success_rate'] - baseline_results['success_rate']):.1%}")