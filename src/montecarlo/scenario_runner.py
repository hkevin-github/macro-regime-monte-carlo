"""
Monte Carlo Scenario Runner
Orchestrates the full regime-aware Monte Carlo simulation.
"""

import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path

from .regime_assumptions import RegimeAssumptions
from .regime_path_simulator import RegimePathSimulator
from .regime_return_sampler import RegimeReturnSampler
from .portfolio_engine import PortfolioEngine, PortfolioConfig
from .utils import yoy_to_monthly


class MonteCarloRunner:
    def __init__(self,
                 portfolio_config: PortfolioConfig,
                 models_dir: str = "data/models",
                 sampling_method: str = 'gaussian'):
        self.assumptions = RegimeAssumptions(models_dir)

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
        rng = np.random.default_rng(random_state)
        n_periods = n_years * 12

        print(f"\n{'=' * 70}")
        print("RUNNING MONTE CARLO SIMULATION")
        print(f"{'=' * 70}")
        print(f"Trials: {n_trials:,}")
        print(f"Horizon: {n_years} years ({n_periods} months)")
        print(f"Initial balance: ${self.portfolio_engine.config.initial_balance:,.0f}")
        print(f"Annual withdrawal: ${self.portfolio_engine.config.annual_withdrawal:,.0f}")

        if initial_regime is not None:
            initial_probs = None
            print(f"Starting regime: {self.assumptions.regime_labels['regime_names'][initial_regime]}")
        elif use_current_regime_probs:
            initial_probs = self.assumptions.get_current_regime_probabilities()
            initial_regime = None
            print("Starting regime distribution: current posteriors")
        else:
            initial_probs = self.assumptions.get_stationary_distribution()
            initial_regime = None
            print("Starting regime distribution: stationary")

        print(f"\n[*] Simulating {n_trials:,} regime paths...")
        regime_paths = self.regime_simulator.simulate_multiple_paths(
            n_paths=n_trials,
            n_periods=n_periods,
            initial_regime=initial_regime,
            initial_probs=initial_probs,
            random_state=random_state
        )

        print("[*] Sampling returns conditional on regimes...")
        returns_array, inflation_array = self.return_sampler.sample_multiple_paths(
            regime_paths,
            random_state=random_state
        )

        print("[*] Projecting portfolio outcomes...")
        results = self.portfolio_engine.project_multiple_scenarios(
            returns_array,
            inflation_array,
            regime_paths
        )

        print("[*] Computing statistics...")
        success_rate = self.portfolio_engine.compute_success_rate(results)
        percentiles = self.portfolio_engine.compute_percentile_outcomes(results)
        failure_stats = self.portfolio_engine.compute_failure_statistics(results)
        regime_occupancy = self.regime_simulator.get_regime_occupancy(regime_paths)
        regime_attribution = self._compute_regime_attribution(results)

        print(f"\n{'=' * 70}")
        print("SIMULATION COMPLETE")
        print(f"{'=' * 70}")
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
            'results': results,
            'regime_paths': regime_paths,
            'returns_array': returns_array,
            'inflation_array': inflation_array,
        }

    def _compute_regime_attribution(self, results: List[Dict]) -> Dict:
        failures = [r for r in results if r['failed']]
        if not failures:
            return {}

        attribution = {}

        for regime in range(self.assumptions.n_regimes):
            regime_name = self.assumptions.regime_labels['regime_names'][regime]

            failures_in_regime = sum(
                1 for r in failures
                if r['regime_path'] is not None
                and r['failure_month'] is not None
                and r['regime_path'][int(r['failure_month'])] == regime
            )

            time_in_regime = []
            for r in failures:
                if r['regime_path'] is not None and r['failure_month'] is not None:
                    path_to_failure = r['regime_path'][:int(r['failure_month']) + 1]
                    time_in_regime.append((path_to_failure == regime).sum())

            attribution[regime_name] = {
                'failures_in_regime': failures_in_regime,
                'failure_rate_in_regime': failures_in_regime / len(failures) if failures else 0,
                'avg_time_in_regime_before_failure': float(np.mean(time_in_regime)) if time_in_regime else 0,
            }

        return attribution

    def run_baseline_comparison(self,
                                n_trials: int,
                                n_years: int,
                                random_state: Optional[int] = None) -> Dict:
        rng = np.random.default_rng(random_state)
        n_periods = n_years * 12

        print(f"\n{'=' * 70}")
        print("RUNNING BASELINE (NON-REGIME) SIMULATION")
        print(f"{'=' * 70}")

        df = pd.read_csv("data/processed/regime_labeled_dataset.csv", index_col='date', parse_dates=True)

        avg_return = df['equity_return'].mean()
        avg_std = df['equity_return'].std()

        if 'cpi_yoy' in df.columns:
            avg_inflation_yoy = df['cpi_yoy'].mean()
            inflation_std_yoy = df['cpi_yoy'].std()
        else:
            avg_inflation_yoy = 0.02
            inflation_std_yoy = 0.01

        print(f"Using historical average return: {avg_return:.4f} per month")
        print(f"Using historical volatility: {avg_std:.4f} per month")
        print(f"Using historical CPI YoY mean: {avg_inflation_yoy:.4f}")

        returns_array = rng.normal(avg_return, avg_std, (n_trials, n_periods))

        inflation_yoy_array = rng.normal(avg_inflation_yoy, inflation_std_yoy, (n_trials, n_periods))
        inflation_yoy_array = np.maximum(inflation_yoy_array, -0.99)
        inflation_array = (1.0 + inflation_yoy_array) ** (1.0 / 12.0) - 1.0

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
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

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