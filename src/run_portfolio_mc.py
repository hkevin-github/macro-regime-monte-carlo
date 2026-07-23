# src/run_regime_portfolio_mc.py
"""
Regime-aware Monte Carlo portfolio simulator.
Accepts portfolio configuration via JSON file.
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

from montecarlo.regime_assumptions import RegimeAssumptions
from montecarlo.regime_path_simulator import RegimePathSimulator
from montecarlo.regime_return_sampler import RegimeReturnSampler
from montecarlo.portfolio_engine import PortfolioEngine, PortfolioConfig


@dataclass
class RegimePortfolioConfig:
    """Configuration for regime-aware portfolio Monte Carlo simulation."""
    initial_balance: float
    annual_withdrawal: float = 0.0
    withdrawal_inflation_adjust: bool = False
    annual_contribution: float = 0.0
    contribution_years: int = 0
    target_end_balance: float = 0.0
    failure_threshold: float = 0.0
    portfolio_weights: Dict[str, float] = field(default_factory=dict)
    annual_fee: float = 0.0
    n_trials: int = 10_000
    n_years: int = 30
    use_current_regime_probs: bool = True
    initial_regime: Optional[int] = None
    random_state: int = 42


def load_config(config_path: str) -> RegimePortfolioConfig:
    """
    Load portfolio configuration from JSON file with validation.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        RegimePortfolioConfig object
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if path.suffix != '.json':
        raise ValueError(f"Only JSON format supported. Got: {path.suffix}")
    
    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    config = RegimePortfolioConfig(**config_dict)
    
    # Validation
    print("\n[*] Validating configuration...")
    
    # Validate portfolio weights sum to 1.0
    total_weight = sum(config.portfolio_weights.values())
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        raise ValueError(
            f"Portfolio weights must sum to 1.0. Got {total_weight:.6f}.\n"
            f"Weights: {config.portfolio_weights}"
        )
    
    # Validate individual weights
    for asset, weight in config.portfolio_weights.items():
        if weight < 0 or weight > 1:
            raise ValueError(f"Weight for '{asset}' must be in [0, 1]. Got {weight}")
    
    # Validate other parameters
    if config.initial_balance <= 0:
        raise ValueError(f"Initial balance must be positive. Got {config.initial_balance}")
    
    if config.annual_withdrawal < 0:
        raise ValueError(f"Annual withdrawal cannot be negative. Got {config.annual_withdrawal}")
    
    if config.annual_fee < 0 or config.annual_fee > 0.1:
        raise ValueError(f"Annual fee must be in [0, 0.1]. Got {config.annual_fee}")
    
    if config.n_years <= 0 or config.n_years > 100:
        raise ValueError(f"n_years must be in (0, 100]. Got {config.n_years}")
    
    if config.n_trials < 1:
        raise ValueError(f"n_trials must be positive. Got {config.n_trials}")
    
    if config.n_trials < 1000:
        print(f"⚠️  WARNING: Only {config.n_trials} trials. Recommend >= 10,000.")
    
    print("[✓] Configuration validated")
    
    return config


class RegimeAwarePortfolioSimulator:
    """
    Regime-aware Monte Carlo portfolio simulator.
    Samples returns conditional on HMM regime states.
    """
    
    def __init__(self, config: RegimePortfolioConfig):
        self.config = config
        
        # Load regime assumptions and market statistics
        self.assumptions = RegimeAssumptions(models_dir="data/models")
        self.regime_labels = self.assumptions.regime_labels['regime_names']
        self.regime_stats = self.assumptions.regime_stats
        self.transition_matrix = self.assumptions.transition_matrix

        # Initialize simulators
        self.regime_sim = RegimePathSimulator(
            transition_matrix=self.transition_matrix,
            regime_names=self.regime_labels
        )

        self.return_sampler = RegimeReturnSampler(
            regime_stats=self.regime_stats,
            sampling_method='gaussian'
        )
        
        portfolio_config = PortfolioConfig(
            initial_balance=config.initial_balance,
            annual_withdrawal=config.annual_withdrawal,
            withdrawal_inflation_adjust=config.withdrawal_inflation_adjust,
            annual_contribution=config.annual_contribution,
            contribution_years=config.contribution_years,
            failure_threshold=config.failure_threshold,
            target_end_balance=config.target_end_balance,
            annual_fee=config.annual_fee
        )

        self.portfolio_engine = PortfolioEngine(portfolio_config)
    
    def run_simulation(self) -> Dict:
        """Run Monte Carlo simulation with regime-aware returns."""
        # Create master RNG from config seed
        master_rng = np.random.default_rng(self.config.random_state)
        
        n_trials = self.config.n_trials
        n_months = self.config.n_years * 12
        
        # Storage for results
        all_balances = np.zeros((n_trials, n_months + 1))
        all_regimes = np.zeros((n_trials, n_months), dtype=int)
        final_balances = np.zeros(n_trials)
        
        equity_weight = self.config.portfolio_weights.get('equity', 0.0)
        bond_weight = self.config.portfolio_weights.get('bonds', 0.0)
        
        print(f"Running {n_trials:,} trials for {self.config.n_years} years...")
        print(f"Portfolio: {equity_weight:.0%} equity / {bond_weight:.0%} bonds")
        
        for trial in range(n_trials):
            if trial % 1000 == 0:
                print(f"  Trial {trial:,}/{n_trials:,}")
            
            # Generate deterministic seed for this trial
            trial_seed = self.config.random_state + trial
            
            # Decide starting regime distribution
            if self.config.initial_regime is not None:
                initial_regime = self.config.initial_regime
                initial_probs = None
            elif self.config.use_current_regime_probs:
                initial_regime = None
                initial_probs = self.assumptions.get_current_regime_probabilities()
            else:
                initial_regime = None
                initial_probs = self.assumptions.get_stationary_distribution()
            
            # Simulate regime path with deterministic seed
            regime_path = self.regime_sim.simulate_path(
                n_periods=n_months,
                initial_regime=initial_regime,
                initial_probs=initial_probs,
                random_state=trial_seed
            )
            all_regimes[trial] = regime_path
            
            # Sample equity returns and inflation with deterministic seed
            equity_returns, monthly_inflation = self.return_sampler.sample_returns(
                regime_path,
                random_state=trial_seed + 1_000_000
            )
            
            # Create RNG for bond returns
            bond_rng = np.random.default_rng(trial_seed + 2_000_000)
            
            # Combine equity and bond returns
            monthly_returns = np.zeros(n_months)
            for month in range(n_months):
                regime = regime_path[month]
                equity_return = equity_returns[month]
                bond_return = self._sample_bond_return(regime, bond_rng)
                
                monthly_returns[month] = (
                    equity_weight * equity_return +
                    bond_weight * bond_return
                )
            
            # Project portfolio
            result = self.portfolio_engine.project_portfolio(
                returns=monthly_returns,
                inflation=monthly_inflation,
                regime_path=regime_path
            )
            
            all_balances[trial] = result['balance_history']
            final_balances[trial] = result['final_balance']
        
        # Calculate metrics
        success_rate = np.mean(final_balances > self.config.failure_threshold)
        
        results = {
            'all_balances': all_balances,
            'all_regimes': all_regimes,
            'final_balances': final_balances,
            'success_rate': success_rate,
            'median_final': np.median(final_balances),
            'p20_final': np.percentile(final_balances, 20),
            'p80_final': np.percentile(final_balances, 80)
        }
        
        return results
    
    def _sample_bond_return(self, regime: int, rng: np.random.Generator) -> float:
        """
        Sample monthly bond return based on regime characteristics.
        
        Args:
            regime: Regime ID
            rng: Random number generator for reproducibility
            
        Returns:
            Monthly bond return
        """
        regime_label = self.regime_labels[regime]
        
        if regime_label == 'Growth':
            return rng.normal(0.003, 0.01)
        elif regime_label in ['Shock', 'Slowdown']:
            return rng.normal(0.005, 0.008)
        elif regime_label == 'Inflationary':
            return rng.normal(-0.001, 0.015)
        else:  # Balanced or unknown
            return rng.normal(0.004, 0.012)


def plot_results(results: Dict, config: RegimePortfolioConfig, output_path: str):
    """Create visualization of simulation results."""
    all_balances = results['all_balances']
    n_years = config.n_years
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Panel 1: Portfolio trajectories
    ax1 = axes[0]
    years = np.linspace(0, n_years, all_balances.shape[1])
    
    # Plot percentiles (20th and 80th)
    p20 = np.percentile(all_balances, 20, axis=0)
    p50 = np.percentile(all_balances, 50, axis=0)
    p80 = np.percentile(all_balances, 80, axis=0)
    
    ax1.fill_between(years, p20, p80, alpha=0.3, label='20th-80th percentile')
    ax1.plot(years, p50, 'b-', linewidth=2, label='Median')
    ax1.axhline(y=config.failure_threshold, color='r', linestyle='--', 
                label=f'Failure threshold: ${config.failure_threshold:,.0f}')
    
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Portfolio Balance ($)')
    ax1.set_title('Portfolio Balance Trajectories (Regime-Aware Monte Carlo)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Final balance distribution
    ax2 = axes[1]
    final_balances = results['final_balances']
    
    ax2.hist(final_balances, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.median(final_balances), color='b', linestyle='-', 
                linewidth=2, label=f'Median: ${np.median(final_balances):,.0f}')
    ax2.axvline(x=config.failure_threshold, color='r', linestyle='--',
                linewidth=2, label=f'Threshold: ${config.failure_threshold:,.0f}')
    
    ax2.set_xlabel('Final Balance ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Final Balance Distribution (Success Rate: {results["success_rate"]:.1%})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python src/run_regime_portfolio_mc.py <config.json>")
        print("\nExample:")
        print("  python src/run_regime_portfolio_mc.py config/80_20_portfolio.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Initialize simulator
    simulator = RegimeAwarePortfolioSimulator(config)
    
    # Run simulation
    results = simulator.run_simulation()
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Median final balance: ${results['median_final']:,.0f}")
    print(f"20th percentile: ${results['p20_final']:,.0f}")
    print(f"80th percentile: ${results['p80_final']:,.0f}")
    
    # Generate output filename from weights
    equity_pct = int(config.portfolio_weights.get('equity', 0) * 100)
    bond_pct = int(config.portfolio_weights.get('bonds', 0) * 100)
    output_name = f"portfolio_{equity_pct}equ_{bond_pct}bon.png"
    output_path = Path("output") / output_name
    output_path.parent.mkdir(exist_ok=True)
    
    # Plot results
    plot_results(results, config, str(output_path))
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()