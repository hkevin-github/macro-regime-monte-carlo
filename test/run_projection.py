"""
Portfolio Projection Runner
Executes regime-aware and baseline Monte Carlo simulations,
then generates reports and visualizations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.montecarlo.scenario_runner import MonteCarloRunner
from scenario_config import CLIENT_SCENARIO, N_TRIALS, N_YEARS, RANDOM_SEED


def check_prerequisites():
    """Ensure required data files exist before running."""
    required_files = [
        "data/processed/aligned_macro_dataset.csv",
        "data/processed/regime_labeled_dataset.csv",
        "data/models/regime_market_assumptions.json",
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("❌ ERROR: Missing required files:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlease run the data pipeline first:")
        print("   1. python src/data_pipeline.py")
        print("   2. python src/model_selection.py")
        print("   3. python src/hmm_regime_engine.py")
        sys.exit(1)
    
    print("✓ All prerequisite files found\n")


def print_scenario_summary():
    """Display the scenario being simulated."""
    print("="*70)
    print("SCENARIO CONFIGURATION")
    print("="*70)
    print(f"Initial Balance:      ${CLIENT_SCENARIO.initial_balance:,.0f}")
    print(f"Annual Withdrawal:    ${CLIENT_SCENARIO.annual_withdrawal:,.0f}")
    print(f"Inflation Adjusted:   {CLIENT_SCENARIO.withdrawal_inflation_adjust}")
    print(f"Annual Contribution:  ${CLIENT_SCENARIO.annual_contribution:,.0f}")
    print(f"Contribution Years:   {CLIENT_SCENARIO.contribution_years}")
    print(f"Planning Horizon:     {N_YEARS} years")
    print(f"Monte Carlo Trials:   {N_TRIALS:,}")
    print(f"Target End Balance:   ${CLIENT_SCENARIO.target_end_balance:,.0f}" if CLIENT_SCENARIO.target_end_balance > 0 else "Target End Balance:   None")
    print("="*70 + "\n")


def run_simulations():
    """Execute both regime-aware and baseline simulations."""
    
    # Initialize runner
    runner = MonteCarloRunner(
        portfolio_config=CLIENT_SCENARIO,
        models_dir="data/models",
        sampling_method='gaussian'
    )
    
    # Run regime-aware simulation
    print("\n" + "="*70)
    print("RUNNING REGIME-AWARE MONTE CARLO")
    print("="*70)
    regime_results = runner.run_simulation(
        n_trials=N_TRIALS,
        n_years=N_YEARS,
        use_current_regime_probs=True,
        random_state=RANDOM_SEED
    )
    
    # Run baseline simulation
    print("\n" + "="*70)
    print("RUNNING BASELINE MONTE CARLO (Traditional)")
    print("="*70)
    baseline_results = runner.run_baseline_comparison(
        n_trials=N_TRIALS,
        n_years=N_YEARS,
        random_state=RANDOM_SEED
    )
    
    return runner, regime_results, baseline_results


def generate_comparison_report(regime_results, baseline_results):
    """Generate detailed comparison between regime-aware and baseline."""
    
    print("\n" + "="*70)
    print("COMPARATIVE RESULTS SUMMARY")
    print("="*70)
    
    # Success rates
    print(f"\nSUCCESS RATES:")
    print(f"  Regime-Aware:  {regime_results['success_rate']:>6.1%}")
    print(f"  Baseline:      {baseline_results['success_rate']:>6.1%}")
    diff = regime_results['success_rate'] - baseline_results['success_rate']
    arrow = "↑" if diff > 0 else "↓"
    print(f"  Difference:    {arrow} {abs(diff):>5.1%}")
    
    # Final balance percentiles
    print(f"\nFINAL BALANCE PERCENTILES:")
    print(f"                  Regime-Aware         Baseline          Difference")
    print(f"  5th %ile:    ${regime_results['percentiles'][2.5]:>12,.0f}   "
          f"${baseline_results['percentiles'][2.5]:>12,.0f}   "
          f"${regime_results['percentiles'][2.5] - baseline_results['percentiles'][2.5]:>12,.0f}")
    print(f"  50th %ile:   ${regime_results['percentiles'][50]:>12,.0f}   "
          f"${baseline_results['percentiles'][50]:>12,.0f}   "
          f"${regime_results['percentiles'][50] - baseline_results['percentiles'][50]:>12,.0f}")
    print(f"  95th %ile:   ${regime_results['percentiles'][97.5]:>12,.0f}   "
          f"${baseline_results['percentiles'][97.5]:>12,.0f}   "
          f"${regime_results['percentiles'][97.5] - baseline_results['percentiles'][97.5]:>12,.0f}")
    
    # Failure analysis
    if regime_results['failure_stats']['failure_rate'] > 0:
        print(f"\nFAILURE ANALYSIS (Regime-Aware):")
        print(f"  Failure Rate:        {regime_results['failure_stats']['failure_rate']:.1%}")
        print(f"  Median Failure Time: Year {regime_results['failure_stats']['median_failure_month']/12:.1f}")
        print(f"  Mean Failure Time:   Year {regime_results['failure_stats']['mean_failure_month']/12:.1f}")
    
    if baseline_results['failure_stats']['failure_rate'] > 0:
        print(f"\nFAILURE ANALYSIS (Baseline):")
        print(f"  Failure Rate:        {baseline_results['failure_stats']['failure_rate']:.1%}")
        print(f"  Median Failure Time: Year {baseline_results['failure_stats']['median_failure_month']/12:.1f}")
        print(f"  Mean Failure Time:   Year {baseline_results['failure_stats']['mean_failure_month']/12:.1f}")
    
    # Regime attribution (regime-aware only)
    if regime_results.get('regime_attribution'):
        print(f"\nREGIME FAILURE ATTRIBUTION:")
        print(f"  Which regimes contributed to failures?")
        for regime_name, stats in regime_results['regime_attribution'].items():
            if stats['failures_in_regime'] > 0:
                print(f"\n  {regime_name}:")
                print(f"    Failures in this regime: {stats['failures_in_regime']}")
                print(f"    % of all failures:       {stats['failure_rate_in_regime']:.1%}")
                print(f"    Avg time in regime:      {stats['avg_time_in_regime_before_failure']:.1f} months")
    
    # Regime occupancy
    if regime_results.get('regime_occupancy'):
        print(f"\nREGIME OCCUPANCY (Expected time in each regime):")
        for regime_name, stats in regime_results['regime_occupancy'].items():
            print(f"\n  {regime_name}:")
            print(f"    Frequency:      {stats['frequency']:.1%}")
            print(f"    Avg Duration:   {stats['mean_duration']:.1f} months")
            print(f"    Max Duration:   {stats['max_duration']:.0f} months")
    
    print("\n" + "="*70)


def create_visualizations(regime_results, baseline_results):
    """Generate charts comparing regime-aware vs baseline results."""
    
    os.makedirs("test/output", exist_ok=True)
    
    # Extract final balances
    regime_finals = np.array([r['final_balance'] for r in regime_results['results']])
    baseline_finals = np.array([r['final_balance'] for r in baseline_results['results']])
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))
    
    # -------------------------------------------------------------------------
    # Plot 1: Distribution of Final Balances
    # -------------------------------------------------------------------------
    ax1 = plt.subplot(1, 3, 1)
    
    ax1.hist(regime_finals, bins=60, alpha=0.6, color='blue', 
             edgecolor='black', label='Regime-Aware')
    ax1.hist(baseline_finals, bins=60, alpha=0.6, color='green', 
             edgecolor='black', label='Baseline')
    
    ax1.axvline(np.median(regime_finals), color='blue', linestyle='--', linewidth=2,
                label=f'Regime Median: ${np.median(regime_finals):,.0f}')
    ax1.axvline(np.median(baseline_finals), color='green', linestyle='--', linewidth=2,
                label=f'Baseline Median: ${np.median(baseline_finals):,.0f}')
    
    ax1.set_xlabel('Final Balance ($)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'Distribution of Outcomes ({N_YEARS} Years)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Percentile Comparison
    # -------------------------------------------------------------------------
    ax2 = plt.subplot(1, 3, 2)
    
    percentiles_to_plot = [5, 25, 50, 75, 95]
    regime_pcts = [np.percentile(regime_finals, p) for p in percentiles_to_plot]
    baseline_pcts = [np.percentile(baseline_finals, p) for p in percentiles_to_plot]
    
    x = np.arange(len(percentiles_to_plot))
    width = 0.35
    
    ax2.bar(x - width/2, regime_pcts, width, label='Regime-Aware', color='blue', alpha=0.7)
    ax2.bar(x + width/2, baseline_pcts, width, label='Baseline', color='green', alpha=0.7)
    
    ax2.set_xlabel('Percentile', fontsize=11)
    ax2.set_ylabel('Final Balance ($)', fontsize=11)
    ax2.set_title('Percentile Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{p}th' for p in percentiles_to_plot])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis='y')
    
    # Format y-axis as currency
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
    
    # -------------------------------------------------------------------------
    # Plot 3: Balance Evolution Over Time (sample paths)
    # -------------------------------------------------------------------------
    ax3 = plt.subplot(1, 3, 3)
    
    # Plot 100 random paths from regime-aware simulation
    n_paths_to_plot = 100
    sample_indices = np.random.choice(len(regime_results['results']), n_paths_to_plot, replace=False)
    
    for idx in sample_indices:
        balance_history = regime_results['results'][idx]['balance_history']
        years = np.arange(len(balance_history)) / 12
        ax3.plot(years, balance_history, color='blue', alpha=0.05, linewidth=0.5)
    
    # Compute and plot median path
    all_balances = np.array([r['balance_history'] for r in regime_results['results']])
    median_balance = np.median(all_balances, axis=0)
    p5_balance = np.percentile(all_balances, 5, axis=0)
    p95_balance = np.percentile(all_balances, 95, axis=0)
    
    years = np.arange(len(median_balance)) / 12
    ax3.plot(years, median_balance, color='darkblue', linewidth=2.5, label='Median')
    ax3.fill_between(years, p5_balance, p95_balance, color='blue', alpha=0.2, label='5th-95th %ile')
    
    ax3.axhline(CLIENT_SCENARIO.initial_balance, color='black', linestyle='--', 
                alpha=0.5, label='Initial Balance')
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5, label='Depletion')
    
    ax3.set_xlabel('Years', fontsize=11)
    ax3.set_ylabel('Portfolio Balance ($)', fontsize=11)
    ax3.set_title('Balance Evolution (Regime-Aware)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
    
    plt.tight_layout()
    
    output_path = "test/output/projection_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Visualization saved to: {output_path}")
    plt.close()


def save_detailed_results(regime_results, baseline_results):
    """Save detailed CSV outputs for further analysis."""
    
    os.makedirs("test/output", exist_ok=True)
    
    # Create summary DataFrame
    summary_data = {
        'Metric': [
            'Success Rate',
            '5th Percentile Final Balance',
            '50th Percentile Final Balance',
            '95th Percentile Final Balance',
            'Failure Rate',
            'Median Failure Month',
        ],
        'Regime-Aware': [
            f"{regime_results['success_rate']:.2%}",
            f"${regime_results['percentiles'][2.5]:,.0f}",
            f"${regime_results['percentiles'][50]:,.0f}",
            f"${regime_results['percentiles'][97.5]:,.0f}",
            f"{regime_results['failure_stats']['failure_rate']:.2%}",
            f"{regime_results['failure_stats'].get('median_failure_month', 'N/A')}",
        ],
        'Baseline': [
            f"{baseline_results['success_rate']:.2%}",
            f"${baseline_results['percentiles'][2.5]:,.0f}",
            f"${baseline_results['percentiles'][50]:,.0f}",
            f"${baseline_results['percentiles'][97.5]:,.0f}",
            f"{baseline_results['failure_stats']['failure_rate']:.2%}",
            f"{baseline_results['failure_stats'].get('median_failure_month', 'N/A')}",
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("test/output/summary_comparison.csv", index=False)
    print(f"[+] Summary saved to: test/output/summary_comparison.csv")
    
    # Save trial-by-trial results
    regime_trial_data = pd.DataFrame([
        {
            'trial': i,
            'final_balance': r['final_balance'],
            'success': r['success'],
            'failure_month': r['failure_month'] if r['failed'] else None,
            'min_balance': r['min_balance'],
            'max_balance': r['max_balance'],
        }
        for i, r in enumerate(regime_results['results'])
    ])
    regime_trial_data.to_csv("test/output/regime_trial_results.csv", index=False)
    print(f"[+] Regime trial results saved to: test/output/regime_trial_results.csv")
    
    baseline_trial_data = pd.DataFrame([
        {
            'trial': i,
            'final_balance': r['final_balance'],
            'success': r['success'],
            'failure_month': r['failure_month'] if r['failed'] else None,
            'min_balance': r['min_balance'],
            'max_balance': r['max_balance'],
        }
        for i, r in enumerate(baseline_results['results'])
    ])
    baseline_trial_data.to_csv("test/output/baseline_trial_results.csv", index=False)
    print(f"[+] Baseline trial results saved to: test/output/baseline_trial_results.csv")


def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print(" REGIME-AWARE PORTFOLIO PROJECTION")
    print("="*70 + "\n")
    
    # Check prerequisites
    check_prerequisites()
    
    # Display scenario
    print_scenario_summary()
    
    # Run simulations
    runner, regime_results, baseline_results = run_simulations()
    
    # Generate reports
    generate_comparison_report(regime_results, baseline_results)
    
    # Create visualizations
    create_visualizations(regime_results, baseline_results)
    
    # Save detailed outputs
    save_detailed_results(regime_results, baseline_results)
    
    print("\n" + "="*70)
    print("✓ PROJECTION COMPLETE")
    print("="*70)
    print("\nOutputs saved to test/output/:")
    print("  - projection_comparison.png")
    print("  - summary_comparison.csv")
    print("  - regime_trial_results.csv")
    print("  - baseline_trial_results.csv")
    print("\n")


if __name__ == "__main__":
    main()