#!/usr/bin/env python3
"""
Analyze a portfolio, decompose holdings into asset classes, and run Monte Carlo.
Now reads simulation parameters from config/info.json
"""

import sys
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from portfolio_analyzer import PortfolioAnalyzer
from regime_portfolio_simulator import RegimeMultiAssetSimulator


def load_config(config_path: str = "config/info.json") -> dict:
    """Load simulation configuration from JSON file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"[!] Config file not found: {config_path}")
        print(f"[*] Creating default config file...")
        
        default_config = {
            "simulation": {
                "n_trials": 10000,
                "n_years": 30,
                "random_seed": 42
            },
            "portfolio": {
                "annual_contribution": 0,
                "annual_withdrawal": 0,
                "contribution_years": 0,
                "withdrawal_inflation_adjust": True,
                "annual_fee": 0.0,
                "failure_threshold": 0.0,
                "target_end_balance": 0.0
            },
            "output": {
                "save_all_paths": False,
                "output_dir": "output"
            }
        }
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"[+] Created default config: {config_path}")
        return default_config
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"[+] Loaded configuration from: {config_path}")
    return config


def plot_results(results: dict, summary: dict, output_path: str):
    all_balances = results["all_balances"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Portfolio trajectory with 20th-80th percentile band
    ax1 = axes[0, 0]
    years = np.linspace(0, all_balances.shape[1] - 1, all_balances.shape[1]) / 12.0
    p20 = np.percentile(all_balances, 20, axis=0)
    p50 = np.percentile(all_balances, 50, axis=0)
    p80 = np.percentile(all_balances, 80, axis=0)

    ax1.fill_between(years, p20, p80, alpha=0.2, color="blue", label="20th-80th percentile")
    ax1.plot(years, p50, color="blue", lw=2, label="Median (50th)")
    ax1.axhline(summary["total_value"], color="green", ls="--", label="Initial", lw=1.5)
    ax1.set_title("Portfolio Balance Trajectories")
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Balance ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))

    # Plot 2: Final balance distribution
    ax2 = axes[0, 1]
    final_balances = results["final_balances"]
    ax2.hist(final_balances, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    
    # Add percentile lines
    p20_final = np.percentile(final_balances, 20)
    p50_final = np.percentile(final_balances, 50)
    p80_final = np.percentile(final_balances, 80)
    
    ax2.axvline(p20_final, color="orange", lw=2, ls="--", label=f"20th: ${p20_final:,.0f}")
    ax2.axvline(p50_final, color="red", lw=2, label=f"Median: ${p50_final:,.0f}")
    ax2.axvline(p80_final, color="purple", lw=2, ls="--", label=f"80th: ${p80_final:,.0f}")
    ax2.axvline(summary["total_value"], color="green", ls=":", lw=1.5, label=f"Initial: ${summary['total_value']:,.0f}")
    
    ax2.set_title(f"Final Balance Distribution (Success: {results['success_rate']:.1%})")
    ax2.set_xlabel("Final Balance ($)")
    ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Asset allocation pie chart
    ax3 = axes[1, 0]
    asset_classes = list(summary["asset_class_weights"].keys())
    weights = list(summary["asset_class_weights"].values())
    colors = plt.cm.Set3(range(len(asset_classes)))
    ax3.pie(weights, labels=asset_classes, autopct="%1.1f%%", startangle=90, colors=colors)
    ax3.set_title("Asset Class Allocation")

    # Plot 4: Regime time distribution
    ax4 = axes[1, 1]
    regime_counts = {}
    for regime_id in range(results["all_regimes"].max() + 1):
        regime_counts[regime_id] = int(np.sum(results["all_regimes"] == regime_id))

    try:
        from montecarlo.regime_assumptions import RegimeAssumptions
        assumptions = RegimeAssumptions()
        regime_names = [assumptions.regime_labels["regime_names"][i] for i in sorted(regime_counts)]
        regime_colors_dict = assumptions.regime_labels["regime_colors"]
        bar_colors = [regime_colors_dict[i] for i in sorted(regime_counts)]
    except Exception:
        regime_names = [f"Regime {i}" for i in sorted(regime_counts)]
        bar_colors = plt.cm.Set2(range(len(regime_counts)))

    counts = [regime_counts[i] for i in sorted(regime_counts)]
    ax4.bar(regime_names, counts, alpha=0.7, edgecolor="black", color=bar_colors)
    ax4.set_title("Time Spent in Each Regime (All Simulations)")
    ax4.set_ylabel("Total Months")
    ax4.grid(True, axis="y", alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[+] Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze portfolio and run regime-aware Monte Carlo simulation"
    )
    parser.add_argument("portfolio", help="Path to CSV portfolio file with columns ticker,shares")
    parser.add_argument("--config", default="config/info.json", 
                       help="Path to configuration JSON file (default: config/info.json)")
    parser.add_argument("--years", type=int, default=None,
                       help="Override: Simulation horizon in years")
    parser.add_argument("--trials", type=int, default=None,
                       help="Override: Number of Monte Carlo trials")
    parser.add_argument("--contribution", type=float, default=None,
                       help="Override: Annual contribution amount")
    parser.add_argument("--withdrawal", type=float, default=None,
                       help="Override: Annual withdrawal amount")
    parser.add_argument("--output", default=None,
                       help="Override: Output directory")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Command-line arguments override config file
    n_years = args.years if args.years is not None else config["simulation"]["n_years"]
    n_trials = args.trials if args.trials is not None else config["simulation"]["n_trials"]
    annual_contribution = args.contribution if args.contribution is not None else config["portfolio"].get("annual_contribution", 0.0)
    annual_withdrawal = args.withdrawal if args.withdrawal is not None else config["portfolio"].get("annual_withdrawal", 0.0)
    annual_fee = config["portfolio"].get("annual_fee", 0.0)
    contribution_years = int(config["portfolio"].get("contribution_years", 0))
    failure_threshold = config["portfolio"].get("failure_threshold", 0.0)
    target_end_balance = config["portfolio"].get("target_end_balance", 0.0)
    output_dir = Path(args.output if args.output is not None else config["output"]["output_dir"])
    random_seed = config["simulation"]["random_seed"]

    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 1: PORTFOLIO ANALYSIS")
    print("=" * 60)

    analyzer = PortfolioAnalyzer()
    summary = analyzer.analyze(args.portfolio)
    analyzer.save_analysis(str(output_dir / "portfolio_analysis.json"))

    print("\n" + "=" * 60)
    print("STEP 2: MONTE CARLO SIMULATION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Trials: {n_trials:,}")
    print(f"  Horizon: {n_years} years")
    print(f"  Annual contribution: ${annual_contribution:,.0f}")
    print(f"  Annual withdrawal: ${annual_withdrawal:,.0f}")
    print(f"  Random seed: {random_seed}")

    simulator = RegimeMultiAssetSimulator()
    results = simulator.run_simulation(
        asset_class_weights=summary["asset_class_weights"],
        initial_balance=summary["total_value"],
        n_trials=n_trials,
        n_years=n_years,
        annual_contribution=annual_contribution,
        annual_withdrawal=annual_withdrawal,
        annual_fee=annual_fee,
        contribution_years=contribution_years,
        failure_threshold=failure_threshold,
        target_end_balance=target_end_balance,
        random_state=random_seed,
    )

    # Save results
    print("\n" + "=" * 60)
    print("STEP 3: SAVING RESULTS")
    print("=" * 60)
    
    # Compute additional percentiles
    final_balances = results["final_balances"]
    p20_final = float(np.percentile(final_balances, 20))
    p80_final = float(np.percentile(final_balances, 80))
    
    # Save summary statistics with 20th/80th percentiles
    summary_results = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) 
        for k, v in results.items() 
        if k not in ["all_balances", "all_regimes"]
    }
    
    # Add 20th and 80th percentiles
    summary_results['p20_final'] = p20_final
    summary_results['p80_final'] = p80_final
    
    with open(output_dir / "simulation_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"[+] Summary saved to: {output_dir / 'simulation_results.json'}")
    
    # Optionally save full paths
    if config["output"].get("save_all_paths", False):
        np.savez_compressed(
            output_dir / "simulation_paths.npz",
            all_balances=results["all_balances"],
            all_regimes=results["all_regimes"]
        )
        print(f"[+] Full paths saved to: {output_dir / 'simulation_paths.npz'}")

    print("\n" + "=" * 60)
    print("STEP 4: VISUALIZATION")
    print("=" * 60)
    plot_results(results, summary, str(output_dir / "simulation_results.png"))

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Outputs saved to: {output_dir}/")
    print(f"\nKey Results:")
    print(f"  Success rate: {results['success_rate']:.1%}")
    print(f"  Median final balance: ${results['median_final']:,.0f}")
    print(f"  20th percentile: ${p20_final:,.0f}")
    print(f"  80th percentile: ${p80_final:,.0f}")
    print(f"  10th percentile: ${results['p10_final']:,.0f}")
    print(f"  90th percentile: ${results['p90_final']:,.0f}")


if __name__ == "__main__":
    main()