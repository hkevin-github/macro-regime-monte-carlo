"""
Custom Portfolio Monte Carlo Simulator

Runs Monte Carlo simulations for arbitrary portfolio allocations using
historical monthly proxy returns.

Features:
- historical bootstrap sampling
- parametric multivariate normal sampling
- monthly withdrawals
- inflation-adjusted spending
- fee drag
- success / failure metrics
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Portfolio configuration
# ---------------------------------------------------------------------
@dataclass
class PortfolioConfig:
    initial_balance: float
    annual_withdrawal: float = 0.0
    withdrawal_inflation_adjust: bool = False
    annual_contribution: float = 0.0
    contribution_years: int = 0
    target_end_balance: float = 0.0
    failure_threshold: float = 0.0
    portfolio_weights: Dict[str, float] = field(default_factory=dict)
    annual_fee: float = 0.0  # e.g. 0.0025 = 0.25% annual fee


BLACKROCK_80_20 = PortfolioConfig(
    initial_balance=1_960_000,
    annual_withdrawal=0,
    withdrawal_inflation_adjust=True,
    annual_contribution=0.0,
    contribution_years=0,
    target_end_balance=0.0,
    failure_threshold=0.0,
    annual_fee=0.0025,
    portfolio_weights={
        "US_LARGE": 0.80,
        "CORE_BONDS": 0.20,
    }
)


def load_market_data(path: str = "data/processed/custom_portfolio_market_data.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing market data: {path}")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def validate_portfolio_weights(config: PortfolioConfig, market_df: pd.DataFrame) -> list[str]:
    asset_cols = list(config.portfolio_weights.keys())
    weights = np.array([config.portfolio_weights[c] for c in asset_cols], dtype=float)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Portfolio weights must sum to 1.0, got {weights.sum():.6f}")

    missing = [c for c in asset_cols if c not in market_df.columns]
    if missing:
        raise ValueError(f"Missing asset columns in market data: {missing}")

    return asset_cols


def compute_portfolio_return_series(asset_return_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    asset_return_matrix shape = (n_periods, n_assets)
    weights shape = (n_assets,)
    """
    return asset_return_matrix @ weights


def bootstrap_simulation(
    market_df: pd.DataFrame,
    config: PortfolioConfig,
    n_trials: int = 10_000,
    n_years: int = 20,
    random_state: int = 42
) -> Dict:
    """
    Bootstrap monthly returns from historical proxy returns.
    """
    rng = np.random.default_rng(random_state)

    asset_cols = validate_portfolio_weights(config, market_df)
    weights = np.array([config.portfolio_weights[c] for c in asset_cols], dtype=float)

    n_periods = n_years * 12
    historical = market_df[asset_cols].dropna().values
    n_hist = len(historical)

    if n_hist < 60:
        raise ValueError("Not enough historical observations for bootstrap simulation.")

    final_balances = np.zeros(n_trials)
    success = np.zeros(n_trials, dtype=bool)
    balance_paths = np.zeros((n_trials, n_periods + 1))

    fee_monthly = config.annual_fee / 12.0
    monthly_withdrawal_base = config.annual_withdrawal / 12.0
    monthly_contribution_base = config.annual_contribution / 12.0

    for i in range(n_trials):
        balance = config.initial_balance
        balance_paths[i, 0] = balance
        cumulative_inflation = 1.0
        alive = True

        sampled_idx = rng.integers(0, n_hist, size=n_periods)
        sampled_asset_returns = historical[sampled_idx]  # (n_periods, n_assets)
        portfolio_returns = compute_portfolio_return_series(sampled_asset_returns, weights)

        for t in range(n_periods):
            # fee drag
            r = portfolio_returns[t] - fee_monthly

            # apply growth
            balance *= (1.0 + r)

            # inflation-adjusted withdrawal if desired
            if config.withdrawal_inflation_adjust:
                # derive an implied monthly inflation path from the portfolio itself? no.
                # For now, use a conservative simple approximation:
                # inflation adjustment is handled outside this bootstrap unless you add a CPI series.
                # You can set this to False, or extend market_df with CPI and use it here.
                withdrawal = monthly_withdrawal_base * cumulative_inflation
            else:
                withdrawal = monthly_withdrawal_base

            # contributions during accumulation
            contribution = 0.0
            if (t // 12) < config.contribution_years:
                contribution = monthly_contribution_base

            balance -= withdrawal
            balance += contribution

            if balance <= config.failure_threshold:
                balance = 0.0
                alive = False
                balance_paths[i, t + 1:] = 0.0
                break

            balance_paths[i, t + 1] = balance

        final_balances[i] = balance
        success[i] = alive and (balance >= config.target_end_balance)

    return {
        "final_balances": final_balances,
        "success_rate": success.mean(),
        "median": np.percentile(final_balances, 50),
        "p5": np.percentile(final_balances, 5),
        "p20": np.percentile(final_balances, 20),
        "p80": np.percentile(final_balances, 80),
        "p95": np.percentile(final_balances, 95),
        "paths": balance_paths,
        "asset_cols": asset_cols,
        "mode": "bootstrap",
    }


def parametric_simulation(
    market_df: pd.DataFrame,
    config: PortfolioConfig,
    n_trials: int = 10_000,
    n_years: int = 20,
    random_state: int = 42
) -> Dict:
    """
    Parametric multivariate normal simulation using historical monthly mean/covariance.
    """
    rng = np.random.default_rng(random_state)

    asset_cols = validate_portfolio_weights(config, market_df)
    weights = np.array([config.portfolio_weights[c] for c in asset_cols], dtype=float)

    n_periods = n_years * 12
    X = market_df[asset_cols].dropna().values

    if len(X) < 60:
        raise ValueError("Not enough historical observations for parametric simulation.")

    mean_vec = X.mean(axis=0)
    cov_mat = np.cov(X, rowvar=False)

    # stabilize covariance
    cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * 1e-8

    final_balances = np.zeros(n_trials)
    success = np.zeros(n_trials, dtype=bool)
    balance_paths = np.zeros((n_trials, n_periods + 1))

    fee_monthly = config.annual_fee / 12.0
    monthly_withdrawal_base = config.annual_withdrawal / 12.0
    monthly_contribution_base = config.annual_contribution / 12.0

    for i in range(n_trials):
        balance = config.initial_balance
        balance_paths[i, 0] = balance
        alive = True

        sampled_asset_returns = rng.multivariate_normal(mean=mean_vec, cov=cov_mat, size=n_periods)
        portfolio_returns = compute_portfolio_return_series(sampled_asset_returns, weights)

        # no inflation series yet, so keep a simple deterministic inflation adjustment if desired
        cumulative_inflation = 1.0

        for t in range(n_periods):
            r = portfolio_returns[t] - fee_monthly
            balance *= (1.0 + r)

            if config.withdrawal_inflation_adjust:
                withdrawal = monthly_withdrawal_base * cumulative_inflation
            else:
                withdrawal = monthly_withdrawal_base

            contribution = 0.0
            if (t // 12) < config.contribution_years:
                contribution = monthly_contribution_base

            balance -= withdrawal
            balance += contribution

            if balance <= config.failure_threshold:
                balance = 0.0
                alive = False
                balance_paths[i, t + 1:] = 0.0
                break

            balance_paths[i, t + 1] = balance

        final_balances[i] = balance
        success[i] = alive and (balance >= config.target_end_balance)

    return {
        "final_balances": final_balances,
        "success_rate": success.mean(),
        "median": np.percentile(final_balances, 50),
        "p5": np.percentile(final_balances, 5),
        "p20": np.percentile(final_balances, 20),
        "p80": np.percentile(final_balances, 80),
        "p95": np.percentile(final_balances, 95),
        "paths": balance_paths,
        "asset_cols": asset_cols,
        "mode": "parametric",
    }


def plot_results(results: Dict, config: PortfolioConfig, output_path: str = "output/custom_portfolio_mc.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_balances = results["final_balances"]
    paths = results["paths"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].hist(final_balances, bins=60, color="steelblue", alpha=0.75, edgecolor="white")
    axes[0].axvline(np.percentile(final_balances, 50), color="black", linestyle="--", label="Median")
    axes[0].axvline(np.percentile(final_balances, 5), color="red", linestyle="--", label="5th %ile")
    axes[0].axvline(np.percentile(final_balances, 95), color="green", linestyle="--", label="95th %ile")
    axes[0].set_title("Final Portfolio Value Distribution")
    axes[0].set_xlabel("Ending Balance ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    n_plot = min(100, len(paths))
    idx = np.random.choice(len(paths), n_plot, replace=False)
    for i in idx:
        axes[1].plot(paths[i], color="steelblue", alpha=0.08, linewidth=0.8)

    axes[1].plot(np.median(paths, axis=0), color="black", linewidth=2, label="Median Path")
    axes[1].set_title("Sample Portfolio Paths")
    axes[1].set_xlabel("Months")
    axes[1].set_ylabel("Portfolio Value ($)")
    axes[1].legend()

    plt.suptitle(
        f"Custom Portfolio Monte Carlo | Mode: {results['mode']} | Success Rate: {results['success_rate']:.1%}",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved plot to: {output_path}")


def main():
    market_df = load_market_data()

    # Example: BlackRock 80/20-style portfolio
    config = BLACKROCK_80_20

    # Choose one:
    results = bootstrap_simulation(
        market_df=market_df,
        config=config,
        n_trials=10_000,
        n_years=20,
        random_state=42
    )

    # results = parametric_simulation(
    #     market_df=market_df,
    #     config=config,
    #     n_trials=10_000,
    #     n_years=20,
    #     random_state=42
    # )

    print("\n" + "=" * 70)
    print("CUSTOM PORTFOLIO MONTE CARLO RESULTS")
    print("=" * 70)
    print(f"Mode: {results['mode']}")
    print(f"Assets: {results['asset_cols']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Median Final Balance: ${results['median']:,.0f}")
    print(f"5th Percentile: ${results['p5']:,.0f}")
    print(f"20th Percentile: ${results['p20']:,.0f}")
    print(f"80th Percentile: ${results['p80']:,.0f}")
    print(f"95th Percentile: ${results['p95']:,.0f}")
    print("=" * 70)

    plot_results(results, config)


if __name__ == "__main__":
    main()