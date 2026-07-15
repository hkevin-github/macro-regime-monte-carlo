"""
Client Scenario Configuration
"""

from src.montecarlo.portfolio_engine import PortfolioConfig

CLIENT_SCENARIO = PortfolioConfig(
    initial_balance=1_500_000,
    annual_withdrawal=60_000,
    withdrawal_inflation_adjust=True,
    annual_contribution=0,
    contribution_years=0,
    failure_threshold=0,
    target_end_balance=0
)

N_TRIALS = 10_000
N_YEARS = 25
RANDOM_SEED = 42