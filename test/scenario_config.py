"""
Client Scenario Configuration
Define the financial situation you want to project here.
"""

from src.montecarlo.portfolio_engine import PortfolioConfig


# =============================================================================
# SCENARIO: Retired Couple, Age 65, 25-Year Projection
# =============================================================================

CLIENT_SCENARIO = PortfolioConfig(
    initial_balance=1_500_000,        # Starting portfolio
    annual_withdrawal=60_000,         # Spending need (inflation-adjusted)
    withdrawal_inflation_adjust=True, # Adjust for inflation each year
    annual_contribution=0,            # No more contributions (retired)
    contribution_years=0,             # N/A
    failure_threshold=0,              # Fails if balance hits $0
    target_end_balance=0              # No bequest requirement
)

# Simulation parameters
N_TRIALS = 10_000      # Number of Monte Carlo scenarios
N_YEARS = 25           # Planning horizon
RANDOM_SEED = 42       # For reproducibility


# =============================================================================
# ALTERNATIVE SCENARIOS (uncomment to use)
# =============================================================================

# # Scenario: Pre-retiree still contributing
# CLIENT_SCENARIO = PortfolioConfig(
#     initial_balance=800_000,
#     annual_withdrawal=0,              # Not withdrawing yet
#     withdrawal_inflation_adjust=False,
#     annual_contribution=30_000,       # Still saving $30k/year
#     contribution_years=10,            # For next 10 years
#     failure_threshold=0,
#     target_end_balance=0
# )
# N_YEARS = 10

# # Scenario: High net worth with bequest goal
# CLIENT_SCENARIO = PortfolioConfig(
#     initial_balance=5_000_000,
#     annual_withdrawal=150_000,
#     withdrawal_inflation_adjust=True,
#     annual_contribution=0,
#     contribution_years=0,
#     failure_threshold=0,
#     target_end_balance=2_000_000      # Want to leave $2M to heirs
# )
# N_YEARS = 30

# # Scenario: Conservative retiree with pension
# CLIENT_SCENARIO = PortfolioConfig(
#     initial_balance=750_000,
#     annual_withdrawal=20_000,          # Low withdrawal (has pension)
#     withdrawal_inflation_adjust=True,
#     annual_contribution=0,
#     contribution_years=0,
#     failure_threshold=0,
#     target_end_balance=0
# )
# N_YEARS = 30