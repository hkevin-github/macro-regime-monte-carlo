"""
Portfolio Projection Engine
Simulates portfolio evolution with returns, cash flows, withdrawals, and rebalancing.
This is the eMoney-equivalent cash flow engine, enhanced with regime awareness.
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class PortfolioConfig:
    """Configuration for portfolio simulation."""
    initial_balance: float
    annual_withdrawal: float
    withdrawal_inflation_adjust: bool = True
    annual_contribution: float = 0.0
    contribution_years: int = 0
    rebalance_frequency: int = 12  # months between rebalances
    failure_threshold: float = 0.0  # portfolio fails if balance drops below this
    target_end_balance: float = 0.0  # optional ending balance requirement


class PortfolioEngine:
    """
    Simulates portfolio evolution over time with cash flows and market returns.
    
    This engine handles:
    - Monthly portfolio returns
    - Inflation-adjusted withdrawals
    - Contributions (e.g., during accumulation phase)
    - Periodic rebalancing (assumed to be costless)
    - Failure detection (portfolio depletion)
    """
    
    def __init__(self, config: PortfolioConfig):
        """
        Initialize portfolio engine with configuration.
        
        Args:
            config: PortfolioConfig object with simulation parameters
        """
        self.config = config
    
    def project_portfolio(self,
                         returns: np.ndarray,
                         inflation: np.ndarray,
                         regime_path: Optional[np.ndarray] = None) -> Dict:
        """
        Project portfolio value over time given returns and inflation.
        
        Args:
            returns: array of monthly returns (as decimals, e.g., 0.01 = 1%)
            inflation: array of monthly inflation rates
            regime_path: optional array of regime indices for each period
            
        Returns:
            Dictionary containing:
                - balance_history: portfolio value each period
                - withdrawal_history: withdrawals each period
                - contribution_history: contributions each period
                - cumulative_inflation: cumulative inflation factor
                - success: whether portfolio survived to end
                - failure_month: month of failure (if failed)
                - final_balance: ending portfolio value
                - regime_path: regime sequence (if provided)
        """
        n_periods = len(returns)
        
        # Initialize tracking arrays
        balance_history = np.zeros(n_periods + 1)
        withdrawal_history = np.zeros(n_periods)
        contribution_history = np.zeros(n_periods)
        cumulative_inflation_history = np.zeros(n_periods + 1)
        
        balance_history[0] = self.config.initial_balance
        cumulative_inflation_history[0] = 1.0
        
        # Convert annual withdrawal to monthly
        monthly_withdrawal_base = self.config.annual_withdrawal / 12
        
        # Track failure
        failed = False
        failure_month = None
        
        for t in range(n_periods):
            # Update cumulative inflation
            cumulative_inflation = cumulative_inflation_history[t] * (1 + inflation[t])
            cumulative_inflation_history[t + 1] = cumulative_inflation
            
            # Apply market return to current balance
            balance = balance_history[t] * (1 + returns[t])
            
            # Calculate withdrawal (inflation-adjusted if configured)
            if self.config.withdrawal_inflation_adjust:
                withdrawal = monthly_withdrawal_base * cumulative_inflation
            else:
                withdrawal = monthly_withdrawal_base
            
            withdrawal_history[t] = withdrawal
            balance -= withdrawal
            
            # Calculate contribution (only during contribution years)
            month_number = t + 1
            year_number = (month_number - 1) // 12 + 1
            
            if year_number <= self.config.contribution_years:
                contribution = self.config.annual_contribution / 12
                contribution_history[t] = contribution
                balance += contribution
            
            # Check for failure
            if balance < self.config.failure_threshold:
                failed = True
                failure_month = t
                balance = 0.0  # Portfolio depleted
                balance_history[t + 1] = balance
                
                # Zero out remaining periods
                balance_history[t + 2:] = 0.0
                break
            
            # Rebalancing (assumed costless and instantaneous)
            # In a more sophisticated model, this would adjust asset allocation
            # For now, we assume returns already reflect the target allocation
            
            balance_history[t + 1] = balance
        
        # Check if ending balance meets target (if specified)
        final_balance = balance_history[-1]
        success = not failed
        
        if success and self.config.target_end_balance > 0:
            if final_balance < self.config.target_end_balance:
                success = False
        
        return {
            'balance_history': balance_history,
            'withdrawal_history': withdrawal_history,
            'contribution_history': contribution_history,
            'cumulative_inflation': cumulative_inflation_history,
            'success': success,
            'failed': failed,
            'failure_month': failure_month,
            'final_balance': final_balance,
            'regime_path': regime_path,
            'min_balance': balance_history.min(),
            'max_balance': balance_history.max(),
        }
    
    def project_multiple_scenarios(self,
                                   returns_array: np.ndarray,
                                   inflation_array: np.ndarray,
                                   regime_paths: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Project portfolio for multiple scenarios (Monte Carlo trials).
        
        Args:
            returns_array: shape (n_scenarios, n_periods)
            inflation_array: shape (n_scenarios, n_periods)
            regime_paths: optional, shape (n_scenarios, n_periods)
            
        Returns:
            List of result dictionaries, one per scenario
        """
        n_scenarios = returns_array.shape[0]
        results = []
        
        for i in range(n_scenarios):
            regime_path = regime_paths[i] if regime_paths is not None else None
            
            result = self.project_portfolio(
                returns=returns_array[i],
                inflation=inflation_array[i],
                regime_path=regime_path
            )
            
            results.append(result)
        
        return results
    
    def compute_success_rate(self, results: List[Dict]) -> float:
        """
        Calculate success rate across multiple scenarios.
        
        Args:
            results: list of projection results
            
        Returns:
            Success rate as a decimal (0.0 to 1.0)
        """
        successes = sum(1 for r in results if r['success'])
        return successes / len(results)
    
    def compute_percentile_outcomes(self, results: List[Dict], percentiles: List[float] = None) -> Dict:
        """
        Compute percentile outcomes for final balance.
        
        Args:
            results: list of projection results
            percentiles: list of percentiles to compute (default: [2.5, 50, 97.5])
            
        Returns:
            Dictionary mapping percentile to final balance
        """
        if percentiles is None:
            percentiles = [2.5, 50, 97.5]
        
        final_balances = np.array([r['final_balance'] for r in results])
        
        percentile_outcomes = {}
        for p in percentiles:
            percentile_outcomes[p] = np.percentile(final_balances, p)
        
        return percentile_outcomes
    
    def compute_failure_statistics(self, results: List[Dict]) -> Dict:
        """
        Compute statistics about failures.
        
        Returns:
            Dictionary with failure rate, median failure month, etc.
        """
        failures = [r for r in results if r['failed']]
        
        if not failures:
            return {
                'failure_rate': 0.0,
                'median_failure_month': None,
                'mean_failure_month': None,
            }
        
        failure_months = [r['failure_month'] for r in failures]
        
        return {
            'failure_rate': len(failures) / len(results),
            'median_failure_month': np.median(failure_months),
            'mean_failure_month': np.mean(failure_months),
            'min_failure_month': np.min(failure_months),
            'max_failure_month': np.max(failure_months),
        }


if __name__ == "__main__":
    # Test portfolio engine
    config = PortfolioConfig(
        initial_balance=1_000_000,
        annual_withdrawal=40_000,
        withdrawal_inflation_adjust=True,
        annual_contribution=0,
        contribution_years=0,
        failure_threshold=0,
        target_end_balance=0
    )
    
    engine = PortfolioEngine(config)
    
    # Simulate 30 years (360 months) with constant returns and inflation
    n_months = 360
    test_returns = np.random.normal(0.07/12, 0.15/np.sqrt(12), n_months)
    test_inflation = np.random.normal(0.02/12, 0.01/np.sqrt(12), n_months)
    
    result = engine.project_portfolio(test_returns, test_inflation)
    
    print(f"Success: {result['success']}")
    print(f"Final balance: ${result['final_balance']:,.0f}")
    print(f"Min balance: ${result['min_balance']:,.0f}")
    print(f"Max balance: ${result['max_balance']:,.0f}")
    
    # Test multiple scenarios
    n_scenarios = 100
    returns_array = np.random.normal(0.07/12, 0.15/np.sqrt(12), (n_scenarios, n_months))
    inflation_array = np.random.normal(0.02/12, 0.01/np.sqrt(12), (n_scenarios, n_months))
    
    results = engine.project_multiple_scenarios(returns_array, inflation_array)
    
    print(f"\nSuccess rate across {n_scenarios} scenarios: {engine.compute_success_rate(results):.1%}")
    
    percentiles = engine.compute_percentile_outcomes(results)
    print(f"Final balance percentiles:")
    for p, value in percentiles.items():
        print(f"  {p}th: ${value:,.0f}")