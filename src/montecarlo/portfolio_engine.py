"""
Portfolio Projection Engine
Simulates portfolio evolution with monthly returns, monthly inflation,
cash flows, withdrawals, and rebalancing.
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class PortfolioConfig:
    initial_balance: float
    annual_withdrawal: float
    withdrawal_inflation_adjust: bool = True
    annual_contribution: float = 0.0
    contribution_years: int = 0
    rebalance_frequency: int = 12
    failure_threshold: float = 0.0
    target_end_balance: float = 0.0


class PortfolioEngine:
    def __init__(self, config: PortfolioConfig):
        self.config = config

    def project_portfolio(self,
                          returns: np.ndarray,
                          inflation: np.ndarray,
                          regime_path: Optional[np.ndarray] = None) -> Dict:
        n_periods = len(returns)

        balance_history = np.zeros(n_periods + 1)
        withdrawal_history = np.zeros(n_periods)
        contribution_history = np.zeros(n_periods)
        cumulative_inflation_history = np.zeros(n_periods + 1)

        balance_history[0] = self.config.initial_balance
        cumulative_inflation_history[0] = 1.0

        monthly_withdrawal_base = self.config.annual_withdrawal / 12.0

        failed = False
        failure_month = None

        for t in range(n_periods):
            cumulative_inflation = cumulative_inflation_history[t] * (1.0 + inflation[t])
            cumulative_inflation_history[t + 1] = cumulative_inflation

            balance = balance_history[t] * (1.0 + returns[t])

            if self.config.withdrawal_inflation_adjust:
                withdrawal = monthly_withdrawal_base * cumulative_inflation
            else:
                withdrawal = monthly_withdrawal_base

            withdrawal_history[t] = withdrawal
            balance -= withdrawal

            month_number = t + 1
            year_number = (month_number - 1) // 12 + 1

            if year_number <= self.config.contribution_years:
                contribution = self.config.annual_contribution / 12.0
                contribution_history[t] = contribution
                balance += contribution

            if balance < self.config.failure_threshold:
                failed = True
                failure_month = t
                balance = 0.0
                balance_history[t + 1] = balance
                balance_history[t + 2:] = 0.0
                break

            balance_history[t + 1] = balance

        final_balance = balance_history[-1]
        success = not failed

        if success and self.config.target_end_balance > 0:
            if final_balance < self.config.target_end_balance:
                success = False

        running_max = np.maximum.accumulate(balance_history[1:])
        drawdown = np.where(running_max > 0, 1.0 - balance_history[1:] / running_max, 0.0)
        max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0.0

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
            'min_balance': float(balance_history.min()),
            'max_balance': float(balance_history.max()),
            'max_drawdown': max_drawdown,
        }

    def project_multiple_scenarios(self,
                                   returns_array: np.ndarray,
                                   inflation_array: np.ndarray,
                                   regime_paths: Optional[np.ndarray] = None) -> List[Dict]:
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
        return sum(1 for r in results if r['success']) / len(results)

    def compute_percentile_outcomes(self, results: List[Dict], percentiles: List[float] = None) -> Dict:
        if percentiles is None:
            percentiles = [2.5, 50, 97.5]
        final_balances = np.array([r['final_balance'] for r in results])
        return {p: np.percentile(final_balances, p) for p in percentiles}

    def compute_failure_statistics(self, results: List[Dict]) -> Dict:
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
            'median_failure_month': float(np.median(failure_months)),
            'mean_failure_month': float(np.mean(failure_months)),
            'min_failure_month': int(np.min(failure_months)),
            'max_failure_month': int(np.max(failure_months)),
        }