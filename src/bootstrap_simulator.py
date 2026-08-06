# src/bootstrap_simulator.py
import numpy as np
import pandas as pd
from typing import Dict

class BootstrapMonteCarloSimulator:
    """Simple Monte Carlo simulator that randomly samples from historical returns"""
    
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize with historical return data
        
        Args:
            historical_data: DataFrame with columns ['equity_return', 'bond_return', 'cpi_yoy']
        """
        self.historical_data = historical_data
        self.equity_returns = historical_data['equity_return'].values
        self.bond_returns = historical_data['bond_return'].values
        
    def run_simulation(
        self,
        asset_class_weights: Dict[str, float],
        initial_balance: float,
        n_trials: int,
        n_years: int,
        annual_contribution: float = 0,
        annual_withdrawal: float = 0,
        annual_fee: float = 0.0,
        contribution_years: int = 0,
        failure_threshold: float = 0.0,
        target_end_balance: float = 0.0,
        random_state: int = 42
    ) -> Dict:
        """
        Run bootstrap Monte Carlo simulation
        
        Returns:
            Dictionary with results matching RegimeMultiAssetSimulator format
        """
        np.random.seed(random_state)
        
        # Aggregate weights into equity/bond allocation
        equity_weight = (
            asset_class_weights.get('US_Equity', 0) + 
            asset_class_weights.get('International_Equity', 0) +
            asset_class_weights.get('Real_Estate', 0) * 0.5 +
            asset_class_weights.get('Commodities', 0) * 0.5
        )
        bond_weight = (
            asset_class_weights.get('US_Bonds', 0) +
            asset_class_weights.get('Real_Estate', 0) * 0.3 +
            asset_class_weights.get('Commodities', 0) * 0.3 +
            asset_class_weights.get('Cash', 0)
        )
        
        # Normalize
        total = equity_weight + bond_weight
        if total > 0:
            equity_weight /= total
            bond_weight /= total
        else:
            equity_weight = 0.6
            bond_weight = 0.4
        
        # Monthly simulation
        n_months = n_years * 12
        all_balances = np.zeros((n_trials, n_months + 1))
        all_balances[:, 0] = initial_balance
        
        for trial in range(n_trials):
            balance = initial_balance
            
            # Randomly pick months from history (with replacement)
            random_month_indices = np.random.choice(
                len(self.equity_returns), 
                size=n_months, 
                replace=True
            )
            
            for month_idx, hist_idx in enumerate(random_month_indices):
                # Get returns from that randomly chosen historical month
                equity_return = self.equity_returns[hist_idx]
                bond_return = self.bond_returns[hist_idx]
                
                # Calculate portfolio return
                portfolio_return = (
                    equity_weight * equity_return + 
                    bond_weight * bond_return
                )
                
                # Apply return
                balance *= (1 + portfolio_return)
                
                # Apply fees (monthly)
                balance *= (1 - annual_fee / 12)
                
                # Apply contributions/withdrawals (monthly)
                current_year = month_idx // 12
                if contribution_years == 0 or current_year < contribution_years:
                    balance += annual_contribution / 12
                balance -= annual_withdrawal / 12
                
                # Allow negative balance (withdrawal debt)
                # balance = max(0, balance)  # Removed floor
                
                all_balances[trial, month_idx + 1] = balance
                
                # Continue tracking even if negative
                # if balance <= 0:
                #     break
        
        # Calculate metrics
        final_balances = all_balances[:, -1]
        
        if target_end_balance > 0:
            success_rate = np.mean(final_balances >= target_end_balance)
        else:
            success_rate = np.mean(final_balances > failure_threshold)
        
        results = {
            'all_balances': all_balances,
            'final_balances': final_balances,
            'success_rate': float(success_rate),
            'median_final': float(np.percentile(final_balances, 50)),
            'mean_final': float(np.mean(final_balances)),
            'min_final': float(np.min(final_balances)),
            'p20_final': float(np.percentile(final_balances, 20)),
            'p50_final': float(np.percentile(final_balances, 50)),
            'p80_final': float(np.percentile(final_balances, 80)),
            'max_final': float(np.max(final_balances)),
        }
        
        return results