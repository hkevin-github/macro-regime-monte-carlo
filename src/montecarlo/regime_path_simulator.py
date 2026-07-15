"""
Regime Path Simulator
Generates sequences of regime states using the HMM transition matrix.
Supports multiple initialization strategies for forward simulation.
"""

import numpy as np
from typing import Optional


class RegimePathSimulator:
    """
    Simulates regime paths using Markov transition probabilities.
    """
    
    def __init__(self, transition_matrix: np.ndarray, regime_names: dict = None):
        """
        Initialize the regime path simulator.
        
        Args:
            transition_matrix: n x n matrix of transition probabilities
            regime_names: optional dict mapping regime indices to labels
        """
        self.transition_matrix = transition_matrix
        self.n_regimes = len(transition_matrix)
        self.regime_names = regime_names or {i: f"Regime_{i}" for i in range(self.n_regimes)}
        
        # Validate transition matrix
        self._validate_transition_matrix()
    
    def _validate_transition_matrix(self):
        """Ensure transition matrix is valid (rows sum to 1)."""
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Transition matrix rows must sum to 1.0")
    
    def simulate_path(self, 
                     n_periods: int, 
                     initial_regime: Optional[int] = None,
                     initial_probs: Optional[np.ndarray] = None,
                     random_state: Optional[int] = None) -> np.ndarray:
        """
        Simulate a single regime path.
        
        Args:
            n_periods: number of time periods to simulate
            initial_regime: starting regime (if None, samples from initial_probs)
            initial_probs: probability distribution over initial regimes
            random_state: random seed for reproducibility
            
        Returns:
            Array of regime indices for each period
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Determine initial regime
        if initial_regime is None:
            if initial_probs is None:
                # Default: use stationary distribution
                initial_probs = self._compute_stationary_distribution()
            initial_regime = np.random.choice(self.n_regimes, p=initial_probs)
        
        path = np.zeros(n_periods, dtype=int)
        path[0] = initial_regime
        
        # Simulate regime transitions
        for t in range(1, n_periods):
            current_regime = path[t - 1]
            next_regime = np.random.choice(
                self.n_regimes,
                p=self.transition_matrix[current_regime]
            )
            path[t] = next_regime
        
        return path
    
    def simulate_multiple_paths(self,
                               n_paths: int,
                               n_periods: int,
                               initial_regime: Optional[int] = None,
                               initial_probs: Optional[np.ndarray] = None,
                               random_state: Optional[int] = None) -> np.ndarray:
        """
        Simulate multiple independent regime paths.
        
        Returns:
            Array of shape (n_paths, n_periods) containing regime sequences
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        paths = np.zeros((n_paths, n_periods), dtype=int)
        
        for i in range(n_paths):
            paths[i] = self.simulate_path(
                n_periods=n_periods,
                initial_regime=initial_regime,
                initial_probs=initial_probs,
                random_state=None  # Let numpy handle randomness
            )
        
        return paths
    
    def _compute_stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution of the Markov chain."""
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, stationary_idx])
        return stationary / stationary.sum()
    
    def get_regime_occupancy(self, paths: np.ndarray) -> dict:
        """
        Compute regime occupancy statistics across multiple paths.
        
        Args:
            paths: array of shape (n_paths, n_periods)
            
        Returns:
            Dict with occupancy statistics per regime
        """
        occupancy = {}
        
        for regime in range(self.n_regimes):
            regime_name = self.regime_names[regime]
            regime_mask = (paths == regime)
            
            occupancy[regime_name] = {
                'frequency': regime_mask.mean(),
                'mean_duration': self._compute_mean_duration(paths, regime),
                'max_duration': self._compute_max_duration(paths, regime)
            }
        
        return occupancy
    
    def _compute_mean_duration(self, paths: np.ndarray, regime: int) -> float:
        """Compute average duration of stays in a given regime."""
        durations = []
        
        for path in paths:
            # Find consecutive runs of the regime
            regime_mask = (path == regime)
            regime_changes = np.diff(np.concatenate([[False], regime_mask, [False]]).astype(int))
            starts = np.where(regime_changes == 1)[0]
            ends = np.where(regime_changes == -1)[0]
            
            run_lengths = ends - starts
            durations.extend(run_lengths)
        
        return np.mean(durations) if durations else 0.0
    
    def _compute_max_duration(self, paths: np.ndarray, regime: int) -> int:
        """Compute maximum observed duration in a given regime."""
        max_duration = 0
        
        for path in paths:
            regime_mask = (path == regime)
            regime_changes = np.diff(np.concatenate([[False], regime_mask, [False]]).astype(int))
            starts = np.where(regime_changes == 1)[0]
            ends = np.where(regime_changes == -1)[0]
            
            if len(starts) > 0:
                max_duration = max(max_duration, (ends - starts).max())
        
        return max_duration


if __name__ == "__main__":
    # Test with a simple 3-state example
    test_transition = np.array([
        [0.8, 0.15, 0.05],
        [0.1, 0.7, 0.2],
        [0.05, 0.25, 0.7]
    ])
    
    simulator = RegimePathSimulator(test_transition)
    
    # Simulate a single path
    path = simulator.simulate_path(n_periods=120, initial_regime=0, random_state=42)
    print(f"Single path (first 20 periods): {path[:20]}")
    
    # Simulate multiple paths
    paths = simulator.simulate_multiple_paths(
        n_paths=1000,
        n_periods=120,
        initial_regime=0,
        random_state=42
    )
    
    print(f"\nSimulated {len(paths)} paths of {paths.shape[1]} periods each")
    
    # Compute occupancy
    occupancy = simulator.get_regime_occupancy(paths)
    print("\nRegime occupancy statistics:")
    for regime, stats in occupancy.items():
        print(f"{regime}: {stats['frequency']:.1%} frequency, "
              f"{stats['mean_duration']:.1f} avg duration")