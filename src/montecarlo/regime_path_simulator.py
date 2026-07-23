"""
Regime Path Simulator
Generates sequences of regime states using the HMM transition matrix.
"""

import numpy as np
from typing import Optional


class RegimePathSimulator:
    def __init__(self, transition_matrix: np.ndarray, regime_names: dict = None):
        self.transition_matrix = transition_matrix
        self.n_regimes = len(transition_matrix)
        self.regime_names = regime_names or {i: f"Regime_{i}" for i in range(self.n_regimes)}
        self._validate_transition_matrix()

    def _validate_transition_matrix(self):
        row_sums = self.transition_matrix.sum(axis=1)
        
        if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-12):
            max_deviation = np.max(np.abs(row_sums - 1.0))
            print(f"⚠️  WARNING: Transition matrix rows don't sum to 1.0")
            print(f"    Row sums: {row_sums}")
            print(f"    Max deviation: {max_deviation:.2e}")
            print(f"    Renormalizing...")
            
            # Renormalize
            self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]
            
            # Verify
            new_row_sums = self.transition_matrix.sum(axis=1)
            if not np.allclose(new_row_sums, 1.0, rtol=1e-14, atol=1e-15):
                raise ValueError("Failed to renormalize transition matrix")
            
            print(f"    Renormalization successful")

    def _compute_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution by solving the linear system.
        More numerically stable than eigenvector extraction.
        """
        P = np.asarray(self.transition_matrix, dtype=float)
        n = P.shape[0]
        
        # Solve: (P^T - I) π = 0 subject to sum(π) = 1
        A = P.T - np.eye(n)
        A[-1, :] = np.ones(n)
        b = np.zeros(n)
        b[-1] = 1.0
        
        try:
            stationary = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to least-squares if singular
            stationary, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Enforce non-negativity and normalization
        stationary = np.clip(stationary, 0.0, None)
        stationary = stationary / stationary.sum()
        
        # Validate
        if not np.allclose(stationary.sum(), 1.0, atol=1e-10):
            raise ValueError("Stationary distribution does not sum to 1")
        if np.any(stationary < -1e-10):
            raise ValueError("Stationary distribution has negative entries")
        
        return stationary

    def simulate_path(self,
                      n_periods: int,
                      initial_regime: Optional[int] = None,
                      initial_probs: Optional[np.ndarray] = None,
                      random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)

        if initial_regime is None:
            if initial_probs is None:
                initial_probs = self._compute_stationary_distribution()
            else:
                # Validate and normalize initial_probs
                initial_probs = np.asarray(initial_probs, dtype=float)
                initial_probs = np.clip(initial_probs, 0.0, None)
                initial_probs = initial_probs / initial_probs.sum()
            
            initial_regime = rng.choice(self.n_regimes, p=initial_probs)

        path = np.zeros(n_periods, dtype=int)
        path[0] = initial_regime

        for t in range(1, n_periods):
            current = path[t - 1]
            path[t] = rng.choice(self.n_regimes, p=self.transition_matrix[current])

        return path

    def simulate_multiple_paths(self,
                                n_paths: int,
                                n_periods: int,
                                initial_regime: Optional[int] = None,
                                initial_probs: Optional[np.ndarray] = None,
                                random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        paths = np.zeros((n_paths, n_periods), dtype=int)

        for i in range(n_paths):
            seed = rng.integers(0, 2**32 - 1)
            paths[i] = self.simulate_path(
                n_periods=n_periods,
                initial_regime=initial_regime,
                initial_probs=initial_probs,
                random_state=int(seed)
            )

        return paths

    def get_regime_occupancy(self, paths: np.ndarray) -> dict:
        occupancy = {}
        for regime in range(self.n_regimes):
            regime_name = self.regime_names[regime]
            regime_mask = (paths == regime)
            occupancy[regime_name] = {
                'frequency': float(regime_mask.mean()),
                'mean_duration': self._compute_mean_duration(paths, regime),
                'max_duration': self._compute_max_duration(paths, regime)
            }
        return occupancy

    def _compute_mean_duration(self, paths: np.ndarray, regime: int) -> float:
        durations = []
        for path in paths:
            mask = (path == regime)
            changes = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            durations.extend((ends - starts).tolist())
        return float(np.mean(durations)) if durations else 0.0

    def _compute_max_duration(self, paths: np.ndarray, regime: int) -> int:
        max_duration = 0
        for path in paths:
            mask = (path == regime)
            changes = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            if len(starts) > 0:
                max_duration = max(max_duration, int((ends - starts).max()))
        return max_duration