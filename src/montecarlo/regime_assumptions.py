"""
Regime Market Assumptions Loader
Loads trained HMM artifacts and regime-conditioned market statistics
for use in Monte Carlo simulation.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class RegimeAssumptions:
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)

        # Load transition matrix FIRST to determine expected state count
        self.transition_matrix = self._load_transition_matrix()
        self.n_regimes = self.transition_matrix.shape[0]
        
        # Load model matching transition matrix dimensions
        self.model = self._load_hmm_model()
        
        # Validate consistency
        if self.model.n_components != self.n_regimes:
            raise ValueError(
                f"Model states ({self.model.n_components}) do not match "
                f"transition matrix size ({self.n_regimes}). Artifacts may be mismatched."
            )
        
        self.regime_stats = self._load_regime_stats()
        self.regime_labels = self._load_regime_labels()

        print(f"[+] Loaded {self.n_regimes}-state regime model")
        print(f"[+] Regimes: {list(self.regime_labels['regime_names'].values())}")

    def _load_transition_matrix(self) -> np.ndarray:
        """Load transition matrix first to determine expected model size."""
        transition_path = self.models_dir / "transition_matrix.csv"
        if not transition_path.exists():
            raise FileNotFoundError(f"Transition matrix not found: {transition_path}")
        df = pd.read_csv(transition_path, index_col=0)
        return df.values

    def _load_hmm_model(self):
        """Load HMM model matching the transition matrix dimensions."""
        expected_states = self.n_regimes
        
        # Look for matching model file
        model_path = self.models_dir / f"hmm_{expected_states}state.pkl"
        
        if model_path.exists():
            print(f"[*] Loading HMM model from: {model_path}")
            return joblib.load(model_path)
        
        # Fallback: search for any model with matching state count
        print(f"[!] Expected model file not found: {model_path}")
        print(f"[*] Searching for alternative models with {expected_states} states...")
        
        model_files = list(self.models_dir.glob("hmm_*state.pkl"))
        for path in sorted(model_files):
            try:
                model = joblib.load(path)
                if model.n_components == expected_states:
                    print(f"[*] Found matching model: {path}")
                    return model
            except Exception as e:
                print(f"[!] Failed to load {path}: {e}")
                continue
        
        # No matching model found
        available = [p.name for p in model_files]
        raise FileNotFoundError(
            f"No HMM model found with {expected_states} states.\n"
            f"Expected: {model_path}\n"
            f"Available models: {available}\n"
            f"Transition matrix requires {expected_states} states."
        )

    def _load_regime_stats(self) -> dict:
        stats_path = self.models_dir / "regime_market_assumptions.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Regime stats not found: {stats_path}")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Validate state count matches
        if len(stats) != self.n_regimes:
            raise ValueError(
                f"Regime stats count ({len(stats)}) does not match "
                f"expected state count ({self.n_regimes})"
            )
        
        return {int(k): v for k, v in stats.items()}

    def _load_regime_labels(self) -> dict:
        labels_path = self.models_dir / "regime_labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"Regime labels not found: {labels_path}")
        with open(labels_path, 'r') as f:
            labels = json.load(f)

        labels['regime_names'] = {int(k): v for k, v in labels['regime_names'].items()}
        labels['regime_colors'] = {int(k): v for k, v in labels['regime_colors'].items()}
        
        # Validate label count matches
        if len(labels['regime_names']) != self.n_regimes:
            raise ValueError(
                f"Regime label count ({len(labels['regime_names'])}) does not match "
                f"expected state count ({self.n_regimes})"
            )
        
        return labels

    def get_regime_return_params(self, regime: int) -> dict:
        if regime not in self.regime_stats:
            raise ValueError(f"Invalid regime: {regime}")

        stats = self.regime_stats[regime]
        return {
            'equity_mean': stats['equity_mean'],
            'equity_std': stats['equity_std'],
            'equity_skew': stats.get('equity_skew', 0),
            'equity_kurt': stats.get('equity_kurt', 3),
            'inflation_yoy_mean': stats.get('inflation_mean', 0.02),
            'inflation_yoy_std': stats.get('inflation_std', 0.01),
        }

    def get_stationary_distribution(self) -> np.ndarray:
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

    def get_current_regime_probabilities(self, labeled_data_path: str = "data/processed/regime_labeled_dataset.csv") -> np.ndarray:
        df = pd.read_csv(labeled_data_path, index_col='date', parse_dates=True)
        df = df.sort_index()
        prob_cols = [f'regime_{i}_prob' for i in range(self.n_regimes)]
        return df[prob_cols].iloc[-1].values

    def print_summary(self):
        print("\n" + "=" * 70)
        print("REGIME MARKET ASSUMPTIONS SUMMARY")
        print("=" * 70)

        for regime in range(self.n_regimes):
            label = self.regime_labels['regime_names'][regime]
            stats = self.regime_stats[regime]
            print(f"\nRegime {regime}: {label}")
            print(f"  Frequency: {stats['frequency']:.1%}")
            print(f"  Avg Duration: {stats['duration_mean']:.1f} months")
            print(f"  Equity Return: {stats['equity_mean']:.2%} ± {stats['equity_std']:.2%}")
            if stats.get('inflation_mean') is not None:
                print(f"  CPI YoY: {stats['inflation_mean']:.2%} ± {stats['inflation_std']:.2%}")

        print("\n[+] Transition Matrix:")
        df = pd.DataFrame(
            self.transition_matrix,
            index=[self.regime_labels['regime_names'][i] for i in range(self.n_regimes)],
            columns=[self.regime_labels['regime_names'][i] for i in range(self.n_regimes)]
        )
        print(df.round(3))

        print("\n[+] Stationary Distribution:")
        stationary = self.get_stationary_distribution()
        for regime, prob in enumerate(stationary):
            label = self.regime_labels['regime_names'][regime]
            print(f"  {label}: {prob:.1%}")

        print("=" * 70)


if __name__ == "__main__":
    assumptions = RegimeAssumptions()
    assumptions.print_summary()