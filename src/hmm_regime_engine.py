"""
Regime Inference Engine - Hidden Markov Model Training
Trains a Gaussian Hidden Markov Model directly on the monthly macro
observation matrix produced by the ingestion pipeline.
"""

import os
import json
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib


FEATURE_COLS = [
    'growth_yoy_zscore',
    'cpi_yoy_zscore',
    'unemployment_delta_zscore',
    'yield_spread_delta_zscore',
    'credit_spread_delta_zscore',
]


def load_processed_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data matrix not discovered at {file_path}")
    print(f"[*] Ingesting historical master data from: {file_path}")
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df = df.sort_index()
    if df.index.duplicated().any():
        print("⚠️  WARNING: Duplicate dates found. Keeping last occurrence.")
        df = df[~df.index.duplicated(keep='last')]
    return df


def load_model_selection_recommendation() -> int:
    rec_path = "data/models/model_selection_recommendation.json"
    if os.path.exists(rec_path):
        with open(rec_path, 'r') as f:
            rec = json.load(f)
        recommended = int(rec['recommended_states'])
        print(f"[*] Using recommended number of states from model selection: {recommended}")
        return recommended

    print("⚠️  No model selection recommendation found. Using default: 4 states")
    return 4


def train_regime_hmm(df: pd.DataFrame, n_components: int = None) -> tuple[GaussianHMM, pd.DataFrame]:
    if n_components is None:
        n_components = load_model_selection_recommendation()

    monthly_df = df[FEATURE_COLS].dropna()
    X_monthly = monthly_df.values

    model = GaussianHMM(
        n_components=n_components,
        covariance_type='diag',
        n_iter=1000,
        init_params="stmc",
        random_state=42
    )

    print(f"[*] Training {n_components}-state HMM (unsupervised initialization)...")
    model.fit(X_monthly)

    monthly_states = model.predict(X_monthly)
    posteriors = model.predict_proba(X_monthly)

    labeled_df = monthly_df.copy()
    labeled_df['inferred_regime'] = monthly_states
    labeled_df['regime_confidence'] = posteriors.max(axis=1)

    sorted_probs = np.sort(posteriors, axis=1)
    labeled_df['regime_margin'] = sorted_probs[:, -1] - sorted_probs[:, -2]
    labeled_df['regime_entropy'] = -(posteriors * np.log(posteriors + 1e-12)).sum(axis=1)

    for i in range(n_components):
        labeled_df[f'regime_{i}_prob'] = posteriors[:, i]

    other_cols = [c for c in df.columns if c not in FEATURE_COLS]
    labeled_df = labeled_df.join(df[other_cols], how='left')

    return model, labeled_df


def classify_regime(row: pd.Series) -> tuple[str, str]:
    growth = row["growth_yoy_zscore"]
    cpi = row["cpi_yoy_zscore"]
    unemp = row["unemployment_delta_zscore"]
    yspread = row["yield_spread_delta_zscore"]
    cspread = row["credit_spread_delta_zscore"]

    if growth > 0.5 and unemp < 0 and yspread < 0 and cspread < 0:
        return "Growth", "#2CA02C"

    if growth < -0.5 and cpi < 0 and unemp > 0:
        return "Slowdown / Shock", "#4FC3F7"

    if growth < 0 and cpi > 0.5 and (yspread > 0 or cspread > 0):
        return "Inflationary Stress", "#D62728"

    return "Balanced", "#FFD966"


def assign_regime_labels(labeled_df: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, dict, dict]:
    regime_means = labeled_df.groupby('inferred_regime')[feature_cols].mean()

    regime_name_map = {}
    regime_color_map = {}

    for regime_id, row in regime_means.iterrows():
        label, color = classify_regime(row)
        regime_name_map[int(regime_id)] = label
        regime_color_map[int(regime_id)] = color

    labeled_df['regime_label'] = labeled_df['inferred_regime'].map(regime_name_map)

    print("\n[+] Regime Label Mapping:")
    for regime_id, label in regime_name_map.items():
        print(f"Regime {regime_id} → {label}")

    return labeled_df, regime_name_map, regime_color_map


def check_empirical_persistence(labeled_df: pd.DataFrame) -> pd.Series:
    print("\n" + "=" * 60)
    print("EMPIRICAL REGIME PERSISTENCE")
    print("=" * 60)

    s = labeled_df['inferred_regime'].values
    regimes = sorted(pd.unique(s))

    overall = np.mean(s[1:] == s[:-1]) if len(s) > 1 else np.nan
    print(f"\n[+] Overall empirical persistence: {overall:.3f}")

    empirical_persist = {}
    for regime in regimes:
        prev = s[:-1]
        nxt = s[1:]
        denom = np.sum(prev == regime)
        numer = np.sum((prev == regime) & (nxt == regime))
        persist = numer / denom if denom > 0 else 0.0
        empirical_persist[int(regime)] = persist

    print("\n[+] Empirical persistence by regime:")
    for regime, persist in empirical_persist.items():
        print(f"Regime {regime}: {persist:.3f}")

    print("=" * 60)
    return pd.Series(empirical_persist)


def check_regime_quality(labeled_df: pd.DataFrame, model: GaussianHMM, feature_cols: list):
    print("\n" + "=" * 60)
    print("REGIME QUALITY DIAGNOSTICS")
    print("=" * 60)

    regime_counts = labeled_df['inferred_regime'].value_counts().sort_index()
    print("\n[*] Regime Frequency:")
    for regime, count in regime_counts.items():
        pct = 100 * count / len(labeled_df)
        print(f"Regime {regime}: {count:4d} months ({pct:5.1f}%)")

    smallest = regime_counts.min()
    print(f"\nSmallest regime: {smallest} months ({100 * smallest / len(labeled_df):.1f}%)")
    if smallest < 50:
        print("⚠️  WARNING: Regime with < 50 months may be unstable")

    persistence = np.diag(model.transmat_)
    print("\n[*] Model-Implied Regime Persistence (from transition matrix):")
    for i, p in enumerate(persistence):
        status = "✓" if p >= 0.75 else "⚠️" if p >= 0.65 else "❌"
        print(f"Regime {i}: {p:.3f} {status}")

    if any(persistence < 0.70):
        print("\n⚠️  WARNING: Some regimes have persistence < 0.70 (frequent switching)")

    regime_means = labeled_df.groupby('inferred_regime')[feature_cols].mean().values
    n_regimes = len(regime_means)

    print("\n[*] Pairwise Distance Between Regime Means (z-score space):")
    close_pairs = []
    for i in range(n_regimes):
        for j in range(i + 1, n_regimes):
            dist = np.linalg.norm(regime_means[i] - regime_means[j])
            status = "✓" if dist >= 1.5 else "⚠️" if dist >= 1.0 else "❌"
            print(f"Regime {i} ↔ Regime {j}: {dist:.3f} {status}")
            if dist < 1.0:
                close_pairs.append((i, j, dist))

    if close_pairs:
        print(f"\n⚠️  WARNING: {len(close_pairs)} regime pair(s) very close (distance < 1.0):")
        for i, j, d in close_pairs:
            print(f"   Regimes {i} and {j} (distance: {d:.3f}) may be near-duplicates")

    print("\n[*] Regime Uncertainty Metrics:")
    print(f"Mean confidence: {labeled_df['regime_confidence'].mean():.3f}")
    print(f"Mean margin (top 2 gap): {labeled_df['regime_margin'].mean():.3f}")
    print(f"Mean entropy: {labeled_df['regime_entropy'].mean():.3f}")

    low_confidence = (labeled_df['regime_confidence'] < 0.5).sum()
    if low_confidence > 0:
        pct = 100 * low_confidence / len(labeled_df)
        print(f"\n⚠️  WARNING: {low_confidence} months ({pct:.1f}%) have confidence < 0.5")

    print("=" * 60)


def output_model_diagnostics(model: GaussianHMM, labeled_df: pd.DataFrame, feature_cols: list):
    n_components = model.n_components
    print("\n" + "=" * 60)
    print("             HMM REGIME MODEL DIAGNOSTICS             ")
    print("=" * 60)

    print(f"\n[+] Observations per Regime (n_components={n_components}):")
    regime_counts = labeled_df['inferred_regime'].value_counts().sort_index()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(labeled_df)
        label = labeled_df[labeled_df['inferred_regime'] == regime]['regime_label'].iloc[0]
        print(f"Regime {regime} ({label}): {count:4d} months ({pct:5.1f}%)")

    print("\n[+] Transition Matrix (probability of moving from row regime to column regime):")
    transition_matrix = pd.DataFrame(
        model.transmat_,
        index=[f"Regime_{i}" for i in range(n_components)],
        columns=[f"To_{i}" for i in range(n_components)]
    )
    print(transition_matrix.round(3))

    print("\n[+] Z-Scored Feature Means per Regime (use this to interpret regimes):")
    regime_feature_means = labeled_df.groupby('inferred_regime')[feature_cols].mean()
    print(regime_feature_means.round(3))

    raw_feature_cols = ['growth_yoy', 'cpi_yoy', 'unemployment_delta', 'yield_spread_delta', 'credit_spread_delta']
    available_raw_cols = [c for c in raw_feature_cols if c in labeled_df.columns]
    if available_raw_cols:
        print("\n[+] Raw Macro Variable Means per Regime:")
        print(labeled_df.groupby('inferred_regime')[available_raw_cols].mean().round(4))

    print("\n[+] Equity Returns per Regime:")
    returns_profile = labeled_df.groupby('inferred_regime')[['equity_return']].agg(['mean', 'std', 'count'])
    returns_profile.columns = ['mean', 'std', 'count']
    print(returns_profile.round(5))

    print("=" * 60)


def save_model_artifacts(model: GaussianHMM, labeled_df: pd.DataFrame, regime_name_map: dict, regime_color_map: dict):
    os.makedirs("data/models", exist_ok=True)

    model_path = f"data/models/hmm_{model.n_components}state.pkl"
    joblib.dump(model, model_path)
    print(f"[+] Fitted HMM saved to: {model_path}")

    label_map_path = "data/models/regime_labels.json"
    with open(label_map_path, "w") as f:
        json.dump({
            "regime_names": {int(k): v for k, v in regime_name_map.items()},
            "regime_colors": {int(k): v for k, v in regime_color_map.items()}
        }, f, indent=2)
    print(f"[+] Regime label mappings saved to: {label_map_path}")

    regime_stats = {}
    for regime in sorted(labeled_df['inferred_regime'].unique()):
        regime_data = labeled_df[labeled_df['inferred_regime'] == regime]

        regime_series = labeled_df['inferred_regime']
        run_id = (regime_series != regime_series.shift()).cumsum()
        durations = regime_series[regime_series == regime].groupby(run_id).size()

        regime_stats[int(regime)] = {
            "label": regime_name_map[int(regime)],
            "count": int(len(regime_data)),
            "frequency": float(len(regime_data) / len(labeled_df)),

            "equity_mean": float(regime_data['equity_return'].mean()),
            "equity_std": float(regime_data['equity_return'].std()),
            "equity_skew": float(regime_data['equity_return'].skew()),
            "equity_kurt": float(regime_data['equity_return'].kurt()),

            # NOTE: CPI YoY stored here; convert to monthly later in Monte Carlo
            "inflation_mean": float(regime_data['cpi_yoy'].mean()) if 'cpi_yoy' in regime_data else None,
            "inflation_std": float(regime_data['cpi_yoy'].std()) if 'cpi_yoy' in regime_data else None,

            "duration_mean": float(durations.mean()) if len(durations) else None,
            "duration_median": float(durations.median()) if len(durations) else None,
            "duration_std": float(durations.std()) if len(durations) else None,

            "feature_profile": {
                col: float(regime_data[col].mean())
                for col in FEATURE_COLS if col in regime_data
            }
        }

    stats_path = "data/models/regime_market_assumptions.json"
    with open(stats_path, "w") as f:
        json.dump(regime_stats, f, indent=2)
    print(f"[+] Regime market assumptions saved to: {stats_path}")

    transition_path = "data/models/transition_matrix.csv"
    pd.DataFrame(
        model.transmat_,
        index=[f"Regime_{i}" for i in range(model.n_components)],
        columns=[f"To_{i}" for i in range(model.n_components)]
    ).to_csv(transition_path)
    print(f"[+] Transition matrix saved to: {transition_path}")


if __name__ == "__main__":
    DATA_INPUT = "data/processed/aligned_macro_dataset.csv"
    DATA_OUTPUT = "data/processed/regime_labeled_dataset.csv"

    try:
        master_df = load_processed_data(DATA_INPUT)

        print("\n" + "=" * 60)
        print("TRAINING HMM")
        print("=" * 60)
        hmm_model, output_df = train_regime_hmm(master_df, n_components=None)

        print("\n" + "=" * 60)
        print("REGIME INTERPRETATION")
        print("=" * 60)
        output_df, regime_names, regime_colors = assign_regime_labels(output_df, FEATURE_COLS)

        print("\n" + "=" * 60)
        print("MODEL VALIDATION")
        print("=" * 60)
        output_model_diagnostics(hmm_model, output_df, FEATURE_COLS)
        check_regime_quality(output_df, hmm_model, FEATURE_COLS)
        check_empirical_persistence(output_df)

        print("\n" + "=" * 60)
        print("PERSISTING MODEL ARTIFACTS")
        print("=" * 60)
        output_df.to_csv(DATA_OUTPUT)
        print(f"[+] Regime-labeled dataset written to: {DATA_OUTPUT}")

        save_model_artifacts(hmm_model, output_df, regime_names, regime_colors)

    except Exception as e:
        print(f"[-] Training Engine Failure: {str(e)}")
        raise