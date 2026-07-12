"""
Regime Map Visualization Engine
Generates an institutional line plot of the historical S&P 500 index
color-coded by inferred HMM macro regimes.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_regime_map(data_path="data/processed/regime_labeled_dataset_v1.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing labeled dataset. Run scripts sequentially.")

    # Read regime-labeled data
    df = pd.read_csv(data_path, index_col='date', parse_dates=True)

    # Reconstruct index level values from returns for clean visualization
    # If standard closing price isn't saved, use cumulative product proxy
    df['sp500_indexed'] = (1 + df['equity_return']).cumprod() * 100

    plt.figure(figsize=(15, 7))

    # IMPORTANT: HMM state indices are assigned arbitrarily during training --
    # a re-fit (different random_state, different data window, etc.) can and
    # will reorder which integer index maps to which regime. The names below
    # are NOT structural -- they're pinned to the *specific* fitted model from
    # the run analyzed in this conversation, confirmed via z-scored feature
    # means (growth, CPI, unemployment delta, yield spread delta, credit
    # spread delta) plus return/vol profiles:
    #
    #   Regime 0 (n=400): growth ~flat/slightly negative, CPI mild negative,
    #     everything else near zero. Return 0.96%/mo, lowest vol (3.67%).
    #     -> "Moderate Growth / Late-Cycle" (mundane, most common backdrop)
    #   Regime 1 (n=48): growth deeply negative (-1.32z) AND CPI deeply
    #     negative (-1.02z) together -- the only regime where both collapse
    #     at once. Highest return (1.71%/mo), elevated vol (4.62%).
    #     -> "Disinflationary Slowdown / Recovery-Anticipation"
    #   Regime 2 (n=174): growth negative (-0.93z), CPI highest of all four
    #     (+1.18z), credit spreads widest (+0.27z). Worst return (0.29%/mo),
    #     highest vol (5.49%).
    #     -> "Stagflation / Credit Stress"
    #   Regime 3 (n=243): growth strongly positive (+1.04z), mild disinflation,
    #     unemployment falling, spreads tightening. Return 0.52%/mo, vol 3.98%.
    #     -> "Expansion"
    #
    # If this pipeline is re-run and the fitted regimes shift (different index
    # assignments, or a genuinely different cluster structure), re-derive these
    # names from output_model_diagnostics() in hmm_regime_engine.py before
    # trusting this map -- do not assume index->name stays fixed.
    REGIME_NAMES = {
        0: 'Moderate Growth / Late-Cycle',
        1: 'Disinflationary Slowdown',
        2: 'Stagflation / Credit Stress',
        3: 'Expansion',
    }
    REGIME_COLORS = {
        0: '#FFD966',  # muted yellow -- unremarkable/mundane backdrop
        1: '#4FC3F7',  # light blue -- cooling, disinflationary, recovery-adjacent
        2: '#D62728',  # red -- the genuinely bad regime
        3: '#2CA02C',  # green -- expansion
    }

    present_regimes = sorted(df['inferred_regime'].unique())
    missing_names = [r for r in present_regimes if r not in REGIME_NAMES]
    if missing_names:
        # Data contains regime indices this map doesn't recognize (e.g. the
        # model was refit with a different n_components or state ordering
        # changed) -- fail loudly rather than silently mislabeling.
        raise ValueError(
            f"REGIME_NAMES/REGIME_COLORS don't cover regime indices {missing_names} "
            f"found in the data. Re-run output_model_diagnostics() in "
            f"hmm_regime_engine.py and update the mapping above before plotting."
        )

    colors = {r: REGIME_COLORS[r] for r in present_regimes}
    labels = {r: f'Regime {r}: {REGIME_NAMES[r]}' for r in present_regimes}

    # Legend/plot order: rank by mean equity return (worst -> best) purely for
    # a sensible visual/legend ordering, not for deriving the names themselves.
    regime_avg_return = df.groupby('inferred_regime')['equity_return'].mean().sort_values()
    sorted_regimes = [r for r in regime_avg_return.index.tolist() if r in present_regimes]

    # Step through data to paint background spans according to active state
    print("[*] Generating historical regime background highlights...")
    current_regime = df['inferred_regime'].iloc[0]
    start_idx = 0

    for i in range(1, len(df)):
        if df['inferred_regime'].iloc[i] != current_regime:
            # Regime changed - paint the previous regime's span
            plt.axvspan(df.index[start_idx], df.index[i-1], color=colors[current_regime], alpha=0.2)
            current_regime = df['inferred_regime'].iloc[i]
            start_idx = i

    # Paint the final regime's span after loop ends
    plt.axvspan(df.index[start_idx], df.index[-1], color=colors[current_regime], alpha=0.2)

    # Plot standard S&P 500 trajectory line on top
    plt.plot(df.index, df['sp500_indexed'], color='#1F77B4', lw=1.5, label='S&P 500 Index (Normalized)')

    # Construct legend handles for however many regimes are present, in the
    # same rank order (worst return -> best return) as the colors/labels above.
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[regime_idx], alpha=0.3, label=labels[regime_idx])
        for regime_idx in sorted_regimes
    ]
    legend_elements.append(plt.Line2D([0], [0], color='#1F77B4', lw=1.5, label='S&P 500'))

    # Highlight "Today" (the final data row)
    latest_date = df.index[-1]
    latest_regime = df['inferred_regime'].iloc[-1]
    plt.axvline(latest_date, color='black', linestyle='--', alpha=0.8)
    plt.scatter(latest_date, df['sp500_indexed'].iloc[-1], color='black', s=50, zorder=5)

    plt.title(f"Historical Macro Regime Map (1953 - Present) | Active Today: {labels[latest_regime]}", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline Year", fontsize=11)
    plt.ylabel("Indexed Performance Baseline", fontsize=11)
    plt.yscale('log') # Use log scale to track 70 years cleanly
    plt.legend(handles=legend_elements, loc='upper left')
    plt.grid(True, alpha=0.2)

    output_img = "notebook/macro_regime_map.png"
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, bbox_inches='tight', dpi=300)
    print(f"[+] Success: Regime visual visualization saved to: {output_img}")

if __name__ == "__main__":
    generate_regime_map()