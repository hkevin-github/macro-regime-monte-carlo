import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from regime_utils import classify_regime

FEATURE_COLS = [
    "growth_yoy_zscore",
    "cpi_yoy_zscore",
    "unemployment_delta_zscore",
    "yield_spread_delta_zscore",
    "credit_spread_delta_zscore",
]

def generate_regime_map(data_path="data/processed/regime_labeled_dataset.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError("Missing labeled dataset. Run scripts sequentially.")

    df = pd.read_csv(data_path, index_col="date", parse_dates=True)
    df["sp500_indexed"] = (1 + df["equity_return"]).cumprod() * 100

    regime_means = df.groupby("inferred_regime")[FEATURE_COLS].mean()

    regime_names = {}
    regime_colors = {}

    for regime_idx, row in regime_means.iterrows():
        label, color = classify_regime(row)
        regime_names[regime_idx] = label
        regime_colors[regime_idx] = color

    present_regimes = sorted(df["inferred_regime"].unique())
    colors = {r: regime_colors[r] for r in present_regimes}
    labels = {r: regime_names[r] for r in present_regimes}

    # Order legend by average return, but don't show regime numbers
    regime_avg_return = df.groupby("inferred_regime")["equity_return"].mean().sort_values()
    sorted_regimes = [r for r in regime_avg_return.index.tolist() if r in present_regimes]

    plt.figure(figsize=(15, 7))

    print("[*] Generating historical regime background highlights...")
    current_regime = df["inferred_regime"].iloc[0]
    start_idx = 0

    for i in range(1, len(df)):
        if df["inferred_regime"].iloc[i] != current_regime:
            plt.axvspan(
                df.index[start_idx],
                df.index[i - 1],
                color=colors[current_regime],
                alpha=0.2
            )
            current_regime = df["inferred_regime"].iloc[i]
            start_idx = i

    plt.axvspan(df.index[start_idx], df.index[-1], color=colors[current_regime], alpha=0.2)

    plt.plot(
        df.index,
        df["sp500_indexed"],
        color="#1F77B4",
        lw=1.5,
        label="S&P 500 Index (Normalized)"
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[r], alpha=0.3, label=labels[r])
        for r in sorted_regimes
    ]
    legend_elements.append(
        plt.Line2D([0], [0], color="#1F77B4", lw=1.5, label="S&P 500")
    )

    latest_date = df.index[-1]
    latest_regime = df["inferred_regime"].iloc[-1]
    plt.axvline(latest_date, color="black", linestyle="--", alpha=0.8)
    plt.scatter(latest_date, df["sp500_indexed"].iloc[-1], color="black", s=50, zorder=5)

    plt.title(
        f"Historical Macro Regime Map (1953 - Present) | Active Today: {labels[latest_regime]}",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel("Timeline Year", fontsize=11)
    plt.ylabel("Indexed Performance Baseline", fontsize=11)
    plt.yscale("log")
    plt.legend(handles=legend_elements, loc="upper left")
    plt.grid(True, alpha=0.2)

    output_img = "notebook/macro_regime_map.png"
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, bbox_inches="tight", dpi=300)
    print(f"[+] Success: Regime visual visualization saved to: {output_img}")


if __name__ == "__main__":
    generate_regime_map()