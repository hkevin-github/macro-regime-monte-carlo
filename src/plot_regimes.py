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
        
    # Read Yahoo Finance Index Data and state labels
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    # Reconstruct index level values from returns for clean visualization
    # If standard closing price isn't saved, use cumulative product proxy
    df['sp500_indexed'] = (1 + df['spy_equivalent_return']).cumprod() * 100
    
    plt.figure(figsize=(15, 7))
    
    # Define mapping color canvas
    # 0 = Stagflation (Orange), 1 = Expansion (Green), 2 = Shock/Recession (Red)
    colors = {0: '#FFA500', 1: '#2CA02C', 2: '#D62728'}
    labels = {0: 'Regime 0: High Inflation', 1: 'Regime 1: Expansion', 2: 'Regime 2: Shock/Recession'}
    
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
    
    # Construct beautiful custom legend handles manually to prevent duplicates
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.3, label=labels[0]),
        Patch(facecolor=colors[1], alpha=0.3, label=labels[1]),
        Patch(facecolor=colors[2], alpha=0.3, label=labels[2]),
        plt.Line2D([0], [0], color='#1F77B4', lw=1.5, label='S&P 500')
    ]
    
    # Highlight "Today" (the final data row)
    latest_date = df.index[-1]
    latest_regime = df['inferred_regime'].iloc[-1]
    plt.axvline(latest_date, color='black', linestyle='--', alpha=0.8)
    plt.scatter(latest_date, df['sp500_indexed'].iloc[-1], color='black', s=50, zorder=5)
    
    plt.title(f"Historical Macro Regime Map (1953 - Present) | Active Today: State {latest_regime}", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline Year", fontsize=11)
    plt.ylabel("Indexed Performance Baseline", fontsize=11)
    plt.yscale('log') # Use log scale to track 70 years cleanly
    plt.legend(handles=legend_elements, loc='upper left')
    plt.grid(True, alpha=0.2)
    
    output_img = "notebook/macro_regime_map.png"
    plt.savefig(output_img, bbox_inches='tight', dpi=300)
    print(f"[+] Success: Regime visual visualization saved to: {output_img}")

if __name__ == "__main__":
    generate_regime_map()