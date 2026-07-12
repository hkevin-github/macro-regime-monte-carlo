"""
Data Ingestion & Alignment Pipeline
Targets long-horizon data from 1953 onwards using Yahoo Finance, FRED,
and the Ken French (Fama-French) Data Library via pandas_datareader.

Economic Concept -> Ticker(s) Used -> Source -> Frequency -> Horizon
--------------------------------------------------------------------
Growth            -> INDPRO           -> FRED           -> Monthly -> 1953 - Present
Inflation         -> CPIAUCSL         -> FRED           -> Monthly -> 1953 - Present
Yield Curve       -> GS10 - TB3MS     -> FRED           -> Monthly -> 1953 - Present
Credit Risk       -> BAA - GS10       -> FRED           -> Monthly -> 1953 - Present
Labor             -> UNRATE           -> FRED           -> Monthly -> 1953 - Present
Equity Returns    -> ^GSPC            -> Yahoo Finance  -> Monthly -> 1953 - Present

All series are aligned on a monthly calendar (Period[M]) so that macro releases,
equity returns, and Fama-French factors line up on the same monthly timeline.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
import pandas_datareader.data as web


def fetch_yahoo_returns(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches the historical S&P 500 Index from Yahoo Finance and calculates
    MONTHLY returns (Equity Returns concept: ^GSPC, Yahoo Finance, Monthly).
    Includes fallback parsing to handle yfinance MultiIndex column structures cleanly.
    """
    print(f"[*] Ingesting S&P 500 index (^GSPC) from Yahoo Finance...")

    # Explicitly set auto_adjust=False to keep standard columns if possible
    df = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=False, progress=False)

    # Flatten MultiIndex columns if yfinance returns them nested under the ticker name
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Robust Fallback Check: Look for 'Adj Close', fall back to 'Close' if missing
    if 'Adj Close' in df.columns:
        adj_close = df['Adj Close']
    elif 'Close' in df.columns:
        print("[!] 'Adj Close' not found. Falling back to 'Close' column.")
        adj_close = df['Close']
    else:
        raise KeyError(f"CRITICAL: Structural match failure. Available columns: {list(df.columns)}")

    # Resample daily closes down to month-end observations, then compute monthly returns
    monthly_close = adj_close.resample('ME').last()
    returns = monthly_close.pct_change().dropna()

    returns_df = pd.DataFrame(returns)
    returns_df.columns = ['equity_return']

    # Canonicalize to a Period[M] index so it aligns cleanly with FRED/Fama-French data
    returns_df.index = returns_df.index.to_period('M')
    return returns_df


def fetch_fama_french_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Queries Ken French's Data Library for MONTHLY research factors (Mkt-RF, SMB, HML, RF)
    so they align with the rest of the monthly matrix.
    """
    print(f"[*] Ingesting Fama-French 3-Factor Monthly Dataset...")
    ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start_date, end=end_date)

    # Table [0] contains the monthly return percentages
    df = ff_data[0] / 100.0  # Convert percentages to actual decimals

    # Ken French monthly tables come back as a PeriodIndex('M') already; normalize just in case
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.asfreq('M')
    else:
        df.index = pd.to_datetime(df.index).to_period('M')

    return df


def fetch_fred_macro(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Queries FRED API for the core monthly macro indicators:
      - Growth:      INDPRO
      - Inflation:   CPIAUCSL
      - Labor:       UNRATE
      - Yield Curve: GS10 - TB3MS
      - Credit Risk: BAA - GS10
    """
    if not api_key or api_key == "YOUR_FALLBACK_KEY_HERE":
        raise ValueError("CRITICAL: FRED API Key is missing. Check setup configurations.")

    fred = Fred(api_key=api_key)

    # Query targets, all Monthly frequency, 1953 - Present coverage
    series_map = {
        'indpro': 'INDPRO',        # Growth: Industrial Production Index
        'cpi': 'CPIAUCSL',         # Inflation: CPI, All Urban Consumers
        'unemployment': 'UNRATE',  # Labor: Civilian Unemployment Rate
        'gs10': 'GS10',            # 10-Year Treasury Constant Maturity rate
        'tb3ms': 'TB3MS',          # 3-Month Treasury Bill rate
        'baa': 'BAA',              # Moody's Seasoned Baa Corporate Bond Yield
    }

    macro_series = {}
    for key, series_id in series_map.items():
        print(f"[*] Pulling FRED Series: {series_id} ({key})...")
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        macro_series[key] = series

    df = pd.DataFrame(macro_series)
    df.index = pd.to_datetime(df.index)

    # Structural Shift Feature Engineering:
    # Yield Curve: synthetic 10Y-3M spread
    df['yield_spread'] = df['gs10'] - df['tb3ms']
    # Credit Risk: synthetic Baa-over-10Y spread (credit/default risk premium)
    df['credit_spread'] = df['baa'] - df['gs10']

    df = df.drop(columns=['gs10', 'tb3ms', 'baa'])

    # Canonicalize to a Period[M] index so it aligns cleanly with equity/FF data
    df.index = df.index.to_period('M')

    return df


def transform_and_align_pipeline(asset_df: pd.DataFrame, ff_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw macro factors for stationarity using smoothed rolling windows
    to enforce time-series momentum, eliminating HMM regime-chattering, then
    aligns everything on a shared monthly (Period[M]) timeline.
    """
    print("[*] Running Smoothed Stationarity Transformations and Alignments...")

    macro_stationed = pd.DataFrame(index=macro_df.index)

    # 1. Transform Macro Features with Smoothing Windows
    # Growth: Industrial Production, YoY % Change
    macro_stationed['growth_yoy'] = macro_df['indpro'].pct_change(12)

    # Inflation: CPI, YoY % Change (already relatively smooth)
    macro_stationed['cpi_yoy'] = macro_df['cpi'].pct_change(12)

    # Labor: 3-month rolling average of the monthly changes in unemployment
    macro_stationed['unemployment_delta'] = macro_df['unemployment'].diff().rolling(window=3, min_periods=1).mean()

    # Yield Curve: 3-month rolling average of the monthly changes in the 10Y-3M spread
    macro_stationed['yield_spread_delta'] = macro_df['yield_spread'].diff().rolling(window=3, min_periods=1).mean()

    # Credit Risk: 3-month rolling average of the monthly changes in the Baa-10Y spread
    macro_stationed['credit_spread_delta'] = macro_df['credit_spread'].diff().rolling(window=3, min_periods=1).mean()

    macro_stationed = macro_stationed.dropna()

    # 2. Concatenate all datasets (equity returns, Fama-French, macro) on their shared
    #    monthly Period[M] index -- no daily forward-fill required anymore since every
    #    source is already monthly.
    master_matrix = pd.concat([asset_df, ff_df, macro_stationed], axis=1).dropna()

    # 3. Feature Standardization via Z-Score
    macro_cols = ['growth_yoy', 'cpi_yoy', 'unemployment_delta', 'yield_spread_delta', 'credit_spread_delta']
    for col in macro_cols:
        mean = master_matrix[col].mean()
        std = master_matrix[col].std()
        master_matrix[f'{col}_zscore'] = (master_matrix[col] - mean) / std

    # 4. Convert the Period[M] index back to month-end Timestamps for readability downstream
    master_matrix.index = master_matrix.index.to_timestamp(how='end').normalize()
    master_matrix.index.name = 'date'

    return master_matrix


def run_data_pipeline(fred_key: str, start: str = "1953-04-01", end: str = "2026-01-01") -> pd.DataFrame:
    """
    Orchestration master function for execution inside notebooks or main application layers.
    """
    # Pull data chunks
    equity_returns = fetch_yahoo_returns(start, end)
    ff_factors = fetch_fama_french_factors(start, end)
    fred_macro = fetch_fred_macro(fred_key, start, end)

    # Align and mutate matrices
    aligned_dataset = transform_and_align_pipeline(equity_returns, ff_factors, fred_macro)
    print(f"[+] Pipeline complete. Generated shape matrix: {aligned_dataset.shape}")
    return aligned_dataset


if __name__ == "__main__":
    # Retrieve key from environment variable or update string fallback right here:
    API_KEY = os.getenv("FRED_API_KEY", "YOUR_ACTUAL_FRED_API_KEY_HERE")

    try:
        sample_df = run_data_pipeline(fred_key=API_KEY, start="1953-04-01", end="2026-07-01")

        # Persist to the path the downstream HMM engine expects.
        output_path = "data/processed/aligned_macro_dataset_v1.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sample_df.to_csv(output_path)
        print(f"[+] Success: Aligned dataset written to: {output_path}")

        print("\n--- SAMPLE VIEW OF ALIGNED PIPELINE ---")
        print(sample_df[[
            'equity_return',
            'Mkt-RF',
            'growth_yoy_zscore',
            'cpi_yoy_zscore',
            'credit_spread_delta_zscore',
            'yield_spread_delta_zscore',
        ]].head())
    except Exception as e:
        print(f"[-] Execution Pipeline Failure: {str(e)}")