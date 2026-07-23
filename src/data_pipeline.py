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
Bond Returns      -> AGG              -> Yahoo Finance  -> Monthly -> 2003 - Present

All series are aligned on a monthly calendar (Period[M]) so that macro releases,
equity returns, and Fama-French factors line up on the same monthly timeline.
"""

import os
import re
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
import pandas_datareader.data as web


def load_env_file(env_path: str | None = None) -> dict[str, str]:
    """Load key/value pairs from a .env file into a dictionary."""
    path = Path(env_path or Path(__file__).resolve().parents[1] / ".env")
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = normalize_api_key(value)
    return values


def normalize_api_key(value: str | None) -> str:
    """Trim whitespace and surrounding quotes from environment values."""
    if value is None:
        return ""
    cleaned = value.strip()
    cleaned = cleaned.strip("\"'")
    return cleaned


def fetch_yahoo_returns(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches the historical S&P 500 Index from Yahoo Finance and calculates
    MONTHLY returns (Equity Returns concept: ^GSPC, Yahoo Finance, Monthly).
    Includes fallback parsing to handle yfinance MultiIndex column structures cleanly.
    """
    print(f"[*] Ingesting S&P 500 index (^GSPC) from Yahoo Finance...")

    df = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if 'Adj Close' in df.columns:
        adj_close = df['Adj Close']
    elif 'Close' in df.columns:
        print("[!] 'Adj Close' not found. Falling back to 'Close' column.")
        adj_close = df['Close']
    else:
        raise KeyError(f"CRITICAL: Structural match failure. Available columns: {list(df.columns)}")

    monthly_close = adj_close.resample('ME').last()
    returns = monthly_close.pct_change().dropna()

    returns_df = pd.DataFrame(returns)
    returns_df.columns = ['equity_return']

    returns_df.index = returns_df.index.to_period('M')
    return returns_df


def fetch_bond_returns(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical bond returns using AGG (Aggregate Bond ETF) as proxy.
    AGG inception: September 2003
    For periods before AGG, returns empty DataFrame (will be handled downstream).
    """
    print(f"[*] Ingesting bond returns (AGG ETF)...")
    
    try:
        agg = yf.download("AGG", start=start_date, end=end_date, auto_adjust=False, progress=False)
        
        if agg.empty:
            print(f"[!] AGG data is empty. Bond data will not be available.")
            return pd.DataFrame()
        
        if isinstance(agg.columns, pd.MultiIndex):
            agg.columns = agg.columns.get_level_values(0)
        
        if 'Adj Close' in agg.columns:
            adj_close = agg['Adj Close']
        elif 'Close' in agg.columns:
            print("[!] 'Adj Close' not found in AGG. Falling back to 'Close' column.")
            adj_close = agg['Close']
        else:
            raise KeyError(f"No price column found in AGG data. Available: {list(agg.columns)}")
        
        monthly_close = adj_close.resample('ME').last()
        returns = monthly_close.pct_change().dropna()
        
        returns_df = pd.DataFrame(returns)
        returns_df.columns = ['bond_return']
        returns_df.index = returns_df.index.to_period('M')
        
        print(f"[+] Fetched AGG bond returns: {len(returns_df)} months")
        return returns_df
        
    except Exception as e:
        print(f"[!] AGG fetch failed: {e}")
        print(f"[!] Bond data will not be available. Simulator will use proxy model.")
        return pd.DataFrame()


def fetch_fama_french_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Queries Ken French's Data Library for MONTHLY research factors (Mkt-RF, SMB, HML, RF)
    so they align with the rest of the monthly matrix.
    """
    print(f"[*] Ingesting Fama-French 3-Factor Monthly Dataset...")
    ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start_date, end=end_date)

    df = ff_data[0] / 100.0

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
    normalized_key = normalize_api_key(api_key)
    if not normalized_key or normalized_key == "YOUR_FALLBACK_KEY_HERE":
        raise ValueError("CRITICAL: FRED API Key is missing. Check setup configurations.")

    if len(normalized_key) != 32 or not re.fullmatch(r"[a-z0-9]+", normalized_key):
        raise ValueError(
            "CRITICAL: FRED API Key is invalid. Provide a 32-character lowercase alphanumeric key."
        )

    fred = Fred(api_key=normalized_key)

    series_map = {
        'indpro': 'INDPRO',
        'cpi': 'CPIAUCSL',
        'unemployment': 'UNRATE',
        'gs10': 'GS10',
        'tb3ms': 'TB3MS',
        'baa': 'BAA',
    }

    macro_series = {}
    for key, series_id in series_map.items():
        print(f"[*] Pulling FRED Series: {series_id} ({key})...")
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        macro_series[key] = series

    df = pd.DataFrame(macro_series)
    df.index = pd.to_datetime(df.index)

    df['yield_spread'] = df['gs10'] - df['tb3ms']
    df['credit_spread'] = df['baa'] - df['gs10']

    df = df.drop(columns=['gs10', 'tb3ms', 'baa'])

    df.index = df.index.to_period('M')

    return df


def transform_and_align_pipeline(asset_df: pd.DataFrame, 
                                 ff_df: pd.DataFrame, 
                                 macro_df: pd.DataFrame,
                                 bond_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Transforms raw macro factors for stationarity using smoothed rolling windows
    and CAUSAL expanding-window z-score standardization to eliminate look-ahead bias.
    
    Args:
        asset_df: Equity returns
        ff_df: Fama-French factors
        macro_df: Macro indicators
        bond_df: Bond returns (optional)
    
    Returns:
        Aligned dataset with causal z-scores
    """
    print("[*] Running Smoothed Stationarity Transformations and Causal Standardization...")

    macro_stationed = pd.DataFrame(index=macro_df.index)

    # 1. Transform Macro Features (backward-looking only)
    macro_stationed['growth_yoy'] = macro_df['indpro'].pct_change(12)
    macro_stationed['cpi_yoy'] = macro_df['cpi'].pct_change(12)
    macro_stationed['unemployment_delta'] = macro_df['unemployment'].diff().rolling(window=3, min_periods=1).mean()
    macro_stationed['yield_spread_delta'] = macro_df['yield_spread'].diff().rolling(window=3, min_periods=1).mean()
    macro_stationed['credit_spread_delta'] = macro_df['credit_spread'].diff().rolling(window=3, min_periods=1).mean()

    macro_stationed = macro_stationed.dropna()

    # 2. Concatenate all datasets on shared monthly Period[M] index
    dataframes_to_concat = [asset_df, ff_df, macro_stationed]
    
    # Add bonds if available
    if bond_df is not None and not bond_df.empty:
        dataframes_to_concat.append(bond_df)
        print("[+] Including bond returns in dataset")
    else:
        print("[!] No bond data available - simulator will use proxy model")
    
    master_matrix = pd.concat(dataframes_to_concat, axis=1).dropna()

    # 3. CAUSAL Feature Standardization via Expanding-Window Z-Score
    # This eliminates look-ahead bias by using only data available up to each point in time
    macro_cols = ['growth_yoy', 'cpi_yoy', 'unemployment_delta', 'yield_spread_delta', 'credit_spread_delta']
    
    print("[*] Applying causal expanding-window z-score standardization...")
    for col in macro_cols:
        expanding_mean = master_matrix[col].expanding(min_periods=24).mean()
        expanding_std = master_matrix[col].expanding(min_periods=24).std(ddof=0)
        expanding_std = expanding_std.replace(0, np.nan)
        master_matrix[f'{col}_zscore'] = (master_matrix[col] - expanding_mean) / expanding_std

    # Drop rows where z-scores couldn't be computed
    master_matrix = master_matrix.dropna(subset=[f'{c}_zscore' for c in macro_cols])

    # 4. Convert Period[M] index back to timestamps for downstream compatibility
    master_matrix.index = master_matrix.index.to_timestamp(how='end').normalize()
    master_matrix.index.name = 'date'

    print(f"[+] Causal transformation complete. Shape: {master_matrix.shape}")
    return master_matrix


def run_data_pipeline(fred_key: str, start: str = "1953-04-01", end: str = "2026-01-01") -> pd.DataFrame:
    """
    Orchestration master function for execution inside notebooks or main application layers.
    """
    equity_returns = fetch_yahoo_returns(start, end)
    ff_factors = fetch_fama_french_factors(start, end)
    fred_macro = fetch_fred_macro(fred_key, start, end)
    bond_returns = fetch_bond_returns(start, end)

    aligned_dataset = transform_and_align_pipeline(equity_returns, ff_factors, fred_macro, bond_returns)
    print(f"[+] Pipeline complete. Generated shape matrix: {aligned_dataset.shape}")
    return aligned_dataset


if __name__ == "__main__":
    env_values = load_env_file()
    API_KEY = normalize_api_key(os.getenv("FRED_API_KEY") or env_values.get("FRED_API_KEY"))
    if not API_KEY:
        API_KEY = "YOUR_ACTUAL_FRED_API_KEY_HERE"

    try:
        sample_df = run_data_pipeline(fred_key=API_KEY, start="1953-04-01", end="2026-07-01")

        output_path = "data/processed/aligned_macro_dataset.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sample_df.to_csv(output_path)
        print(f"[+] Success: Aligned dataset written to: {output_path}")

        print("\n--- SAMPLE VIEW OF ALIGNED PIPELINE ---")
        cols_to_show = ['equity_return', 'Mkt-RF', 'growth_yoy_zscore', 'cpi_yoy_zscore', 
                       'credit_spread_delta_zscore', 'yield_spread_delta_zscore']
        if 'bond_return' in sample_df.columns:
            cols_to_show.insert(2, 'bond_return')
        print(sample_df[cols_to_show].head())
    except Exception as e:
        print(f"[-] Execution Pipeline Failure: {str(e)}")