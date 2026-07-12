"""
Data Ingestion & Alignment Pipeline
Targets long-horizon data from 1953 onwards using Yahoo Finance, FRED, 
and the Ken French (Fama-French) Data Library via pandas_datareader.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
import pandas_datareader.data as web


def fetch_yahoo_returns(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches the historical S&P 500 Index from Yahoo Finance and calculates daily returns.
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
    
    # Calculate daily returns, drop first row NaN
    returns = adj_close.pct_change().dropna()
    
    returns_df = pd.DataFrame(returns)
    returns_df.columns = ['spy_equivalent_return']
    return returns_df


def fetch_fama_french_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Queries Ken French's Data Library for daily research factors (Mkt-RF, SMB, HML, RF).
    Converts PeriodIndex to DatetimeIndex to prevent pipeline alignment crashes.
    """
    print(f"[*] Ingesting Fama-French 3-Factor Daily Dataset...")
    ff_data = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=start_date, end=end_date)
    
    # Table [0] contains the daily return percentages
    df = ff_data[0] / 100.0  # Convert percentages to actual decimals
    
    # Explicitly convert PeriodIndex to standard DatetimeIndex timestamps
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    else:
        df.index = pd.to_datetime(df.index)
        
    return df


def fetch_fred_macro(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Queries FRED API for core macro indicators and derives a long-history yield curve spread.
    """
    if not api_key or api_key == "YOUR_FALLBACK_KEY_HERE":
        raise ValueError("CRITICAL: FRED API Key is missing. Check setup configurations.")
        
    fred = Fred(api_key=api_key)
    
    # Query targets: CPI (Monthly), Unemployment (Monthly), 10Y Yield (Monthly), 3M Yield (Monthly)
    series_map = {
        'cpi': 'CPIAUCSL',
        'unemployment': 'UNRATE',
        'gs10': 'GS10',   # 10-Year Treasury Constant Maturity rate
        'tb3ms': 'TB3MS'  # 3-Month Treasury Bill rate
    }
    
    macro_series = {}
    for key, series_id in series_map.items():
        print(f"[*] Pulling FRED Series: {series_id} ({key})...")
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        macro_series[key] = series
        
    df = pd.DataFrame(macro_series)
    df.index = pd.to_datetime(df.index)
    
    # Structural Shift Feature Engineering: Construct synthetic 10Y-3M yield curve spread
    df['yield_spread'] = df['gs10'] - df['tb3ms']
    df = df.drop(columns=['gs10', 'tb3ms'])
    
    return df


def transform_and_align_pipeline(asset_df: pd.DataFrame, ff_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw macro factors for stationarity, resolves mixed frequency calendars, 
    and aligns all indicators to daily asset-trading dates.
    """
    print("[*] Running Stationarity Transformations and Alignments...")
    
    # 1. Transform Macro Features to enforce Stationarity
    macro_stationed = pd.DataFrame(index=macro_df.index)
    
    # CPI Raw to YoY % Change
    macro_stationed['cpi_yoy'] = macro_df['cpi'].pct_change(12)
    # Unemployment to Month-over-Month Delta changes
    macro_stationed['unemployment_delta'] = macro_df['unemployment'].diff()
    # Yield Spread to month-over-month rate changes
    macro_stationed['yield_spread_delta'] = macro_df['yield_spread'].diff()
    
    macro_stationed = macro_stationed.dropna()
    
    # 2. Reindex and Forward-Fill Monthly Macro Data onto Daily Asset Trading Calendars
    daily_timeline = asset_df.index
    
    macro_daily = macro_stationed.reindex(daily_timeline)
    macro_daily = macro_daily.ffill().bfill()  # Propagate the macro snapshot until a new month drops
    
    # 3. Concatenate all datasets into a unified matrix
    master_matrix = pd.concat([asset_df, ff_df, macro_daily], axis=1)
    
    # Standardize remaining data rows
    master_matrix = master_matrix.dropna()
    
    # 4. Feature Standardization via Z-Score (Prevents optimization scale dominance)
    macro_cols = ['cpi_yoy', 'unemployment_delta', 'yield_spread_delta']
    for col in macro_cols:
        mean = master_matrix[col].mean()
        std = master_matrix[col].std()
        master_matrix[f'{col}_zscore'] = (master_matrix[col] - mean) / std
        
    return master_matrix


def run_data_pipeline(fred_key: str, start: str = "1953-04-01", end: str = "2026-01-01") -> pd.DataFrame:
    """
    Orchestration master function for execution inside notebooks or main application layers.
    """
    # Pull data chunks
    spy_returns = fetch_yahoo_returns(start, end)
    ff_factors = fetch_fama_french_factors(start, end)
    fred_macro = fetch_fred_macro(fred_key, start, end)
    
    # Align and mutate matrices
    aligned_dataset = transform_and_align_pipeline(spy_returns, ff_factors, fred_macro)
    print(f"[+] Pipeline complete. Generated shape matrix: {aligned_dataset.shape}")
    return aligned_dataset


if __name__ == "__main__":
    # Retrieve key from environment variable or update string fallback right here:
    API_KEY = os.getenv("FRED_API_KEY", "YOUR_ACTUAL_FRED_API_KEY_HERE")
    
    try:
        sample_df = run_data_pipeline(fred_key=API_KEY, start="1953-04-01", end="2026-07-01")
        print("\n--- SAMPLE VIEW OF ALIGNED PIPELINE ---")
        print(sample_df[['spy_equivalent_return', 'Mkt-RF', 'cpi_yoy_zscore', 'yield_spread_delta_zscore']].head())
    except Exception as e:
        print(f"[-] Execution Pipeline Failure: {str(e)}")