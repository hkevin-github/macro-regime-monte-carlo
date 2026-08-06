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
Bond Returns      -> GS10 (synthetic) -> FRED           -> Monthly -> 1953 - Present
                     AGG (real, short)-> Yahoo Finance  -> Monthly -> 2003 - Present

Bond returns are now built primarily from a synthetic constant-maturity 10-year
Treasury total-return proxy derived from FRED's GS10 yield series (see
synthesize_treasury_returns_from_yield below), which matches the 1953+ horizon of
the rest of the macro dataset. The real AGG ETF series is still fetched and used
to validate/patch recent months where available, since it reflects actual traded
returns rather than an approximation -- see merge_bond_return_sources.

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


def synthesize_treasury_returns_from_yield(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Builds a synthetic constant-maturity 10-year Treasury MONTHLY total-return series
    from FRED's GS10 yield series, back to 1953 -- matching the horizon of the rest
    of the macro dataset (unlike AGG, which only starts in 2003).

    This is a standard duration-based total-return approximation used widely in
    academic/practitioner long-run bond-return work when a real total-return index
    isn't available for the full window:

        duration[t]      ~= 1 / yield[t]                  (rough constant-maturity
                                                             duration approximation)
        price_return[t]  ~= -duration[t-1] * (yield[t] - yield[t-1])
        income_return[t] ~=  yield[t-1] / 12
        total_return[t]  ~=  price_return[t] + income_return[t]

    IMPORTANT CAVEATS (documented explicitly, not hidden):
    - This is a CONSTRUCTED PROXY, not a measured total-return index. It approximates
      a constant-maturity 10-year Treasury, not a diversified aggregate bond fund
      like AGG (no corporate credit, no MBS, different duration profile).
    - The duration approximation (1 / yield) is a simplification; it does not account
      for convexity or the true cash-flow structure of an actual 10-year note.
    - Because it is duration-based and 10Y-specific, this proxy will show MORE
      interest-rate sensitivity (higher volatility) than AGG did historically,
      particularly during the high-rate-volatility 1970s-1980s. This is expected
      and is arguably more representative of true long-duration Treasury risk in
      those regimes than assuming AGG-like behavior would have applied.
    """
    print(f"[*] Synthesizing constant-maturity 10Y Treasury returns from GS10 (1953+ proxy)...")
    normalized_key = normalize_api_key(api_key)
    try:
        fred = Fred(api_key=normalized_key)
        yields = fred.get_series('GS10', observation_start=start_date, observation_end=end_date)
        yields.index = pd.to_datetime(yields.index)

        monthly_yield = yields.resample('ME').last() / 100.0  # decimal, e.g. 0.045
        monthly_yield = monthly_yield.dropna()

        prev_yield = monthly_yield.shift(1)
        duration = 1.0 / prev_yield.replace(0, np.nan)

        price_return = -duration * (monthly_yield - prev_yield)
        income_return = prev_yield / 12.0

        total_return = (price_return + income_return).dropna()
        total_return.index = total_return.index.to_period('M')
        total_return.name = 'bond_return_synthetic'

        print(f"[+] Synthesized Treasury proxy: {len(total_return)} months, "
              f"starts {total_return.index.min()}")
        return total_return.to_frame()

    except Exception as e:
        print(f"[!] Synthetic Treasury return construction failed: {e}")
        print(f"[!] Falling back to AGG-only bond coverage (2003+).")
        return pd.DataFrame()


def merge_bond_return_sources(synthetic_df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines the long-horizon synthetic Treasury proxy (1953+) with the real AGG
    ETF series (2003+) into a single 'bond_return' column spanning the full window.

    Preference logic: use REAL AGG returns wherever they exist (they reflect actual
    traded fund performance, including credit/MBS exposure and true fund-level
    duration), and fall back to the synthetic GS10-based proxy for months before
    AGG's 2003 inception. This is flagged via a 'bond_return_source' column so
    downstream consumers can see which months are real vs. synthetic if needed.
    """
    if synthetic_df.empty and agg_df.empty:
        print("[!] No bond data available from either source.")
        return pd.DataFrame()

    if synthetic_df.empty:
        print("[!] No synthetic Treasury proxy available; using AGG-only coverage.")
        out = agg_df.rename(columns={'bond_return': 'bond_return'}).copy()
        out['bond_return_source'] = 'agg'
        return out[['bond_return', 'bond_return_source']]

    if agg_df.empty:
        print("[!] No AGG data available; using synthetic Treasury proxy only.")
        out = synthetic_df.rename(columns={'bond_return_synthetic': 'bond_return'}).copy()
        out['bond_return_source'] = 'synthetic_gs10'
        return out[['bond_return', 'bond_return_source']]

    merged = synthetic_df.join(agg_df, how='outer')
    merged['bond_return'] = merged['bond_return'].combine_first(merged['bond_return_synthetic'])
    merged['bond_return_source'] = np.where(
        merged['bond_return'].notna() & merged.index.isin(agg_df.index),
        'agg',
        'synthetic_gs10',
    )

    n_agg = int((merged['bond_return_source'] == 'agg').sum())
    n_synthetic = int((merged['bond_return_source'] == 'synthetic_gs10').sum())
    print(f"[+] Merged bond return sources: {n_agg} months real AGG, "
          f"{n_synthetic} months synthetic GS10 proxy, "
          f"full range {merged.index.min()} to {merged.index.max()}")

    return merged[['bond_return', 'bond_return_source']].dropna(subset=['bond_return'])


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


def fetch_international_equity_returns(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches MONTHLY international developed-market equity returns from Ken French's
    Developed ex-US 3-Factor library. This is real, free, programmatically-available
    total-return data (unlike an international ETF, which only has ~15-20 years of
    history). Coverage starts around 1990 rather than 1953 -- still short of the full
    macro window, but a genuine measured series rather than a fabricated one.
    """
    print(f"[*] Ingesting Ken French Developed ex-US Equity Factors (international proxy)...")
    try:
        ff_intl = web.DataReader(
            'Developed_ex_US_3_Factors', 'famafrench', start=start_date, end=end_date
        )
        df = ff_intl[0] / 100.0

        if isinstance(df.index, pd.PeriodIndex):
            df.index = df.index.asfreq('M')
        else:
            df.index = pd.to_datetime(df.index).to_period('M')

        # Total developed ex-US market return = Mkt-RF + RF
        intl_returns = (df['Mkt-RF'] + df['RF']).to_frame(name='intl_equity_return')
        print(f"[+] Fetched intl equity proxy: {len(intl_returns)} months, "
              f"starts {intl_returns.index.min()}")
        return intl_returns
    except Exception as e:
        print(f"[!] International equity fetch failed: {e}")
        print(f"[!] International equity regime stats will fall back to US equity.")
        return pd.DataFrame()


def fetch_real_estate_proxy(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches the Case-Shiller U.S. National Home Price Index (FRED: CSUSHPINSA) as a
    real-estate return proxy, monthly, back to 1987.

    IMPORTANT: This is a HOME-PRICE APPRECIATION series, not a REIT total-return index.
    It excludes dividend yield, leverage, and public-market volatility that a REIT ETF
    would exhibit. It is used here because it is real, free, and long-history -- but
    should be understood as a conservative, lower-volatility proxy for "real estate"
    exposure, not an equivalent substitute for listed REIT returns.
    """
    print(f"[*] Ingesting Case-Shiller Home Price Index (real estate proxy, CSUSHPINSA)...")
    normalized_key = normalize_api_key(api_key)
    try:
        fred = Fred(api_key=normalized_key)
        series = fred.get_series('CSUSHPINSA', observation_start=start_date, observation_end=end_date)
        series.index = pd.to_datetime(series.index)
        monthly = series.resample('ME').last()
        returns = monthly.pct_change().dropna()
        returns.index = returns.index.to_period('M')
        returns.name = 'real_estate_return'
        print(f"[+] Fetched real estate proxy: {len(returns)} months, starts {returns.index.min()}")
        return returns.to_frame()
    except Exception as e:
        print(f"[!] Real estate proxy fetch failed: {e}")
        print(f"[!] Real estate regime stats will fall back to US equity.")
        return pd.DataFrame()


def fetch_commodities_proxy(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches the Producer Price Index: All Commodities (FRED: PPIACO) as a commodities
    return proxy, monthly, back to 1913.

    IMPORTANT: PPIACO is a wholesale PRICE index, not an investable commodities total-
    return index (like the S&P GSCI). It has no roll yield, collateral yield, or futures
    curve effects. It is used here as a real, free, long-history proxy for commodity
    price cycles -- flagged explicitly as a constructed proxy rather than a measured
    investable return.
    """
    print(f"[*] Ingesting Producer Price Index: All Commodities (commodities proxy, PPIACO)...")
    normalized_key = normalize_api_key(api_key)
    try:
        fred = Fred(api_key=normalized_key)
        series = fred.get_series('PPIACO', observation_start=start_date, observation_end=end_date)
        series.index = pd.to_datetime(series.index)
        monthly = series.resample('ME').last()
        returns = monthly.pct_change().dropna()
        returns.index = returns.index.to_period('M')
        returns.name = 'commodities_return'
        print(f"[+] Fetched commodities proxy: {len(returns)} months, starts {returns.index.min()}")
        return returns.to_frame()
    except Exception as e:
        print(f"[!] Commodities proxy fetch failed: {e}")
        print(f"[!] Commodities regime stats will fall back to US equity.")
        return pd.DataFrame()


def fetch_extended_asset_returns(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches all extended asset-class return proxies (international equity, real estate,
    commodities) and merges them into a single DataFrame keyed on Period[M].

    These series have shorter histories than the core 1953+ macro dataset, so this
    DataFrame will contain leading NaNs per column until that series' inception date.
    This is expected: regime statistics for each asset class are computed later using
    only the months where that specific asset class has real data (see
    hmm_regime_engine.py), and older regimes with no coverage will fall back to
    US equity stats at simulation time rather than being filled here.

    NOTE: Emerging markets equity is intentionally NOT included as a separate series.
    There is no free, long-history, programmatically-fetchable EM total-return series
    available, so EM exposure is folded into the International_Equity bucket in the
    simulator rather than being given a fabricated distribution.
    """
    frames = []

    intl = fetch_international_equity_returns(start_date, end_date)
    if not intl.empty:
        frames.append(intl)

    real_estate = fetch_real_estate_proxy(api_key, start_date, end_date)
    if not real_estate.empty:
        frames.append(real_estate)

    commodities = fetch_commodities_proxy(api_key, start_date, end_date)
    if not commodities.empty:
        frames.append(commodities)

    if not frames:
        return pd.DataFrame()

    extended_df = pd.concat(frames, axis=1)
    return extended_df


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
                                 bond_df: pd.DataFrame = None,
                                 extended_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Transforms raw macro factors for stationarity using smoothed rolling windows
    and CAUSAL expanding-window z-score standardization to eliminate look-ahead bias.
    
    Args:
        asset_df: Equity returns
        ff_df: Fama-French factors
        macro_df: Macro indicators
        bond_df: Bond returns (optional). Merged via LEFT JOIN (see step 4a below)
            rather than the core concat/dropna, so that bond coverage does NOT
            truncate the 1953+ training window -- this now spans the merged
            synthetic-GS10 (1953+) and real-AGG (2003+) series from
            merge_bond_return_sources, so coverage should be full-history in
            practice, but the join stays defensive in case bond_df is ever partial.
        extended_df: Extended asset-class return proxies -- international equity,
            real estate, commodities (optional). Merged via LEFT JOIN after the core
            dropna() step so that these shorter-history columns do NOT truncate the
            core 1953+ training window used for HMM fitting. Expect NaNs in these
            columns before each series' inception date.
    
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

    # 2. Concatenate CORE datasets only (equity, Fama-French, macro) on shared
    #    monthly Period[M] index. Bonds are intentionally NOT included here anymore
    #    -- see step 4a -- so that partial bond coverage can never truncate the
    #    core 1953+ training window the way AGG's 2003 inception used to.
    dataframes_to_concat = [asset_df, ff_df, macro_stationed]
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

    # 4a. Left-join bond returns (merged synthetic GS10 1953+ / real AGG 2003+ series
    #     from merge_bond_return_sources). Left-joined rather than concatenated into
    #     the core dropna() step above, so that any remaining gaps in bond coverage
    #     cannot truncate the core 1953+ macro/equity training window the way AGG's
    #     2003 inception used to when bonds were part of the hard concat.
    if bond_df is not None and not bond_df.empty:
        master_matrix = master_matrix.join(bond_df, how='left')
        n_bond_obs = master_matrix['bond_return'].notna().sum() if 'bond_return' in master_matrix.columns else 0
        print(f"[+] Merged bond returns (non-null months): {n_bond_obs} of {len(master_matrix)}")
    else:
        print("[!] No bond data available - simulator will use proxy model")

    # 4b. Left-join extended asset-class return proxies (international, real estate,
    #    commodities). These have shorter histories than the core matrix, so this
    #    intentionally introduces NaNs for the pre-inception months of each series
    #    rather than dropping rows -- the core 1953+ training window is preserved.
    if extended_df is not None and not extended_df.empty:
        master_matrix = master_matrix.join(extended_df, how='left')
        coverage = {
            col: master_matrix[col].notna().sum() for col in extended_df.columns
        }
        print(f"[+] Merged extended asset-class proxies (non-null months): {coverage}")

    # 5. Convert Period[M] index back to timestamps for downstream compatibility
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

    agg_bond_returns = fetch_bond_returns(start, end)
    synthetic_bond_returns = synthesize_treasury_returns_from_yield(fred_key, start, end)
    bond_returns = merge_bond_return_sources(synthetic_bond_returns, agg_bond_returns)

    extended_returns = fetch_extended_asset_returns(fred_key, start, end)

    aligned_dataset = transform_and_align_pipeline(
        equity_returns, ff_factors, fred_macro, bond_returns, extended_returns
    )
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

        print("\n--- EXTENDED ASSET-CLASS PROXY COVERAGE ---")
        for col in ['intl_equity_return', 'real_estate_return', 'commodities_return']:
            if col in sample_df.columns:
                n_obs = sample_df[col].notna().sum()
                first_valid = sample_df[col].first_valid_index()
                print(f"  {col}: {n_obs} months available, starting {first_valid}")
    except Exception as e:
        print(f"[-] Execution Pipeline Failure: {str(e)}")