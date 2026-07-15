"""
Build historical market data for custom portfolio Monte Carlo.

Downloads proxy asset prices / index series, converts them to monthly returns,
and saves a clean return matrix for simulation.

This version favors longer-history proxies where possible.
"""

import os
from typing import Dict

import pandas as pd
import numpy as np
import yfinance as yf


# ---------------------------------------------------------------------
# Long-history proxy mapping
# ---------------------------------------------------------------------
# These are reasonable starting proxies. Some series will still be shorter
# than others depending on availability.
ASSET_PROXIES: Dict[str, str] = {
    # Equity proxies
    "US_LARGE": "^GSPC",     # S&P 500 index proxy
    "US_SMALL": "IWM",       # shorter history, optional
    "INTERNATIONAL": "EFA",   # shorter history, optional
    "EMERGING": "EEM",       # shorter history, optional
    "REIT": "VNQ",           # shorter history, optional

    # Fixed income / alternatives
    "CORE_BONDS": "AGG",     # shorter history, but common
    "HIGH_YIELD": "HYG",     # shorter history
    "COMMODITIES": "DBC",    # shorter history

    # Cash proxy
    "CASH": "^IRX",          # 13-week T-bill yield proxy, not a price series
}


def download_price_series(ticker: str, start: str = "2000-01-01", end: str | None = None) -> pd.Series:
    """
    Download adjusted close series from Yahoo Finance.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if "Close" not in df.columns:
        raise KeyError(f"Missing Close column for {ticker}")

    close = df["Close"].copy()

    # yfinance sometimes returns a DataFrame if the download structure is odd
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            raise KeyError(f"Ambiguous Close data for {ticker}")

    close.index = pd.to_datetime(close.index)
    close.name = ticker
    return close


def to_monthly_returns(price_series: pd.Series) -> pd.Series:
    """
    Convert daily price series to month-end returns.
    """
    monthly_prices = price_series.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    monthly_returns.index = monthly_returns.index.to_period("M")
    return monthly_returns


def build_market_return_matrix(
    proxies: Dict[str, str],
    start: str = "2000-01-01",
    end: str | None = None
) -> pd.DataFrame:
    """
    Build a monthly return matrix from proxy tickers.

    Notes:
    - '^IRX' is a yield series, not a price series. For now we handle it separately.
    - You can remove CASH if you want only return assets.
    """
    series_list = []

    for asset_name, ticker in proxies.items():
        print(f"[*] Downloading {asset_name} ({ticker})...")

        if asset_name == "CASH":
            # Convert 13-week T-bill yield into an approximate monthly cash return
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

            if raw.empty or "Close" not in raw.columns:
                raise ValueError(f"No data returned for cash proxy {ticker}")

            y = raw["Close"].copy()
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]

            y.index = pd.to_datetime(y.index)

            # '^IRX' is quoted in percent yield terms, so:
            # monthly cash return approx = annual_yield / 100 / 12
            monthly_cash = y.resample("ME").last().dropna() / 100.0 / 12.0
            monthly_cash.index = monthly_cash.index.to_period("M")
            monthly_cash.name = asset_name
            series_list.append(monthly_cash)
            continue

        prices = download_price_series(ticker, start=start, end=end)
        rets = to_monthly_returns(prices)
        rets.name = asset_name
        series_list.append(rets)

    market_df = pd.concat(series_list, axis=1).dropna()
    market_df.index.name = "date"
    return market_df


def save_market_data(
    df: pd.DataFrame,
    output_path: str = "data/processed/custom_portfolio_market_data.csv"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = df.copy()
    out.index = out.index.to_timestamp(how="end").normalize()
    out.index.name = "date"
    out.to_csv(output_path)
    print(f"[+] Saved market data to: {output_path}")


if __name__ == "__main__":
    market_df = build_market_return_matrix(ASSET_PROXIES, start="2000-01-01")
    print(market_df.head())
    print("\nColumns:", list(market_df.columns))
    print("\nShape:", market_df.shape)
    save_market_data(market_df)