#!/usr/bin/env python3
"""
Regime-Aware Monte Carlo Portfolio Dashboard

Features:
- User enters holdings directly in a table, OR loads them from a CSV path
- Uses session state to persist holdings across reruns
- Runs portfolio analyzer + regime-aware Monte Carlo simulation
- Displays success probability, downside (20th %ile), median, upside (80th %ile)
- Shows allocation pie chart, fan chart, and final balance histogram
- NOW: Side-by-side comparison with bootstrap historical sampling
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from portfolio_analyzer import PortfolioAnalyzer
from regime_portfolio_simulator import RegimeMultiAssetSimulator
from bootstrap_simulator import BootstrapMonteCarloSimulator


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Regime-Aware Monte Carlo Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_config(config_path: str = "config/info.json") -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)

    # fallback defaults if config file missing
    return {
        "simulation": {
            "n_trials": 10000,
            "n_years": 30,
            "random_seed": 42,
        },
        "portfolio": {
            "annual_contribution": 0.0,
            "annual_withdrawal": 0.0,
        },
        "output": {
            "output_dir": "output",
        },
    }


def format_currency(x: float) -> str:
    sign = '-' if x < 0 else ''
    return f"{sign}${abs(x):,.0f}"


def default_holdings_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "BIAPX", "shares": 1000.0},
        ]
    )


DEFAULT_PORTFOLIO_CSV_PATH = "/workspaces/macro-regime-monte-carlo/config/my_portfolio.csv"


def load_holdings_from_csv(csv_path: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Load a ticker,shares holdings table from a CSV path.
    Returns (dataframe, error_message). Exactly one of the two will be None.
    """
    path = Path(csv_path)
    if not path.exists():
        return None, f"File not found: {csv_path}"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, f"Failed to read CSV: {e}"

    missing_cols = {"ticker", "shares"} - set(df.columns.str.lower())
    if missing_cols:
        return None, f"CSV must contain columns: ticker, shares (found: {list(df.columns)})"

    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)[["ticker", "shares"]]
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df = df.dropna(subset=["shares"])

    if df.empty:
        return None, "CSV was read but contained no valid ticker/shares rows"

    return df.reset_index(drop=True), None


def dataframe_to_portfolio_dict(df: pd.DataFrame) -> dict:
    portfolio = {}
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        shares = row.get("shares", None)

        if ticker == "":
            continue
        if shares is None or pd.isna(shares):
            continue

        try:
            shares = float(shares)
        except Exception:
            continue

        if shares <= 0:
            continue

        portfolio[ticker] = shares

    return portfolio


def make_fan_chart(all_balances: np.ndarray, title: str = "Projected Portfolio Balance"):
    years = np.arange(all_balances.shape[1]) / 12.0

    p20 = np.percentile(all_balances, 20, axis=0)
    p50 = np.percentile(all_balances, 50, axis=0)
    p80 = np.percentile(all_balances, 80, axis=0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([p80, p20[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 123, 255, 0.20)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="20th–80th percentile",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=p50,
            mode="lines",
            line=dict(color="blue", width=3),
            name="Median",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Years",
        yaxis_title="Balance",
        template="plotly_white",
        height=400,
    )
    return fig


def make_histogram(final_balances: np.ndarray, title: str = "Final Balance Distribution"):
    p20 = np.percentile(final_balances, 20)
    p50 = np.percentile(final_balances, 50)
    p80 = np.percentile(final_balances, 80)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=final_balances,
            nbinsx=50,
            marker_color="steelblue",
            opacity=0.75,
            name="Final Balances",
        )
    )

    for val, color, label in [
        (p20, "orange", "20th"),
        (p50, "red", "Median"),
        (p80, "purple", "80th"),
    ]:
        fig.add_vline(x=val, line_width=2, line_dash="dash", line_color=color)
        fig.add_annotation(
            x=val,
            y=1,
            yref="paper",
            text=f"{label}: {format_currency(val)}",
            showarrow=False,
            textangle=90,
            font=dict(color=color),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Final Balance",
        yaxis_title="Count",
        template="plotly_white",
        height=300,
    )
    return fig


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = default_holdings_table()

if "regime_results" not in st.session_state:
    st.session_state.regime_results = None

if "bootstrap_results" not in st.session_state:
    st.session_state.bootstrap_results = None

if "summary" not in st.session_state:
    st.session_state.summary = None


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("Regime-Aware Monte Carlo Portfolio Dashboard")
st.caption("Compare regime-aware simulation vs. bootstrap historical sampling")

config = load_config()

# Sidebar controls
with st.sidebar:
    st.header("Simulation Settings")

    n_years = st.number_input(
        "Time Horizon (years)",
        min_value=1,
        max_value=60,
        value=int(config["simulation"].get("n_years", 30)),
        step=1,
    )

    n_trials = st.number_input(
        "Monte Carlo Trials",
        min_value=100,
        max_value=100000,
        value=int(config["simulation"].get("n_trials", 10000)),
        step=1000,
    )

    annual_contribution = st.number_input(
        "Annual Deposit",
        min_value=0.0,
        value=float(config["portfolio"].get("annual_contribution", 0.0)),
        step=1000.0,
    )

    annual_withdrawal = st.number_input(
        "Annual Withdrawal",
        min_value=0.0,
        value=float(config["portfolio"].get("annual_withdrawal", 0.0)),
        step=1000.0,
    )

    annual_fee = float(config["portfolio"].get("annual_fee", 0.0))
    contribution_years = int(config["portfolio"].get("contribution_years", 0))
    failure_threshold = float(config["portfolio"].get("failure_threshold", 0.0))
    target_end_balance = float(config["portfolio"].get("target_end_balance", 0.0))

    random_seed = st.number_input(
        "Random Seed",
        min_value=0,
        value=int(config["simulation"].get("random_seed", 42)),
        step=1,
    )

    run_button = st.button("Run Both Simulations", type="primary")

# Main layout
st.subheader("Portfolio Holdings")
st.write("Enter your holdings directly below, add/remove rows, or load them from a CSV file.")

with st.expander("Load holdings from CSV", expanded=False):
    csv_path_input = st.text_input(
        "CSV path (must have columns: ticker, shares)",
        value=DEFAULT_PORTFOLIO_CSV_PATH,
    )
    load_csv_button = st.button("Load CSV")

    if load_csv_button:
        loaded_df, load_error = load_holdings_from_csv(csv_path_input)
        if load_error:
            st.error(load_error)
        else:
            st.session_state.portfolio_df = loaded_df
            st.success(f"Loaded {len(loaded_df)} holding(s) from {csv_path_input}")
            st.rerun()

edited_df = st.data_editor(
    st.session_state.portfolio_df,
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_editor",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker"),
        "shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=1.0),
    },
)

st.session_state.portfolio_df = edited_df

portfolio_dict = dataframe_to_portfolio_dict(edited_df)

if len(portfolio_dict) == 0:
    st.warning("No valid holdings entered yet. Add at least one ticker with positive shares.")

with st.expander("Current Parsed Portfolio", expanded=False):
    st.json(portfolio_dict if portfolio_dict else {})

# -----------------------------------------------------------------------------
# Run simulation
# -----------------------------------------------------------------------------
if run_button:
    if len(portfolio_dict) == 0:
        st.error("Please enter at least one valid holding before running the simulation.")
    else:
        with st.spinner("Running regime-aware and bootstrap simulations..."):
            try:
                # Load historical data
                historical_data = pd.read_csv("data/processed/regime_labeled_dataset.csv")
                
                # 1) Portfolio analysis
                analyzer = PortfolioAnalyzer()
                summary = analyzer.analyze(portfolio_dict)
                st.session_state.summary = summary

                # 2) Regime-aware simulation
                regime_simulator = RegimeMultiAssetSimulator()
                regime_results = regime_simulator.run_simulation(
                    asset_class_weights=summary["asset_class_weights"],
                    initial_balance=summary["total_value"],
                    n_trials=int(n_trials),
                    n_years=int(n_years),
                    annual_contribution=float(annual_contribution),
                    annual_withdrawal=float(annual_withdrawal),
                    annual_fee=annual_fee,
                    contribution_years=contribution_years,
                    failure_threshold=failure_threshold,
                    target_end_balance=target_end_balance,
                    random_state=int(random_seed),
                )
                st.session_state.regime_results = regime_results

                # 3) Bootstrap simulation
                bootstrap_simulator = BootstrapMonteCarloSimulator(historical_data)
                bootstrap_results = bootstrap_simulator.run_simulation(
                    asset_class_weights=summary["asset_class_weights"],
                    initial_balance=summary["total_value"],
                    n_trials=int(n_trials),
                    n_years=int(n_years),
                    annual_contribution=float(annual_contribution),
                    annual_withdrawal=float(annual_withdrawal),
                    annual_fee=annual_fee,
                    contribution_years=contribution_years,
                    failure_threshold=failure_threshold,
                    target_end_balance=target_end_balance,
                    random_state=int(random_seed),
                )
                st.session_state.bootstrap_results = bootstrap_results

                st.success("Both simulations complete")

                # 4) Portfolio summary
                st.subheader("Portfolio Summary")
                summary_cols = st.columns([1, 1])

                with summary_cols[0]:
                    st.write(f"**Total Value:** {format_currency(summary['total_value'])}")
                    st.write(f"**Number of Holdings:** {summary['num_holdings']}")
                    st.write("**Asset Allocation:**")
                    alloc_df = pd.DataFrame(
                        [{"Asset Class": k, "Weight": v} for k, v in summary["asset_class_weights"].items()]
                    )
                    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

                with summary_cols[1]:
                    fig_alloc = px.pie(
                        alloc_df,
                        names="Asset Class",
                        values="Weight",
                        title="Portfolio Allocation",
                    )
                    st.plotly_chart(fig_alloc, use_container_width=True)

                # 5) Side-by-side comparison
                st.subheader("📊 Simulation Comparison")
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.markdown("### 🎯 Regime-Aware")
                    regime_final = regime_results["final_balances"]
                    regime_p20 = float(np.percentile(regime_final, 20))
                    regime_p50 = float(np.percentile(regime_final, 50))
                    regime_p80 = float(np.percentile(regime_final, 80))
                    
                    st.metric("Success Rate", f"{regime_results['success_rate']:.1%}")
                    st.metric("20th %ile", format_currency(regime_p20))
                    st.metric("Median", format_currency(regime_p50))
                    st.metric("80th %ile", format_currency(regime_p80))
                    
                    st.plotly_chart(
                        make_fan_chart(regime_results["all_balances"], "Regime-Aware Balance"),
                        use_container_width=True
                    )
                    st.plotly_chart(
                        make_histogram(regime_final, "Regime-Aware Final Distribution"),
                        use_container_width=True
                    )
                
                with comp_col2:
                    st.markdown("### 🎲 Bootstrap Historical")
                    bootstrap_final = bootstrap_results["final_balances"]
                    bootstrap_p20 = float(np.percentile(bootstrap_final, 20))
                    bootstrap_p50 = float(np.percentile(bootstrap_final, 50))
                    bootstrap_p80 = float(np.percentile(bootstrap_final, 80))
                    
                    st.metric("Success Rate", f"{bootstrap_results['success_rate']:.1%}")
                    st.metric("20th %ile", format_currency(bootstrap_p20))
                    st.metric("Median", format_currency(bootstrap_p50))
                    st.metric("80th %ile", format_currency(bootstrap_p80))
                    
                    st.plotly_chart(
                        make_fan_chart(bootstrap_results["all_balances"], "Bootstrap Balance"),
                        use_container_width=True
                    )
                    st.plotly_chart(
                        make_histogram(bootstrap_final, "Bootstrap Final Distribution"),
                        use_container_width=True
                    )

                # 6) Save outputs
                output_dir = Path(config.get("output", {}).get("output_dir", "output"))
                output_dir.mkdir(parents=True, exist_ok=True)

                regime_summary = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in regime_results.items()
                    if k not in ["all_balances", "all_regimes"]
                }
                regime_summary["p20_final"] = regime_p20
                regime_summary["p80_final"] = regime_p80

                bootstrap_summary = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in bootstrap_results.items()
                    if k not in ["all_balances", "all_regimes"]
                }
                bootstrap_summary["p20_final"] = bootstrap_p20
                bootstrap_summary["p80_final"] = bootstrap_p80

                with open(output_dir / "regime_results.json", "w") as f:
                    json.dump(regime_summary, f, indent=2)
                
                with open(output_dir / "bootstrap_results.json", "w") as f:
                    json.dump(bootstrap_summary, f, indent=2)

                st.info(f"Results saved to {output_dir}")

            except Exception as e:
                st.exception(e)


# -----------------------------------------------------------------------------
# Show cached results on reload
# -----------------------------------------------------------------------------
if st.session_state.regime_results is not None and st.session_state.bootstrap_results is not None:
    st.markdown("---")
    st.subheader("Last Simulation Results")

    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("### 🎯 Regime-Aware")
        regime_results = st.session_state.regime_results
        regime_final = regime_results["final_balances"]
        regime_p20 = float(np.percentile(regime_final, 20))
        regime_p50 = float(np.percentile(regime_final, 50))
        regime_p80 = float(np.percentile(regime_final, 80))
        
        st.metric("Success Rate", f"{regime_results['success_rate']:.1%}")
        st.metric("20th %ile", format_currency(regime_p20))
        st.metric("Median", format_currency(regime_p50))
        st.metric("80th %ile", format_currency(regime_p80))
    
    with comp_col2:
        st.markdown("### 🎲 Bootstrap Historical")
        bootstrap_results = st.session_state.bootstrap_results
        bootstrap_final = bootstrap_results["final_balances"]
        bootstrap_p20 = float(np.percentile(bootstrap_final, 20))
        bootstrap_p50 = float(np.percentile(bootstrap_final, 50))
        bootstrap_p80 = float(np.percentile(bootstrap_final, 80))
        
        st.metric("Success Rate", f"{bootstrap_results['success_rate']:.1%}")
        st.metric("20th %ile", format_currency(bootstrap_p20))
        st.metric("Median", format_currency(bootstrap_p50))
        st.metric("80th %ile", format_currency(bootstrap_p80))