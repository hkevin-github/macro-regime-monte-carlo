#!/usr/bin/env python3
"""
Regime-Aware Monte Carlo Portfolio Dashboard

Features:
- User enters holdings directly in a table
- No CSV upload
- Uses session state to persist holdings across reruns
- Runs portfolio analyzer + regime-aware Monte Carlo simulation
- Displays success probability, downside (20th %ile), median, upside (80th %ile)
- Shows allocation pie chart, fan chart, and final balance histogram
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
    return f"${x:,.0f}"


def default_holdings_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "BIAPX", "shares": 1000.0},
        ]
    )


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


def make_fan_chart(all_balances: np.ndarray):
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
        title="Projected Portfolio Balance",
        xaxis_title="Years",
        yaxis_title="Balance",
        template="plotly_white",
        height=500,
    )
    return fig


def make_histogram(final_balances: np.ndarray):
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
            text=f"{label}: ${val:,.0f}",
            showarrow=False,
            textangle=90,
            font=dict(color=color),
        )

    fig.update_layout(
        title="Final Balance Distribution",
        xaxis_title="Final Balance",
        yaxis_title="Count",
        template="plotly_white",
        height=400,
    )
    return fig


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = default_holdings_table()

if "results" not in st.session_state:
    st.session_state.results = None

if "summary" not in st.session_state:
    st.session_state.summary = None


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("Regime-Aware Monte Carlo Portfolio Dashboard")
st.caption("Enter holdings, choose assumptions, and simulate portfolio outcomes.")

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

    run_button = st.button("Run Simulation", type="primary")

# Main layout
st.subheader("Portfolio Holdings")
st.write("Enter your holdings directly below. Add or remove rows as needed.")

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

# Save edited table into session state immediately so it persists
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
        with st.spinner("Fetching prices, classifying holdings, and running Monte Carlo..."):
            try:
                # 1) Portfolio analysis
                analyzer = PortfolioAnalyzer()
                summary = analyzer.analyze(portfolio_dict)
                st.session_state.summary = summary

                # 2) Monte Carlo simulation
                simulator = RegimeMultiAssetSimulator()
                results = simulator.run_simulation(
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
                st.session_state.results = results

                # 3) Metrics
                final_balances = results["final_balances"]
                p20 = float(np.percentile(final_balances, 20))
                p50 = float(np.percentile(final_balances, 50))
                p80 = float(np.percentile(final_balances, 80))
                success_rate = float(results["success_rate"])

                st.success("Simulation complete")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Probability of Success", f"{success_rate:.1%}")
                col2.metric("Downside (20th %ile)", format_currency(p20))
                col3.metric("Median", format_currency(p50))
                col4.metric("Upside (80th %ile)", format_currency(p80))

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

                # 5) Balance fan chart
                st.subheader("Projected Balance Fan Chart")
                st.plotly_chart(make_fan_chart(results["all_balances"]), use_container_width=True)

                # 6) Final balance histogram
                st.subheader("Final Balance Distribution")
                st.plotly_chart(make_histogram(final_balances), use_container_width=True)

                # 7) Optional detailed stats
                with st.expander("Simulation Statistics", expanded=False):
                    stats_cols = st.columns(4)
                    stats_cols[0].metric("Mean Final Balance", format_currency(results["mean_final"]))
                    stats_cols[1].metric("10th %ile", format_currency(results["p10_final"]))
                    stats_cols[2].metric("25th %ile", format_currency(results["p25_final"]))
                    stats_cols[3].metric("75th %ile", format_currency(results["p75_final"]))

                    st.write("### Full Results Snapshot")
                    st.json(
                        {
                            "success_rate": results["success_rate"],
                            "median_final": results["median_final"],
                            "mean_final": results["mean_final"],
                            "p10_final": results["p10_final"],
                            "p20_final": p20,
                            "p50_final": p50,
                            "p80_final": p80,
                            "p90_final": results["p90_final"],
                            "min_final": results["min_final"],
                            "max_final": results["max_final"],
                        }
                    )

                # 8) Save outputs
                output_dir = Path(config.get("output", {}).get("output_dir", "output"))
                output_dir.mkdir(parents=True, exist_ok=True)

                summary_results = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in results.items()
                    if k not in ["all_balances", "all_regimes"]
                }

                summary_results["p20_final"] = p20
                summary_results["p80_final"] = p80

                with open(output_dir / "simulation_results.json", "w") as f:
                    json.dump(summary_results, f, indent=2)

                st.info(f"Results saved to {output_dir / 'simulation_results.json'}")

            except Exception as e:
                st.exception(e)


# -----------------------------------------------------------------------------
# If simulation already ran, show cached results on reload
# -----------------------------------------------------------------------------
if st.session_state.results is not None and st.session_state.summary is not None:
    st.markdown("---")
    st.subheader("Last Simulation Results")

    results = st.session_state.results
    summary = st.session_state.summary
    final_balances = results["final_balances"]

    p20 = float(np.percentile(final_balances, 20))
    p50 = float(np.percentile(final_balances, 50))
    p80 = float(np.percentile(final_balances, 80))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Probability of Success", f"{results['success_rate']:.1%}")
    col2.metric("Downside (20th %ile)", format_currency(p20))
    col3.metric("Median", format_currency(p50))
    col4.metric("Upside (80th %ile)", format_currency(p80))