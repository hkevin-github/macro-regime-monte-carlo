#!/usr/bin/env python3
"""
Cycle-Aware Monte Carlo Portfolio Dashboard

Features:
- User enters holdings directly in a table, OR loads them from a CSV path
- Uses session state to persist holdings across reruns
- Runs portfolio analyzer + cycle-aware Monte Carlo simulation
- Displays success probability, downside (20th %ile), median, upside (80th %ile)
- Shows allocation, fan chart, and final balance distribution
- Side-by-side comparison with bootstrap historical sampling
- Plain-English explanation of why the two projections differ

Note on naming: internally this still uses the original "regime" model classes
(RegimeMultiAssetSimulator, etc.) -- only the dashboard-facing copy says
"cycle" instead of "regime".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from portfolio_analyzer import PortfolioAnalyzer
from regime_portfolio_simulator import RegimeMultiAssetSimulator
from bootstrap_simulator import BootstrapMonteCarloSimulator
from regime_explainer import render_streamlit_explanation


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cycle-Aware Portfolio Projections",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Visual system
# -----------------------------------------------------------------------------
# Palette: warm paper background, near-black ink for text, a single muted
# slate-teal accent used sparingly for emphasis and the "cycle-aware" series.
# The bootstrap/historical series uses a neutral warm grey so the two never
# compete for attention -- the accent color always means "cycle-aware".
INK = "#1C1E21"
PAPER = "#FBFAF7"
PANEL = "#FFFFFF"
LINE = "#E4E1D8"
MUTED = "#7A7668"
ACCENT = "#3E6B64"        # slate teal -- cycle-aware series
ACCENT_SOFT = "rgba(62, 107, 100, 0.14)"
NEUTRAL = "#A8A296"       # warm grey -- bootstrap/historical series
NEUTRAL_SOFT = "rgba(168, 162, 150, 0.20)"

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        .stApp {{
            background-color: {PAPER};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {PANEL};
            border-right: 1px solid {LINE};
        }}

        h1, h2, h3 {{
            font-family: 'Source Serif 4', Georgia, serif !important;
            color: {INK} !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }}

        .app-kicker {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {MUTED};
            margin-bottom: 0.2rem;
        }}

        .app-title {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 2.1rem;
            font-weight: 600;
            color: {INK};
            margin: 0 0 0.15rem 0;
            line-height: 1.15;
        }}

        .app-subtitle {{
            color: {MUTED};
            font-size: 0.95rem;
            margin-bottom: 1.6rem;
        }}

        .section-label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {MUTED};
            border-bottom: 1px solid {LINE};
            padding-bottom: 0.5rem;
            margin: 2.2rem 0 1rem 0;
        }}

        /* Stat cards */
        .stat-card {{
            background: {PANEL};
            border: 1px solid {LINE};
            border-radius: 10px;
            padding: 1rem 1.1rem;
            height: 100%;
        }}
        .stat-card .stat-label {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {MUTED};
            margin-bottom: 0.35rem;
        }}
        .stat-card .stat-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.5rem;
            font-weight: 500;
            color: {INK};
            line-height: 1.1;
        }}
        .stat-card.accent {{
            border-color: {ACCENT};
            background: {ACCENT_SOFT};
        }}
        .stat-card.accent .stat-value {{ color: {ACCENT}; }}

        /* Series panel headers */
        .series-header {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 0.9rem;
        }}
        .series-dot {{
            width: 10px; height: 10px; border-radius: 50%;
            flex-shrink: 0;
        }}
        .series-title {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 1.15rem;
            font-weight: 600;
            color: {INK};
        }}
        .series-caption {{
            color: {MUTED};
            font-size: 0.82rem;
            margin: -0.5rem 0 0.9rem 1.55rem;
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {INK};
            color: {PAPER};
            border-radius: 7px;
            border: none;
            font-weight: 500;
            padding: 0.55rem 1.1rem;
        }}
        .stButton > button:hover {{
            background-color: {ACCENT};
            color: white;
        }}

        /* Data editor / dataframe corners */
        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid {LINE};
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid {LINE};
            border-radius: 8px;
        }}

        hr {{ border-color: {LINE}; }}

        /* Widget labels and input text */
        label, 
        div[data-testid="stWidgetLabel"] p, 
        div[data-testid="stWidgetLabel"] label, 
        div[data-baseweb="input"] input, 
        div[data-baseweb="base-input"] input {{
            color: {INK} !important;
        }}

        /* Expander headers (Advanced, Load from CSV instead) */
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary span,
        div[data-testid="stExpander"] summary svg {{
            color: {INK} !important;
            fill: {INK} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_config(config_path: str = "config/info.json") -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)

    return {
        "simulation": {"n_trials": 10000, "n_years": 30, "random_seed": 42},
        "portfolio": {"annual_contribution": 0.0, "annual_withdrawal": 0.0},
        "output": {"output_dir": "output"},
    }


def format_currency(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def default_holdings_table() -> pd.DataFrame:
    return pd.DataFrame([{"ticker": "BIAPX", "shares": 1000.0}])


DEFAULT_PORTFOLIO_CSV_PATH = "/workspaces/macro-regime-monte-carlo/config/my_portfolio.csv"


def load_holdings_from_csv(csv_path: str) -> tuple[pd.DataFrame | None, str | None]:
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

        if ticker == "" or shares is None or pd.isna(shares):
            continue
        try:
            shares = float(shares)
        except Exception:
            continue
        if shares <= 0:
            continue

        portfolio[ticker] = shares
    return portfolio


def stat_card(label: str, value: str, accent: bool = False) -> str:
    cls = "stat-card accent" if accent else "stat-card"
    return f"""<div class="{cls}"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>"""


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def series_header(title: str, caption: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="series-header">
            <div class="series-dot" style="background:{color};"></div>
            <div class="series-title">{title}</div>
        </div>
        <div class="series-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
PLOT_FONT = dict(family="Inter, sans-serif", color=INK, size=12)


def make_fan_chart(all_balances: np.ndarray, color: str, fill: str) -> go.Figure:
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
            fillcolor=fill,
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="20th\u201380th pct.",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years, y=p50, mode="lines",
            line=dict(color=color, width=2.5),
            name="Median",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=PANEL,
        paper_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        xaxis=dict(title="Years", gridcolor=LINE, zeroline=False),
        yaxis=dict(title=None, gridcolor=LINE, zeroline=False, tickprefix="$"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10)),
        showlegend=True,
    )
    return fig


def make_histogram(final_balances: np.ndarray, color: str) -> go.Figure:
    p50 = np.percentile(final_balances, 50)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=final_balances, nbinsx=45,
            marker_color=color, opacity=0.55,
            name="Outcomes",
        )
    )
    fig.add_vline(x=p50, line_width=2, line_color=INK, line_dash="dot")
    fig.add_annotation(
        x=p50, y=1, yref="paper", showarrow=False, textangle=90,
        text=f"Median {format_currency(p50)}", font=dict(size=10, color=INK), xshift=-8,
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=PANEL,
        paper_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        xaxis=dict(title="Final balance", gridcolor=LINE, zeroline=False, tickprefix="$"),
        yaxis=dict(title=None, showgrid=False),
        showlegend=False,
        bargap=0.05,
    )
    return fig


def make_allocation_chart(alloc_df: pd.DataFrame) -> go.Figure:
    colors = [ACCENT, "#6B8F88", "#9CB5B0", MUTED, "#C4BFAF", "#D8D3C4", LINE]
    fig = go.Figure(
        go.Pie(
            labels=alloc_df["Asset Class"],
            values=alloc_df["Weight"],
            hole=0.62,
            marker=dict(colors=colors, line=dict(color=PAPER, width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, family="Inter, sans-serif"),
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        showlegend=False,
    )
    return fig


def render_result_panel(results: dict, title: str, caption: str, color: str, fill: str) -> None:
    final = results["final_balances"]
    p20 = float(np.percentile(final, 20))
    p50 = float(np.percentile(final, 50))
    p80 = float(np.percentile(final, 80))

    series_header(title, caption, color)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(stat_card("Success rate", f"{results['success_rate']:.0%}", accent=(color == ACCENT)), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_card("Median outcome", format_currency(p50), accent=(color == ACCENT)), unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(stat_card("20th pct.", format_currency(p20)), unsafe_allow_html=True)
    with r2:
        st.markdown(stat_card("80th pct.", format_currency(p80)), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.plotly_chart(make_fan_chart(results["all_balances"], color, fill), use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(make_histogram(final, color), use_container_width=True, config={"displayModeBar": False})


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
for key, default in [
    ("portfolio_df", None),
    ("regime_results", None),
    ("bootstrap_results", None),
    ("summary", None),
    ("regime_simulator", None),
    ("explanation_inputs", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default_holdings_table() if key == "portfolio_df" else default


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="app-kicker">Portfolio Projection</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">Cycle-Aware Monte Carlo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Projects your portfolio two ways: conditioned on today\u2019s market cycle, '
    'and against unconditioned historical bootstrap sampling.</div>',
    unsafe_allow_html=True,
)

config = load_config()

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="section-label" style="margin-top:0;">Simulation</div>', unsafe_allow_html=True)

    n_years = st.number_input(
        "Time horizon (years)", min_value=1, max_value=60,
        value=int(config["simulation"].get("n_years", 30)), step=1,
    )
    n_trials = st.number_input(
        "Trials", min_value=100, max_value=100000,
        value=int(config["simulation"].get("n_trials", 10000)), step=1000,
    )

    st.markdown('<div class="section-label">Cash flows</div>', unsafe_allow_html=True)
    annual_contribution = st.number_input(
        "Annual deposit", min_value=0.0,
        value=float(config["portfolio"].get("annual_contribution", 0.0)), step=1000.0,
    )
    annual_withdrawal = st.number_input(
        "Annual withdrawal", min_value=0.0,
        value=float(config["portfolio"].get("annual_withdrawal", 0.0)), step=1000.0,
    )

    annual_fee = float(config["portfolio"].get("annual_fee", 0.0))
    contribution_years = int(config["portfolio"].get("contribution_years", 0))
    failure_threshold = float(config["portfolio"].get("failure_threshold", 0.0))
    target_end_balance = float(config["portfolio"].get("target_end_balance", 0.0))

    with st.expander("Advanced"):
        random_seed = st.number_input(
            "Random seed", min_value=0,
            value=int(config["simulation"].get("random_seed", 42)), step=1,
        )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    run_button = st.button("Run projection", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Holdings input
# -----------------------------------------------------------------------------
section_label("Holdings")

with st.expander("Load from CSV instead"):
    csv_path_input = st.text_input("CSV path (columns: ticker, shares)", value=DEFAULT_PORTFOLIO_CSV_PATH)
    if st.button("Load CSV"):
        loaded_df, load_error = load_holdings_from_csv(csv_path_input)
        if load_error:
            st.error(load_error)
        else:
            st.session_state.portfolio_df = loaded_df
            st.success(f"Loaded {len(loaded_df)} holding(s).")
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
    st.warning("Add at least one ticker with positive shares to run a projection.")


# -----------------------------------------------------------------------------
# Run simulation
# -----------------------------------------------------------------------------
def run_and_render():
    historical_data = pd.read_csv(
        "data/processed/regime_labeled_dataset.csv", index_col="date", parse_dates=True
    )

    analyzer = PortfolioAnalyzer()
    summary = analyzer.analyze(portfolio_dict)
    st.session_state.summary = summary

    regime_simulator = RegimeMultiAssetSimulator()
    st.session_state.regime_simulator = regime_simulator
    regime_results = regime_simulator.run_simulation(
        asset_class_weights=summary["asset_class_weights"],
        initial_balance=summary["total_value"],
        n_trials=int(n_trials), n_years=int(n_years),
        annual_contribution=float(annual_contribution),
        annual_withdrawal=float(annual_withdrawal),
        annual_fee=annual_fee, contribution_years=contribution_years,
        failure_threshold=failure_threshold, target_end_balance=target_end_balance,
        random_state=int(random_seed),
    )
    st.session_state.regime_results = regime_results

    bootstrap_data = historical_data[["equity_return", "bond_return"]].dropna()
    if len(bootstrap_data) == 0:
        raise ValueError("No valid historical data for bootstrap simulation")

    bootstrap_simulator = BootstrapMonteCarloSimulator(bootstrap_data)
    bootstrap_results = bootstrap_simulator.run_simulation(
        asset_class_weights=summary["asset_class_weights"],
        initial_balance=summary["total_value"],
        n_trials=int(n_trials), n_years=int(n_years),
        annual_contribution=float(annual_contribution),
        annual_withdrawal=float(annual_withdrawal),
        annual_fee=annual_fee, contribution_years=contribution_years,
        failure_threshold=failure_threshold, target_end_balance=target_end_balance,
        random_state=int(random_seed),
    )
    st.session_state.bootstrap_results = bootstrap_results
    st.session_state.explanation_inputs = {
        "asset_class_weights": summary["asset_class_weights"],
        "n_years": int(n_years),
    }

    # --- Portfolio summary ---
    section_label("Portfolio")
    alloc_df = pd.DataFrame(
        [{"Asset Class": k, "Weight": v} for k, v in summary["asset_class_weights"].items()]
    )
    sc1, sc2 = st.columns([1, 1.3])
    with sc1:
        st.markdown(stat_card("Total value", format_currency(summary["total_value"])), unsafe_allow_html=True)
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown(stat_card("Holdings", str(summary["num_holdings"])), unsafe_allow_html=True)
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.dataframe(
            alloc_df.assign(Weight=lambda d: (d["Weight"] * 100).round(1).astype(str) + "%"),
            use_container_width=True, hide_index=True,
        )
    with sc2:
        st.plotly_chart(make_allocation_chart(alloc_df), use_container_width=True, config={"displayModeBar": False})

    # --- Comparison ---
    section_label("Projection")
    col1, col2 = st.columns(2)
    with col1:
        render_result_panel(
            regime_results, "Cycle-aware", "Conditioned on today\u2019s market cycle", ACCENT, ACCENT_SOFT
        )
    with col2:
        render_result_panel(
            bootstrap_results, "Bootstrap", "Sampled uniformly across all of history", NEUTRAL, NEUTRAL_SOFT
        )

    # --- Explanation ---
    section_label("Interpretation")
    render_streamlit_explanation(
        regime_simulator=regime_simulator,
        asset_class_weights=summary["asset_class_weights"],
        regime_results=regime_results,
        bootstrap_results=bootstrap_results,
        n_years=int(n_years),
    )

    # --- Save outputs ---
    output_dir = Path(config.get("output", {}).get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    regime_summary = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in regime_results.items() if k not in ["all_balances", "all_regimes"]
    }
    bootstrap_summary = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in bootstrap_results.items() if k not in ["all_balances", "all_regimes"]
    }
    with open(output_dir / "regime_results.json", "w") as f:
        json.dump(regime_summary, f, indent=2)
    with open(output_dir / "bootstrap_results.json", "w") as f:
        json.dump(bootstrap_summary, f, indent=2)

    st.caption(f"Saved to {output_dir}/")


if run_button:
    if len(portfolio_dict) == 0:
        st.error("Add at least one valid holding before running.")
    else:
        with st.spinner("Running projections\u2026"):
            try:
                run_and_render()
            except Exception as e:
                st.exception(e)

elif st.session_state.regime_results is not None and st.session_state.bootstrap_results is not None:
    # Show cached results from the last run without re-simulating.
    regime_results = st.session_state.regime_results
    bootstrap_results = st.session_state.bootstrap_results

    section_label("Projection \u00b7 last run")
    col1, col2 = st.columns(2)
    with col1:
        render_result_panel(
            regime_results, "Cycle-aware", "Conditioned on today\u2019s market cycle", ACCENT, ACCENT_SOFT
        )
    with col2:
        render_result_panel(
            bootstrap_results, "Bootstrap", "Sampled uniformly across all of history", NEUTRAL, NEUTRAL_SOFT
        )

    if st.session_state.regime_simulator is not None and st.session_state.explanation_inputs is not None:
        section_label("Interpretation")
        render_streamlit_explanation(
            regime_simulator=st.session_state.regime_simulator,
            asset_class_weights=st.session_state.explanation_inputs["asset_class_weights"],
            regime_results=regime_results,
            bootstrap_results=bootstrap_results,
            n_years=st.session_state.explanation_inputs["n_years"],
        )