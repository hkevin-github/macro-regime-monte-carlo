"""
Regime Explainer
-----------------
Generates a short, human-readable explanation of *why* the regime-aware
Monte Carlo simulation produced results that differ from the
bootstrap/standard historical simulation for this specific portfolio and run.

The core driver of the divergence is that the regime-aware simulator seeds
every trial from the *current* regime-probability distribution (i.e. "what
does the economy look like right now"), then lets the Markov chain evolve
from there -- whereas the bootstrap simulator draws blocks/returns uniformly
across all of history, with no notion of where we are today.

So:
  - If we're currently sitting in a strong, low-inflation growth regime, and
    the portfolio is growth-asset heavy (equities), regime-aware will tend to
    look *better* than bootstrap over the horizon, especially in the near
    term before mean-reversion (via the transition matrix) pulls the mix back
    toward the long-run average.
  - If we're currently in a Crisis/Stagflation/Shock-like regime, regime-aware
    will tend to look *more conservative* than bootstrap near-term.
  - The size of the effect scales with (a) how far the current regime's
    expected return is from the long-run (stationary) blended return, and
    (b) how much of the portfolio is in regime-sensitive growth assets
    (equities, intl equities, real estate, commodities) vs. ballast
    (bonds, cash).

This module inspects the loaded regime model + the portfolio's asset class
weights + the two simulation results, and produces:
  - a structured dict of the diagnostic numbers (useful for tests / debugging)
  - a short natural-language paragraph for display in the dashboard
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# Asset classes treated as "growth / regime-sensitive" vs "ballast".
# Mirrors RegimeMultiAssetSimulator.ASSET_CLASS_MAPPING but collapsed into
# two buckets for the purposes of the narrative explanation.
GROWTH_ASSET_CLASSES = {
    "US_Equity",
    "International_Equity",
    "Emerging_Markets_Equity",
    "Real_Estate",
    "Commodities",
}
BALLAST_ASSET_CLASSES = {
    "US_Bonds",
    "International_Bonds",
    "Cash",
}


def _growth_weight(asset_class_weights: Dict[str, float]) -> float:
    """Fraction of the portfolio sitting in regime-sensitive growth assets."""
    return sum(
        w for ac, w in asset_class_weights.items() if ac in GROWTH_ASSET_CLASSES
    )


def _annualized(monthly_mean: float) -> float:
    """Convert a monthly arithmetic mean return to an approximate annualized rate."""
    return (1.0 + monthly_mean) ** 12 - 1.0


def compute_regime_diagnostics(
    regime_simulator,
    asset_class_weights: Dict[str, float],
) -> dict:
    """
    Pull the numbers needed for the narrative out of an already-constructed
    RegimeMultiAssetSimulator (so we reuse its loaded regime_stats,
    regime_labels, transition_matrix -- no re-loading from disk).

    Returns a dict of diagnostics; safe to call even if some pieces are
    missing (falls back gracefully with None values the caller can check).
    """
    assumptions = regime_simulator.assumptions
    regime_names = regime_simulator.regime_labels
    n_regimes = assumptions.n_regimes

    # --- current ("as of today") regime distribution -----------------------
    try:
        current_probs = assumptions.get_current_regime_probabilities()
    except Exception:
        current_probs = None

    # --- long-run (stationary) regime distribution --------------------------
    try:
        stationary_probs = assumptions.get_stationary_distribution()
    except Exception:
        stationary_probs = None

    growth_w = _growth_weight(asset_class_weights)
    ballast_w = 1.0 - growth_w

    # Portfolio-weighted monthly return for a given regime, using each asset
    # class's *actual* weight and the simulator's own per-asset-class param
    # lookup (so bonds use bond stats/fallback proxy, cash is 0, equities use
    # equity stats, etc.) -- NOT hardcoded to equity. This is what makes the
    # explanation correct for bond-heavy, cash-heavy, or mixed portfolios,
    # not just equity-heavy ones.
    def portfolio_monthly_mean(regime: int) -> float:
        total = 0.0
        for asset_class, weight in asset_class_weights.items():
            params = regime_simulator.get_asset_class_params(asset_class, regime)
            total += weight * params["mean"]
        return total

    def blended_portfolio_mean(probs: np.ndarray) -> float:
        return float(sum(
            probs[r] * portfolio_monthly_mean(r) for r in range(n_regimes)
        ))

    current_blend = (
        blended_portfolio_mean(current_probs) if current_probs is not None else None
    )
    stationary_blend = (
        blended_portfolio_mean(stationary_probs) if stationary_probs is not None else None
    )

    # Most likely regime right now (argmax of current distribution)
    dominant_regime_id: Optional[int] = None
    dominant_regime_name: Optional[str] = None
    dominant_regime_prob: Optional[float] = None
    if current_probs is not None:
        dominant_regime_id = int(np.argmax(current_probs))
        dominant_regime_name = regime_names.get(dominant_regime_id, f"Regime {dominant_regime_id}")
        dominant_regime_prob = float(current_probs[dominant_regime_id])

    # Which asset class actually dominates the portfolio -- used to phrase
    # the explanation in terms of the right asset ("bond returns" vs "equity
    # returns") rather than always saying "equity".
    dominant_asset_class = None
    if asset_class_weights:
        dominant_asset_class = max(asset_class_weights.items(), key=lambda kv: kv[1])[0]

    diagnostics = {
        "growth_weight": growth_w,
        "ballast_weight": ballast_w,
        "current_probs": current_probs,
        "stationary_probs": stationary_probs,
        "dominant_regime_id": dominant_regime_id,
        "dominant_regime_name": dominant_regime_name,
        "dominant_regime_prob": dominant_regime_prob,
        "dominant_asset_class": dominant_asset_class,
        "current_monthly_portfolio_mean": current_blend,
        "stationary_monthly_portfolio_mean": stationary_blend,
        "current_annual_portfolio_mean": _annualized(current_blend) if current_blend is not None else None,
        "stationary_annual_portfolio_mean": _annualized(stationary_blend) if stationary_blend is not None else None,
        "regime_names": regime_names,
    }

    if current_blend is not None and stationary_blend is not None:
        diagnostics["annual_tilt"] = (
            diagnostics["current_annual_portfolio_mean"] - diagnostics["stationary_annual_portfolio_mean"]
        )
    else:
        diagnostics["annual_tilt"] = None

    return diagnostics


def _fmt_pct(x: Optional[float], decimals: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.{decimals}f}%"


def explain_regime_vs_bootstrap(
    diagnostics: dict,
    regime_results: dict,
    bootstrap_results: dict,
    n_years: int,
) -> str:
    """
    Turn the diagnostics + the two simulation result dicts into a short
    natural-language paragraph, in the style of:

      "Regime-aware was [more optimistic / more conservative / similar]
       than the standard simulation because ..."

    Falls back to a generic, still-accurate explanation of the mechanism
    if the regime model data needed for the numeric tilt isn't available.
    """
    regime_median = regime_results.get("median_final")
    bootstrap_median = bootstrap_results.get("median_final") or float(
        np.median(bootstrap_results["final_balances"])
    )

    if regime_median is None:
        regime_median = float(np.median(regime_results["final_balances"]))

    pct_diff = None
    if bootstrap_median:
        pct_diff = (regime_median - bootstrap_median) / abs(bootstrap_median)

    growth_w = diagnostics.get("growth_weight")
    dominant_regime_name = diagnostics.get("dominant_regime_name")
    dominant_regime_prob = diagnostics.get("dominant_regime_prob")
    dominant_asset_class = diagnostics.get("dominant_asset_class")
    annual_tilt = diagnostics.get("annual_tilt")
    current_ann = diagnostics.get("current_annual_portfolio_mean")
    stationary_ann = diagnostics.get("stationary_annual_portfolio_mean")

    # Decide the headline direction from the actual simulation outputs
    # (ground truth), not just the theoretical tilt.
    if pct_diff is not None and abs(pct_diff) >= 0.02:
        direction = "more optimistic" if pct_diff > 0 else "more conservative"
    else:
        direction = "similar to"

    # --- Build the explanation (kept to ~2 short sentences) -------------------
    if dominant_regime_name is None or annual_tilt is None:
        # Fallback: mechanism-only explanation, no numeric tilt available.
        connector = "than" if direction != "similar to" else "to"
        return (
            f"**Regime-aware was {direction} {connector} bootstrap** because it conditions on "
            f"today's market regime instead of averaging over all of history. "
            f"{'This portfolio is growth-heavy, so that difference gets amplified.' if (growth_w or 0) >= 0.6 else ''}"
        ).strip()

    tilt_word = "higher than" if annual_tilt > 0 else "lower than"
    if abs(annual_tilt) < 0.003:
        tilt_word = "in line with"

    connector = "than" if direction != "similar to" else "to"
    diff_clause = f" (median differed by {_fmt_pct(pct_diff, 0)})" if pct_diff is not None else ""

    # Describe returns in terms of "this portfolio's holdings" rather than
    # always saying "equity" -- correct whether the portfolio is bond-heavy,
    # cash-heavy, equity-heavy, or a mix.
    holdings_phrase = "this portfolio's holdings"
    if dominant_asset_class:
        readable = dominant_asset_class.replace("_", " ")
        holdings_phrase = f"assets like {readable}"

    sentence1 = (
        f"**Regime-aware was {direction} {connector} bootstrap**{diff_clause} because the model "
        f"currently sees a **\"{dominant_regime_name}\"** regime (~{_fmt_pct(dominant_regime_prob, 0)} confidence) "
        f"implying ~{_fmt_pct(current_ann, 1)}/yr returns for {holdings_phrase}, {tilt_word} the "
        f"{_fmt_pct(stationary_ann, 1)}/yr long-run average bootstrap assumes."
    )

    # Only add the growth-weight sentence when it tells the reader something
    # they can't already infer from sentence 1 -- i.e. when the portfolio is
    # a genuine mix of growth and ballast assets. Pure bond/cash or pure
    # equity portfolios already fully explain themselves via the returns
    # figure above, so a "0% in growth assets, so this mutes the tilt" line
    # just restates the obvious.
    sentence2 = ""
    if growth_w is not None and 0.15 < growth_w < 0.85:
        strength = "amplifies" if growth_w >= 0.6 else "softens"
        sentence2 = f" This portfolio is a mix (~{_fmt_pct(growth_w, 0)} growth assets), which {strength} that effect somewhat."

    return sentence1 + sentence2


def render_streamlit_explanation(
    regime_simulator,
    asset_class_weights: Dict[str, float],
    regime_results: dict,
    bootstrap_results: dict,
    n_years: int,
) -> None:
    """
    Convenience wrapper for use inside streamlit_app.py:
    computes diagnostics, builds the explanation text, and renders it with
    st.info() plus an optional expandable "why" detail table.
    Import streamlit lazily so this module stays importable/testable without
    a streamlit runtime.
    """
    import streamlit as st

    diagnostics = compute_regime_diagnostics(regime_simulator, asset_class_weights)
    explanation = explain_regime_vs_bootstrap(
        diagnostics, regime_results, bootstrap_results, n_years
    )

    st.markdown("### 🧭 Why do these two simulations differ?")
    st.info(explanation)

    with st.expander("See the underlying regime diagnostics", expanded=False):
        current_probs = diagnostics.get("current_probs")
        stationary_probs = diagnostics.get("stationary_probs")
        regime_names = diagnostics.get("regime_names", {})

        if current_probs is not None and stationary_probs is not None:
            import pandas as pd

            n_regimes = len(current_probs)
            df = pd.DataFrame(
                {
                    "Regime": [regime_names.get(i, f"Regime {i}") for i in range(n_regimes)],
                    "Current Probability": [f"{p:.1%}" for p in current_probs],
                    "Long-Run (Stationary) Probability": [f"{p:.1%}" for p in stationary_probs],
                }
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

        cols = st.columns(3)
        cols[0].metric(
            "Portfolio Growth-Asset Weight",
            f"{diagnostics['growth_weight']:.0%}",
            help="Share of the portfolio in equities, intl equities, real estate, and commodities "
                 "-- the asset classes whose return assumptions vary most by regime.",
        )
        cols[1].metric(
            "Current-Regime Implied Annual Return",
            _fmt_pct(diagnostics.get("current_annual_portfolio_mean"), 1),
            help="Annualized return implied by today's most likely regime, weighted by this "
                 "portfolio's actual asset class mix (equities, bonds, cash, etc.).",
        )
        cols[2].metric(
            "Long-Run Blended Annual Return",
            _fmt_pct(diagnostics.get("stationary_annual_portfolio_mean"), 1),
            help="Same portfolio-weighted return, but blended across the long-run (stationary) "
                 "distribution of all regimes -- roughly what the bootstrap simulation reflects.",
        )