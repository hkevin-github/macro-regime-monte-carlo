"""
Shared utilities for regime classification and labeling.
"""

def classify_regime(row) -> tuple[str, str]:
    """
    Classify regime based on macro feature profile.
    
    Args:
        row: Series with z-scored macro features
        
    Returns:
        (label, color) tuple
    """
    growth = row["growth_yoy_zscore"]
    cpi = row["cpi_yoy_zscore"]
    unemp = row["unemployment_delta_zscore"]
    yspread = row["yield_spread_delta_zscore"]
    cspread = row["credit_spread_delta_zscore"]

    # Strong growth, improving labor, tighter spreads
    if growth > 0.5 and unemp < 0 and yspread < 0 and cspread < 0:
        return "Growth", "#2CA02C"

    # Weak growth, low inflation, worsening labor
    if growth < -0.5 and cpi < 0 and unemp > 0:
        return "Shock", "#4FC3F7"

    # Weak growth, elevated inflation, widening spreads
    if growth < 0 and cpi > 0.5 and (yspread > 0 or cspread > 0):
        return "Inflationary", "#D62728"

    # Default middle regime
    return "Balanced", "#FFD966"