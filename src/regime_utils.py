"""
Shared utilities for regime classification and labeling.
"""

def classify_regime(row) -> tuple[str, str]:
    """
    Classify regime based on macro feature profile with hierarchical logic
    to avoid duplicate labels.
    
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

    # CRISIS/SHOCK: Severe stress across multiple indicators
    # High unemployment, widening credit spreads, weak growth
    if unemp > 0.5 and cspread > 0.5 and growth < -0.3:
        return "Crisis", "#D62728"
    
    # More lenient shock definition for smaller samples
    if (unemp > 1.0 or cspread > 1.0) and growth < 0:
        return "Shock", "#4FC3F7"
    
    # STAGFLATION: High inflation + weak growth + rising unemployment
    if cpi > 0.8 and growth < 0 and unemp > 0:
        return "Stagflation", "#FF6B6B"
    
    # INFLATIONARY: Elevated inflation, may have decent growth
    if cpi > 0.5:
        if growth > 0.3:
            return "Hot Economy", "#FF9800"
        else:
            return "Inflationary", "#FFC107"
    
    # GROWTH/EXPANSION: Strong growth, low inflation, improving labor
    if growth > 0.5 and unemp < -0.3 and cpi < 0.3:
        return "Strong Growth", "#4CAF50"
    
    # MODERATE GROWTH: Positive growth, stable conditions
    if growth > 0.2 and abs(cpi) < 0.5 and abs(unemp) < 0.3:
        return "Moderate Growth", "#8BC34A"
    
    # RECOVERY: Improving from weakness
    if growth > 0 and unemp < -0.3:
        return "Recovery", "#66BB6A"
    
    # LOW VOLATILITY / GOLDILOCKS: Everything calm
    if abs(growth) < 0.3 and abs(cpi) < 0.3 and abs(unemp) < 0.3:
        return "Low Volatility", "#FFD966"
    
    # Default: Balanced (should be rare now)
    return "Balanced", "#9E9E9E"