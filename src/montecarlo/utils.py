"""
Monte Carlo utilities.
"""

import numpy as np


def yoy_to_monthly(rate_yoy: float) -> float:
    """
    Convert a year-over-year rate to an equivalent monthly rate.

    Example:
        0.024 YoY -> ~0.00198 monthly
    """
    rate_yoy = max(rate_yoy, -0.99)
    return (1.0 + rate_yoy) ** (1.0 / 12.0) - 1.0


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Numerically safe log."""
    return np.log(np.maximum(x, eps))