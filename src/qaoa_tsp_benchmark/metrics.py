from __future__ import annotations

import math

import numpy as np


def optimality_gap_pct(cost: float, optimal: float | None) -> float | None:
    if optimal is None or not np.isfinite(optimal) or optimal <= 0:
        return None
    return ((cost - optimal) / optimal) * 100.0


def scaling_exponent(x: np.ndarray, y: np.ndarray) -> float | None:
    """Fit y ~= c * x^k and return k in log-space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.count_nonzero(mask) < 2:
        return None
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    slope, _intercept = np.polyfit(lx, ly, 1)
    if math.isnan(slope):
        return None
    return float(slope)
