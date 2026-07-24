from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle
from statsmodels.tsa.stattools import adfuller


def adf_residual_test(residuals: np.ndarray) -> tuple[float, float]:
    values = np.asarray(residuals, dtype=float)
    if values.size < 20 or np.allclose(values, values[0]):
        return float("nan"), float("nan")
    result = adfuller(values, autolag="BIC")
    return float(result[0]), float(result[1])


def lomb_log_periodic_test(
    t: np.ndarray,
    residuals: np.ndarray,
    tc: float,
    omega: float,
) -> tuple[float, float]:
    """Approximate Lomb significance around the fitted log frequency.

    The p-value is a conservative exponential-tail approximation and is used as
    a diagnostic, not as a standalone formal test.
    """
    dt = tc - np.asarray(t, dtype=float)
    if np.any(dt <= 0) or len(dt) < 20:
        return float("nan"), float("nan")
    x = np.log(dt)
    y = np.asarray(residuals, dtype=float) - float(np.mean(residuals))
    if np.allclose(y, 0):
        return 0.0, 1.0
    freqs = np.linspace(max(0.5, omega - 2.0), omega + 2.0, 256)
    power = lombscargle(x, y, freqs, normalize=True)
    peak = float(np.max(power))
    pvalue = float(min(1.0, np.exp(-max(peak, 0.0) * len(y) / 2.0)))
    return peak, pvalue
