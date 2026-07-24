from __future__ import annotations

import numpy as np


def design_matrix(t: np.ndarray, tc: float, m: float, omega: float) -> np.ndarray:
    """Return the linearized LPPLS design matrix [1, f, g, h]."""
    dt = tc - np.asarray(t, dtype=float)
    if np.any(dt <= 0):
        raise ValueError("tc must be greater than every observation time")
    f = np.power(dt, m)
    log_dt = np.log(dt)
    g = f * np.cos(omega * log_dt)
    h = f * np.sin(omega * log_dt)
    return np.column_stack((np.ones_like(f), f, g, h))


def solve_linear_parameters(
    t: np.ndarray,
    log_price: np.ndarray,
    tc: float,
    m: float,
    omega: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve A, B, C1, C2 by least squares for fixed tc, m, omega."""
    x = design_matrix(t, tc, m, omega)
    params, _, _, _ = np.linalg.lstsq(x, np.asarray(log_price, dtype=float), rcond=None)
    fitted = x @ params
    residuals = np.asarray(log_price, dtype=float) - fitted
    sse = float(residuals @ residuals)
    return params, residuals, sse


def lppls_values(
    t: np.ndarray,
    tc: float,
    m: float,
    omega: float,
    A: float,
    B: float,
    C1: float,
    C2: float,
) -> np.ndarray:
    x = design_matrix(np.asarray(t, dtype=float), tc, m, omega)
    return x @ np.asarray([A, B, C1, C2], dtype=float)


def oscillation_amplitude(B: float, C1: float, C2: float) -> float:
    """Return the conventional LPPLS relative oscillation amplitude C."""
    if abs(B) < 1e-15:
        return float("inf")
    return float(np.hypot(C1, C2) / abs(B))


def phase(C1: float, C2: float) -> float:
    return float(np.arctan2(-C2, C1))
