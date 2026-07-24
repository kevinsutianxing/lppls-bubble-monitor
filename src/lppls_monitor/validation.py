from __future__ import annotations

import math

import numpy as np

from .config import CalibrationConfig
from .schemas import FitDiagnostics, FitStatus


def _near_boundary(value: float, bounds: tuple[float, float], tolerance: float) -> bool:
    lo, hi = bounds
    span = hi - lo
    if span <= 0:
        return True
    return (value - lo) / span <= tolerance or (hi - value) / span <= tolerance


def validate_fit(
    *,
    t: np.ndarray,
    log_price: np.ndarray,
    tc: float,
    m: float,
    omega: float,
    B: float,
    C: float,
    residuals: np.ndarray,
    optimizer_success: bool,
    config: CalibrationConfig,
    adf_stat: float | None = None,
    adf_pvalue: float | None = None,
    lomb_power: float | None = None,
    lomb_pvalue_approx: float | None = None,
) -> tuple[FitStatus, FitDiagnostics]:
    diagnostics = FitDiagnostics(
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        lomb_power=lomb_power,
        lomb_pvalue_approx=lomb_pvalue_approx,
    )
    t_end = float(np.max(t))
    tc_bounds = (t_end + config.tc_min_ahead, t_end + config.tc_max_ahead)

    for name, value, bounds in (
        ("tc", tc, tc_bounds),
        ("m", m, config.m_bounds),
        ("omega", omega, config.omega_bounds),
    ):
        if _near_boundary(value, bounds, config.boundary_tolerance):
            diagnostics.boundary_parameters.append(name)

    dt_ratio = max((tc - float(np.min(t))) / max(tc - t_end, 1e-12), 1.0)
    diagnostics.oscillation_count = float(omega / (2.0 * math.pi) * math.log(dt_ratio))
    diagnostics.damping = float(m * abs(B) / max(omega * abs(B) * abs(C), 1e-12))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    scale = max(float(np.ptp(log_price)), 1e-8)
    diagnostics.relative_rmse = rmse / scale

    if not optimizer_success:
        diagnostics.reasons.append("optimizer did not report convergence")
        return FitStatus.OPTIMIZER_FAILED, diagnostics
    if diagnostics.boundary_parameters:
        diagnostics.reasons.append(
            "solution is too close to search boundary: " + ", ".join(diagnostics.boundary_parameters)
        )
        return FitStatus.BOUNDARY_SATURATED, diagnostics
    if B >= 0 or not np.isfinite(C) or abs(C) >= 1.0:
        diagnostics.reasons.append("parameters do not describe a positive bubble regime")
        return FitStatus.NON_BUBBLE_SHAPE, diagnostics
    if config.require_damping and diagnostics.damping < 1.0:
        diagnostics.reasons.append("LPPLS damping condition is not satisfied")
        return FitStatus.DAMPING_CONDITION_FAILED, diagnostics
    if diagnostics.oscillation_count < config.min_oscillations:
        diagnostics.reasons.append("too few log-periodic oscillations in the window")
        return FitStatus.INSUFFICIENT_OSCILLATIONS, diagnostics
    if diagnostics.relative_rmse > config.max_relative_rmse:
        diagnostics.reasons.append("relative RMSE exceeds configured threshold")
        return FitStatus.POOR_FIT, diagnostics
    return FitStatus.VALID, diagnostics
