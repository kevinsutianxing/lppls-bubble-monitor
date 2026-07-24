from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .config import CalibrationConfig
from .diagnostics import adf_residual_test, lomb_log_periodic_test
from .model import oscillation_amplitude, phase, solve_linear_parameters
from .schemas import FitDiagnostics, FitStatus, LPPLSFit
from .validation import validate_fit


def _objective(theta: np.ndarray, t: np.ndarray, log_price: np.ndarray) -> float:
    tc, m, omega = map(float, theta)
    try:
        linear, _, sse = solve_linear_parameters(t, log_price, tc, m, omega)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return 1e30
    A, B, C1, C2 = linear
    C = oscillation_amplitude(B, C1, C2)
    if not np.isfinite(sse) or not np.all(np.isfinite(linear)):
        return 1e30
    penalty = 0.0
    if B >= 0:
        penalty += 1e5 * (1.0 + B * B)
    if not np.isfinite(C) or C >= 1.0:
        penalty += 1e5 * (1.0 + min(C if np.isfinite(C) else 10.0, 10.0) ** 2)
    return float(sse + penalty)


def calibrate_lppls(
    log_price: np.ndarray,
    t: np.ndarray | None = None,
    config: CalibrationConfig | None = None,
) -> LPPLSFit:
    config = config or CalibrationConfig()
    y = np.asarray(log_price, dtype=float)
    if t is None:
        t = np.arange(1, len(y) + 1, dtype=float)
    else:
        t = np.asarray(t, dtype=float)
    if y.ndim != 1 or len(y) != len(t) or len(y) < 60:
        diagnostics = FitDiagnostics(reasons=["at least 60 finite observations are required"])
        return LPPLSFit(
            tc=float("nan"), m=float("nan"), omega=float("nan"),
            A=float("nan"), B=float("nan"), C1=float("nan"), C2=float("nan"),
            C=float("nan"), phi=float("nan"), sse=float("nan"),
            optimizer_success=False, optimizer_message="insufficient data",
            status=FitStatus.INSUFFICIENT_DATA, diagnostics=diagnostics,
            n_obs=len(y), t_end=float(t[-1]) if len(t) else float("nan"),
        )
    finite = np.isfinite(y) & np.isfinite(t)
    y = y[finite]
    t = t[finite]
    t_end = float(t[-1])
    bounds = [
        (t_end + config.tc_min_ahead, t_end + config.tc_max_ahead),
        config.m_bounds,
        config.omega_bounds,
    ]
    de = differential_evolution(
        _objective,
        bounds=bounds,
        args=(t, y),
        maxiter=config.maxiter,
        popsize=config.popsize,
        seed=config.seed,
        tol=1e-8,
        polish=False,
        updating="immediate",
        workers=1,
    )
    local = minimize(
        _objective,
        x0=de.x,
        args=(t, y),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if local.success and (not de.success or local.fun <= de.fun):
        chosen = local
    elif de.success:
        chosen = de
    else:
        chosen = local if local.fun <= de.fun else de
    tc, m, omega = map(float, chosen.x)
    linear, residuals, sse = solve_linear_parameters(t, y, tc, m, omega)
    A, B, C1, C2 = map(float, linear)
    C = oscillation_amplitude(B, C1, C2)
    adf_stat, adf_pvalue = adf_residual_test(residuals)
    power_trend = A + B * np.power(tc - t, m)
    log_periodic_component = y - power_trend
    lomb_power, lomb_pvalue = lomb_log_periodic_test(t, log_periodic_component, tc, omega)
    optimizer_success = bool(chosen.success)
    status, diagnostics = validate_fit(
        t=t,
        log_price=y,
        tc=tc,
        m=m,
        omega=omega,
        B=B,
        C=C,
        residuals=residuals,
        optimizer_success=optimizer_success,
        config=config,
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        lomb_power=lomb_power,
        lomb_pvalue_approx=lomb_pvalue,
    )
    return LPPLSFit(
        tc=tc,
        m=m,
        omega=omega,
        A=A,
        B=B,
        C1=C1,
        C2=C2,
        C=C,
        phi=phase(C1, C2),
        sse=sse,
        optimizer_success=optimizer_success,
        optimizer_message=str(chosen.message),
        status=status,
        diagnostics=diagnostics,
        n_obs=len(y),
        t_end=t_end,
    )
