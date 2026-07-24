from __future__ import annotations

import numpy as np

from .calibration import calibrate_lppls
from .config import MonitorConfig
from .schemas import FitStatus, MonitorResult, WindowFit


def analyze_multiscale(
    log_price: np.ndarray,
    name: str = "Unknown",
    config: MonitorConfig | None = None,
) -> MonitorResult:
    config = config or MonitorConfig()
    y = np.asarray(log_price, dtype=float)
    y = y[np.isfinite(y)]
    windows: list[WindowFit] = []

    for size in config.confidence.window_sizes:
        if size > len(y) or size < config.confidence.min_window:
            continue
        sample = y[-size:]
        try:
            fit = calibrate_lppls(sample, config=config.calibration)
            windows.append(WindowFit(window_size=size, attempted=True, fit=fit))
        except Exception as exc:
            windows.append(WindowFit(window_size=size, attempted=True, error=str(exc)))

    attempted = len(windows)
    valid = [w.fit for w in windows if w.fit is not None and w.fit.is_valid]
    boundary = [
        w.fit for w in windows
        if w.fit is not None and w.fit.status is FitStatus.BOUNDARY_SATURATED
    ]
    valid_ratio = len(valid) / attempted if attempted else 0.0
    boundary_ratio = len(boundary) / attempted if attempted else 0.0

    supported = [
        fit for fit in valid
        if fit.diagnostics.adf_pvalue is not None
        and np.isfinite(fit.diagnostics.adf_pvalue)
        and fit.diagnostics.adf_pvalue <= 0.05
        and fit.diagnostics.lomb_pvalue_approx is not None
        and np.isfinite(fit.diagnostics.lomb_pvalue_approx)
        and fit.diagnostics.lomb_pvalue_approx <= 0.10
    ]
    confidence = len(supported) / attempted if attempted else 0.0
    tc_days = np.asarray([fit.tc_days_ahead for fit in supported], dtype=float)
    tc_median = float(np.median(tc_days)) if tc_days.size else None
    tc_p10 = float(np.quantile(tc_days, 0.10)) if tc_days.size else None
    tc_p90 = float(np.quantile(tc_days, 0.90)) if tc_days.size else None
    tc_std = float(np.std(tc_days)) if tc_days.size else None

    enough = len(valid) >= config.confidence.min_valid_windows
    stable = tc_std is not None and tc_std <= config.confidence.max_tc_dispersion_days
    if enough and stable and confidence >= config.confidence.high_confidence:
        risk = "HIGH"
        description = "High multiscale LPPLS confidence with stable critical-time distribution"
    elif enough and confidence >= config.confidence.moderate_confidence:
        risk = "MODERATE"
        description = "Moderate multiscale LPPLS confidence; confirm with other risk evidence"
    elif confidence >= config.confidence.watch_confidence:
        risk = "WATCH"
        description = "Some valid LPPLS windows detected, but evidence is incomplete or unstable"
    elif attempted == 0:
        risk = "ERROR"
        description = "No eligible analysis windows"
    else:
        risk = "LOW"
        description = "No robust multiscale positive-bubble signature"

    latest_fit = windows[0].fit if windows and windows[0].fit is not None else None
    return MonitorResult(
        name=name,
        risk_level=risk,
        risk_description=description,
        positive_bubble_confidence=float(confidence),
        valid_fit_ratio=float(valid_ratio),
        boundary_saturation_ratio=float(boundary_ratio),
        n_attempted_windows=attempted,
        n_valid_windows=len(valid),
        tc_median_days=tc_median,
        tc_p10_days=tc_p10,
        tc_p90_days=tc_p90,
        tc_std_days=tc_std,
        windows=windows,
        latest_fit=latest_fit,
    )
