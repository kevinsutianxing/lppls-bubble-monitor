from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class CalibrationConfig:
    """Numerical and economic constraints for one LPPLS fit."""

    m_bounds: tuple[float, float] = (0.10, 0.90)
    omega_bounds: tuple[float, float] = (6.0, 13.0)
    tc_min_ahead: float = 1.0
    tc_max_ahead: float = 252.0
    maxiter: int = 300
    popsize: int = 15
    seed: int = 42
    boundary_tolerance: float = 0.02
    min_oscillations: float = 2.5
    max_relative_rmse: float = 0.15
    require_damping: bool = True


@dataclass(frozen=True)
class ConfidenceConfig:
    """Multiscale-window settings for the LPPLS confidence indicator."""

    window_sizes: Sequence[int] = field(
        default_factory=lambda: (120, 180, 250, 360, 500, 750, 1000, 1250)
    )
    min_window: int = 120
    min_valid_windows: int = 3
    high_confidence: float = 0.60
    moderate_confidence: float = 0.35
    watch_confidence: float = 0.15
    max_tc_dispersion_days: float = 90.0


@dataclass(frozen=True)
class MonitorConfig:
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
