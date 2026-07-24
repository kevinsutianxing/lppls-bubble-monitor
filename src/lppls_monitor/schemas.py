from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class FitStatus(str, Enum):
    VALID = "VALID"
    BOUNDARY_SATURATED = "BOUNDARY_SATURATED"
    OPTIMIZER_FAILED = "OPTIMIZER_FAILED"
    NON_BUBBLE_SHAPE = "NON_BUBBLE_SHAPE"
    DAMPING_CONDITION_FAILED = "DAMPING_CONDITION_FAILED"
    INSUFFICIENT_OSCILLATIONS = "INSUFFICIENT_OSCILLATIONS"
    POOR_FIT = "POOR_FIT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class FitDiagnostics:
    boundary_parameters: list[str] = field(default_factory=list)
    oscillation_count: float | None = None
    damping: float | None = None
    relative_rmse: float | None = None
    adf_stat: float | None = None
    adf_pvalue: float | None = None
    lomb_power: float | None = None
    lomb_pvalue_approx: float | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class LPPLSFit:
    tc: float
    m: float
    omega: float
    A: float
    B: float
    C1: float
    C2: float
    C: float
    phi: float
    sse: float
    optimizer_success: bool
    optimizer_message: str
    status: FitStatus
    diagnostics: FitDiagnostics
    n_obs: int
    t_end: float

    @property
    def tc_days_ahead(self) -> float:
        return float(self.tc - self.t_end)

    @property
    def is_valid(self) -> bool:
        return self.status is FitStatus.VALID

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class WindowFit:
    window_size: int
    attempted: bool
    fit: LPPLSFit | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class MonitorResult:
    name: str
    risk_level: str
    risk_description: str
    positive_bubble_confidence: float
    valid_fit_ratio: float
    boundary_saturation_ratio: float
    n_attempted_windows: int
    n_valid_windows: int
    tc_median_days: float | None
    tc_p10_days: float | None
    tc_p90_days: float | None
    tc_std_days: float | None
    windows: list[WindowFit]
    latest_fit: LPPLSFit | None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
