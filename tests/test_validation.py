import numpy as np

from lppls_monitor.config import CalibrationConfig
from lppls_monitor.schemas import FitStatus
from lppls_monitor.validation import validate_fit


def test_boundary_solution_is_rejected_before_risk_classification():
    t = np.arange(1, 301, dtype=float)
    y = np.linspace(4.0, 5.0, len(t))
    status, diagnostics = validate_fit(
        t=t,
        log_price=y,
        tc=t[-1] + 252.0,
        m=0.5,
        omega=8.0,
        B=-0.3,
        C=0.02,
        residuals=np.zeros_like(y),
        optimizer_success=True,
        config=CalibrationConfig(),
    )
    assert status is FitStatus.BOUNDARY_SATURATED
    assert "tc" in diagnostics.boundary_parameters
