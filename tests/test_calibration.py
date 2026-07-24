import numpy as np

from lppls_monitor.calibration import calibrate_lppls
from lppls_monitor.config import CalibrationConfig
from lppls_monitor.model import lppls_values
from lppls_monitor.schemas import FitStatus


def test_synthetic_lppls_fit_recovers_interior_solution():
    rng = np.random.default_rng(7)
    t = np.arange(1, 361, dtype=float)
    y = lppls_values(
        t,
        tc=420.0,
        m=0.45,
        omega=8.5,
        A=5.0,
        B=-0.30,
        C1=0.008,
        C2=-0.006,
    ) + rng.normal(0, 0.001, len(t))
    fit = calibrate_lppls(
        y,
        t=t,
        config=CalibrationConfig(maxiter=100, popsize=10, seed=7, max_relative_rmse=0.20),
    )
    assert fit.status is FitStatus.VALID
    assert abs(fit.tc - 420.0) < 30.0
    assert abs(fit.m - 0.45) < 0.20
    assert abs(fit.omega - 8.5) < 1.5
