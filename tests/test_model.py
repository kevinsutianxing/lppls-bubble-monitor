import numpy as np

from lppls_monitor.model import lppls_values, solve_linear_parameters


def test_linear_parameter_recovery_without_noise():
    t = np.arange(1, 301, dtype=float)
    expected = dict(tc=360.0, m=0.45, omega=8.5, A=5.0, B=-0.30, C1=0.008, C2=-0.006)
    y = lppls_values(t, **expected)
    linear, residuals, sse = solve_linear_parameters(
        t, y, expected["tc"], expected["m"], expected["omega"]
    )
    assert np.allclose(linear, [expected["A"], expected["B"], expected["C1"], expected["C2"]])
    assert np.max(np.abs(residuals)) < 1e-10
    assert sse < 1e-18
