import numpy as np

from lppls_monitor.confidence import analyze_multiscale
from lppls_monitor.config import CalibrationConfig, ConfidenceConfig, MonitorConfig


def test_random_walk_is_not_promoted_to_bubble_signal():
    rng = np.random.default_rng(123)
    log_price = 5.0 + np.cumsum(rng.normal(0.0, 0.01, 500))
    config = MonitorConfig(
        calibration=CalibrationConfig(maxiter=50, popsize=8, seed=3),
        confidence=ConfidenceConfig(
            window_sizes=(120, 250, 500),
            min_valid_windows=2,
        ),
    )
    result = analyze_multiscale(log_price, config=config)
    assert result.risk_level in {"LOW", "WATCH"}
    assert result.positive_bubble_confidence < 0.35
