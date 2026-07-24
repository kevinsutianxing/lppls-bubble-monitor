import numpy as np

from lppls_monitor.backtest import _future_max_drawdown


def test_future_max_drawdown():
    prices = np.array([100.0, 110.0, 90.0, 95.0])
    assert np.isclose(_future_max_drawdown(prices), 90.0 / 110.0 - 1.0)
