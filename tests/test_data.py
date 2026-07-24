import numpy as np
import pandas as pd

from lppls_monitor.data import (
    build_adjusted_close,
    build_qfq_from_pctchg,
    frame_fingerprint,
    normalize_daily_frame,
)


def test_qfq_reconstruction_preserves_returns_and_anchor():
    close = np.array([10.0, 10.5, 11.025])
    pct = np.array([0.0, 5.0, 5.0])
    qfq = build_qfq_from_pctchg(close, pct)
    assert np.isclose(qfq[-1], close[-1])
    assert np.allclose(qfq[1:] / qfq[:-1] - 1.0, [0.05, 0.05])


def test_normalization_deduplicates_and_hash_is_stable():
    raw = pd.DataFrame(
        {
            "trade_date": ["20260102", "20260101", "20260102"],
            "close": [11, 10, 11.1],
            "pct_chg": [10, 0, 11],
        }
    )
    normalized = normalize_daily_frame(raw)
    assert len(normalized) == 2
    assert normalized.iloc[-1]["close"] == 11.1
    assert frame_fingerprint(normalized) == frame_fingerprint(normalized.copy())


def test_official_adjustment_factor_is_preferred():
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "close": [10.0, 11.0],
            "adj_factor": [2.0, 2.2],
        }
    )
    adjusted, method = build_adjusted_close(frame)
    assert method == "adj_factor"
    assert np.isclose(adjusted[-1], 11.0)
