from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .confidence import analyze_multiscale
from .config import MonitorConfig


@dataclass(frozen=True)
class WalkForwardConfig:
    min_history: int = 500
    step: int = 20
    horizons: tuple[int, ...] = (20, 60, 120, 250)


def _future_max_drawdown(prices: np.ndarray) -> float:
    running_max = np.maximum.accumulate(prices)
    drawdowns = prices / running_max - 1.0
    return float(np.min(drawdowns))


def walk_forward_backtest(
    adjusted_prices: np.ndarray,
    monitor_config: MonitorConfig | None = None,
    backtest_config: WalkForwardConfig | None = None,
) -> pd.DataFrame:
    monitor_config = monitor_config or MonitorConfig()
    cfg = backtest_config or WalkForwardConfig()
    prices = np.asarray(adjusted_prices, dtype=float)
    rows: list[dict] = []
    max_horizon = max(cfg.horizons)
    for end in range(cfg.min_history, len(prices) - max_horizon, cfg.step):
        result = analyze_multiscale(np.log(prices[:end]), config=monitor_config)
        row = {
            "end_index": end - 1,
            "risk_level": result.risk_level,
            "confidence": result.positive_bubble_confidence,
            "valid_fit_ratio": result.valid_fit_ratio,
            "tc_median_days": result.tc_median_days,
        }
        current = prices[end - 1]
        for horizon in cfg.horizons:
            future = prices[end : end + horizon]
            row[f"return_{horizon}d"] = float(future[-1] / current - 1.0)
            row[f"max_drawdown_{horizon}d"] = _future_max_drawdown(
                np.concatenate([[current], future])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_backtest(
    frame: pd.DataFrame,
    signal_levels: tuple[str, ...] = ("HIGH", "MODERATE"),
    drawdown_horizon: int = 120,
    drawdown_threshold: float = -0.15,
) -> dict[str, float]:
    if frame.empty:
        return {"precision": float("nan"), "recall": float("nan"), "alert_rate": 0.0}
    signal = frame["risk_level"].isin(signal_levels)
    event = frame[f"max_drawdown_{drawdown_horizon}d"] <= drawdown_threshold
    true_positive = int((signal & event).sum())
    precision = true_positive / int(signal.sum()) if signal.any() else 0.0
    recall = true_positive / int(event.sum()) if event.any() else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "alert_rate": float(signal.mean()),
        "event_rate": float(event.mean()),
    }
