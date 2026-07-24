from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .confidence import analyze_multiscale
from .config import MonitorConfig


@dataclass(frozen=True)
class NullSimulationResult:
    n_trials: int
    high_rate: float
    moderate_or_higher_rate: float
    mean_confidence: float
    confidence_p95: float


def simulate_null_false_positive_rate(
    n_trials: int = 100,
    n_obs: int = 750,
    drift: float = 0.0003,
    volatility: float = 0.015,
    seed: int = 42,
    config: MonitorConfig | None = None,
) -> NullSimulationResult:
    rng = np.random.default_rng(seed)
    levels: list[str] = []
    confidences: list[float] = []
    for _ in range(n_trials):
        log_returns = rng.normal(drift, volatility, n_obs)
        result = analyze_multiscale(np.cumsum(log_returns), config=config)
        levels.append(result.risk_level)
        confidences.append(result.positive_bubble_confidence)
    high = np.mean([level == "HIGH" for level in levels])
    moderate = np.mean([level in {"HIGH", "MODERATE"} for level in levels])
    return NullSimulationResult(
        n_trials=n_trials,
        high_rate=float(high),
        moderate_or_higher_rate=float(moderate),
        mean_confidence=float(np.mean(confidences)),
        confidence_p95=float(np.quantile(confidences, 0.95)),
    )
