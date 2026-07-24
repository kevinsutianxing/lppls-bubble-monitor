from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class SystemicExposure:
    weighted_confidence: float
    valid_market_coverage: float
    high_risk_market_weight: float
    tc_near_term_weight: float
    tc_concentration: float
    n_assets: int

    def to_dict(self) -> dict:
        return {
            "weighted_confidence": self.weighted_confidence,
            "valid_market_coverage": self.valid_market_coverage,
            "high_risk_market_weight": self.high_risk_market_weight,
            "tc_near_term_weight": self.tc_near_term_weight,
            "tc_concentration": self.tc_concentration,
            "n_assets": self.n_assets,
        }


def aggregate_systemic_exposure(
    results: Iterable[Mapping],
    market_weights: Mapping[str, float],
    near_term_days: float = 120.0,
    tc_bin_days: float = 30.0,
) -> SystemicExposure:
    rows = list(results)
    raw_weights = np.asarray([
        max(float(market_weights.get(str(row.get("code")), 0.0)), 0.0) for row in rows
    ])
    total = float(raw_weights.sum())
    weights = raw_weights / total if total > 0 else np.zeros_like(raw_weights)
    confidence = np.asarray([float(row.get("positive_bubble_confidence", 0.0)) for row in rows])
    valid_ratio = np.asarray([float(row.get("valid_fit_ratio", 0.0)) for row in rows])
    high = np.asarray([str(row.get("risk_level", "")) == "HIGH" for row in rows], dtype=float)
    tc = np.asarray([
        np.nan if row.get("tc_median_days") is None else float(row.get("tc_median_days"))
        for row in rows
    ])

    weighted_confidence = float(np.sum(weights * confidence))
    valid_coverage = float(np.sum(weights * (valid_ratio > 0)))
    high_weight = float(np.sum(weights * high))
    near_term = float(np.sum(weights * confidence * np.isfinite(tc) * (tc <= near_term_days)))

    finite = np.isfinite(tc) & (weights > 0) & (confidence > 0)
    if finite.any():
        bins = np.floor(tc[finite] / tc_bin_days).astype(int)
        scenario_weights: dict[int, float] = {}
        for key, value in zip(bins, weights[finite] * confidence[finite]):
            scenario_weights[int(key)] = scenario_weights.get(int(key), 0.0) + float(value)
        denominator = sum(scenario_weights.values())
        concentration = max(scenario_weights.values()) / denominator if denominator > 0 else 0.0
    else:
        concentration = 0.0

    return SystemicExposure(
        weighted_confidence=weighted_confidence,
        valid_market_coverage=valid_coverage,
        high_risk_market_weight=high_weight,
        tc_near_term_weight=near_term,
        tc_concentration=float(concentration),
        n_assets=len(rows),
    )
