from lppls_monitor.systemic import aggregate_systemic_exposure


def test_systemic_exposure_uses_market_weights_and_tc_clusters():
    rows = [
        {
            "code": "A",
            "risk_level": "HIGH",
            "positive_bubble_confidence": 0.8,
            "valid_fit_ratio": 0.8,
            "tc_median_days": 50,
        },
        {
            "code": "B",
            "risk_level": "MODERATE",
            "positive_bubble_confidence": 0.5,
            "valid_fit_ratio": 0.6,
            "tc_median_days": 65,
        },
        {
            "code": "C",
            "risk_level": "LOW",
            "positive_bubble_confidence": 0.0,
            "valid_fit_ratio": 0.0,
            "tc_median_days": None,
        },
    ]
    result = aggregate_systemic_exposure(rows, {"A": 0.5, "B": 0.3, "C": 0.2})
    assert result.weighted_confidence == 0.55
    assert result.high_risk_market_weight == 0.5
    assert result.valid_market_coverage == 0.8
    assert result.tc_concentration > 0.5
