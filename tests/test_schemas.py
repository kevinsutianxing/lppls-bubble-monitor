import json

from lppls_monitor.schemas import FitDiagnostics, FitStatus, LPPLSFit


def test_serialization_preserves_boolean_and_enum_types():
    fit = LPPLSFit(
        tc=120.0,
        m=0.5,
        omega=8.0,
        A=5.0,
        B=-0.2,
        C1=0.01,
        C2=0.01,
        C=0.07,
        phi=0.0,
        sse=1.0,
        optimizer_success=True,
        optimizer_message="ok",
        status=FitStatus.VALID,
        diagnostics=FitDiagnostics(),
        n_obs=100,
        t_end=100.0,
    )
    payload = fit.to_dict()
    assert payload["optimizer_success"] is True
    assert payload["status"] == "VALID"
    encoded = json.dumps(payload)
    assert '"optimizer_success": true' in encoded
