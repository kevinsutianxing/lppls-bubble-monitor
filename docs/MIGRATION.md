# Migration from v2

The root-level `lppls_monitor.py` and `lppls_monitor_v2.py` implementations were removed because
their seven-dimensional nonlinear search could classify boundary-saturated solutions as valid.
Import the package instead:

```python
from lppls_monitor import analyze_multiscale, calibrate_lppls
```

Old fields such as `constraints_ok` are replaced by explicit fit status, rejection reasons,
positive-bubble confidence, valid-fit ratio, boundary-saturation ratio, and a `tc` distribution.
Booleans are serialized as JSON booleans, never strings.

The old v2 CSV, JSON, report and charts were removed because their labels predated the new gates.
Regenerate artifacts from the new pipeline.
