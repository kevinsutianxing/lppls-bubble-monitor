# LPPLS Bubble Monitor for A-Share Technology Stocks

A research-grade **Log-Periodic Power Law Singularity (LPPLS)** diagnostic system for detecting
super-exponential price regimes. Version 0.3 replaces the original single-window, seven-parameter
fit with a linearized multiscale implementation and explicit fit-validity gates.

## Why version 0.3

The original prototype could label a fit `HIGH` even when `tc`, `m`, `omega` or `C` sat directly on
the optimizer boundary. A stable sequence of boundary solutions can produce a misleadingly small
`tc` standard deviation. Version 0.3 therefore applies this order of operations:

1. calibrate the model;
2. reject invalid, boundary-saturated or economically inconsistent fits;
3. test residual and log-periodic diagnostics;
4. aggregate evidence over multiple lookback windows;
5. assign a monitoring state only after the validity gate.

A rejected fit is **not** weak evidence. It is excluded from the confidence numerator.

## Model and calibration

```text
ln p(t) = A + B f(t) + C1 g(t) + C2 h(t)
f(t) = (tc - t)^m
g(t) = f(t) cos(omega ln(tc - t))
h(t) = f(t) sin(omega ln(tc - t))
```

Only `tc`, `m` and `omega` are searched nonlinearly. For every candidate, `A`, `B`, `C1` and `C2`
are solved by linear least squares. This reduces the nonlinear search from seven dimensions to
three and removes the phase parameter from the optimizer.

The default strict profile uses:

- `0.10 <= m <= 0.90`;
- `6 <= omega <= 13`;
- `1 <= tc - t_end <= 252` trading days;
- `B < 0` and relative oscillation amplitude `|C| < 1`;
- damping condition;
- at least 2.5 log-periodic oscillations;
- relative-RMSE threshold;
- explicit rejection near any search boundary.

## Multiscale confidence indicator

The monitor scans the latest 120, 180, 250, 360, 500, 750, 1000 and 1250 observations when
available. Every attempted window remains in the denominator, including optimizer failures.

```text
positive bubble confidence
= qualified windows with ADF p <= 5% and Lomb p <= 10% / all attempted windows
```

Outputs include positive-bubble confidence, valid-fit ratio, boundary-saturation ratio, attempted
and valid window counts, a `tc` distribution, and per-window rejection reasons. `tc` is a fragility
interval, **not a deterministic crash date**.

## Project structure

```text
src/lppls_monitor/
├── model.py          # linearized LPPLS model
├── calibration.py    # 3D nonlinear search + linear least squares
├── validation.py     # boundary, damping, oscillation and fit gates
├── diagnostics.py    # residual ADF and Lomb diagnostic
├── confidence.py     # multiscale confidence indicator
├── data.py           # fmdata HTTP and CSV providers, data-quality checks
├── pipeline.py       # stock-level pipeline and data provenance
├── backtest.py       # point-in-time walk-forward evaluation
├── systemic.py       # market-cap-weighted systemic exposure
├── null_simulation.py# geometric-random-walk false-positive tests
├── reporting.py      # JSON, CSV and Markdown artifacts
└── schemas.py        # typed, JSON-safe output objects
```

## Installation

```bash
python -m pip install -e ".[dev]"
pytest
```

## Run with fmdata

```bash
python run_v2.py --fmdata-url http://127.0.0.1:1934
```

For offline runs, each ticker CSV must contain `trade_date`, `close`, and either an official
`adj_factor` or `pct_chg`:

```bash
python run_v2.py --csv-dir ./data/snapshots/2026-07-23
```

Every run writes a manifest, summary CSV, detailed JSON and Markdown report under
`artifacts/<UTC run_id>/`. The stock payload includes a SHA-256 hash of the normalized input frame.

## Validation utilities

```python
from lppls_monitor import (
    aggregate_systemic_exposure,
    simulate_null_false_positive_rate,
    walk_forward_backtest,
)
```

The backtest evaluates future 20/60/120/250-day returns and maximum drawdowns using only information
available at each historical evaluation date. Thresholds should be promoted to production only after
false-positive, precision/recall and regime-stability analysis on point-in-time data.

## Risk interpretation

- `HIGH`: sufficiently many valid windows, high confidence and a stable `tc` distribution.
- `MODERATE`: meaningful multiscale evidence, but incomplete or less stable.
- `WATCH`: a minority of windows pass all gates.
- `LOW`: no robust positive-bubble signature.
- invalid fits retain their explicit `fit_status`; they are never silently converted to `LOW`.

LPPLS is a diagnostic, not a trading signal. Combine it with valuation, earnings, liquidity,
positioning, policy and market-structure evidence.

## References

- Filimonov, V. & Sornette, D. (2013). A stable and robust calibration scheme of the LPPLS model.
- Grobys, K. (2025). *Magnificent 7: unsustainable growth and systemic risk*.
- Sornette, D. (2017). *Why Stock Markets Crash*.
