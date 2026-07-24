# Methodology and Validation Contract

## Separation of concerns

The monitor deliberately separates three questions:

1. **Did the optimizer converge?**
2. **Does the fitted curve satisfy LPPLS validity conditions?**
3. **How much multiscale evidence remains after invalid fits are removed?**

Risk classification is forbidden until questions 1 and 2 pass.

## Fit-status contract

- `VALID`: all numerical and economic gates pass.
- `BOUNDARY_SATURATED`: a nonlinear parameter is too close to its search bound.
- `OPTIMIZER_FAILED`: the selected optimizer result did not report convergence.
- `NON_BUBBLE_SHAPE`: `B >= 0`, non-finite amplitude, or `|C| >= 1`.
- `DAMPING_CONDITION_FAILED`: the damping condition is not satisfied.
- `INSUFFICIENT_OSCILLATIONS`: too few log-periodic cycles occur inside the window.
- `POOR_FIT`: relative RMSE exceeds the configured limit.
- `INSUFFICIENT_DATA`: fewer than 60 finite observations.

## Statistical diagnostics

ADF is applied to residuals after the full LPPLS fit. Lomb power is calculated after removing the
pure power-law component, so it targets log-periodic structure. Lomb significance is approximate
and must not replace simulation-based calibration.

## Production gate

Before thresholds are treated as production thresholds, run point-in-time historical snapshots and
random-walk / GARCH-like null simulations. Required outputs include false-positive rates by regime,
precision and recall, signal lead time and duration, parameter sensitivity, and incremental value
relative to momentum and volatility baselines.
