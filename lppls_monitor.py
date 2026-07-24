#!/usr/bin/env python3
"""
LPPLS Bubble Monitor for A-Share Large-Cap Tech Stocks
Based on: Grobys (2025), "Magnificent 7: unsustainable growth and systemic risk",
           Review of Quantitative Finance and Accounting, 67:437-468.

LPPLS Model:
    ln[p(t)] = A + B(tc - t)^β * [1 + C * cos(ω * ln(tc - t) + φ)]

Parameters:
    A  > 0   : expected log-price at critical time tc
    B  < 0   : amplitude of the power-law growth
    β  ∈ (0,1): power-law exponent (super-exponential growth)
    C  ∈ (-1,1): amplitude of log-periodic oscillations
    ω  > 0   : angular log-frequency of oscillations
    φ        : phase parameter (unrestricted)
    tc > t   : critical time (finite-time singularity)

Validation:
    1. NLLS calibration with parameter constraints
    2. ADF test on residuals (99% significance => strong bubble signature)
    3. Iterative subsample re-estimation (cut 20 obs per iteration, 20 iterations)
    4. Confidence interval on tc from iterative estimates
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. LPPLS Model Functions
# ============================================================

def lppls(t, tc, A, B, C, beta, omega, phi):
    """
    Log-Periodic Power Law Singularity model.
    ln[p(t)] = A + B*(tc-t)^beta * [1 + C*cos(omega*ln(tc-t) + phi)]

    Note: tc - t must be positive (tc is in the future relative to all t).
    """
    dt = tc - t
    # Ensure dt is positive; clip to small positive value
    dt = np.maximum(dt, 1e-10)
    log_dt = np.log(dt)
    power = np.power(dt, beta)
    oscillation = 1.0 + C * np.cos(omega * log_dt + phi)
    return A + B * power * oscillation


def lppls_residuals(params, t, log_p):
    """Sum of squared residuals for NLLS."""
    tc, A, B, C, beta, omega, phi = params
    pred = lppls(t, tc, A, B, C, beta, omega, phi)
    resid = log_p - pred
    # Penalize NaN/Inf heavily
    if not np.all(np.isfinite(resid)):
        return 1e15
    return np.sum(resid ** 2)


# ============================================================
# 2. Calibration with Constraints
# ============================================================

def calibrate_lppls(log_p, t=None, tc_bounds=None, maxiter=1000, seed=42):
    """
    Calibrate LPPLS model using differential evolution (global) + L-BFGS-B (local refinement).
    
    Constraints (following Grobys 2025 / Sornette 2017):
        A  > 0
        B  < 0
        0.1 < β < 1.0  (allow lower bound to catch extreme bubbles)
        |C| < 1
        ω  > 0
        tc > t_end    (critical time must be in the future)
    """
    n = len(log_p)
    if t is None:
        t = np.arange(1, n + 1, dtype=np.float64)
    
    t_max = t[-1]
    if tc_bounds is None:
        # tc should be within a reasonable window ahead of current time
        # Allow up to ~252 trading days (1 year) ahead for detection purposes
        tc_bounds = (t_max + 1, t_max + 252)
    
    log_p_min = float(log_p.min())
    log_p_max = float(log_p.max())
    
    # Bounds: [tc, A, B, C, beta, omega, phi]
    bounds = [
        tc_bounds,                          # tc: future critical time
        (max(log_p_min * 0.5, 0.01), log_p_max * 2),  # A: log-price level
        (-10.0, -1e-6),                     # B: negative (power-law amplitude)
        (-0.99, 0.99),                      # C: |C| < 1
        (0.1, 0.999),                       # beta: 0 < beta < 1
        (2.0, 15.0),                        # omega: log-frequency (typical 5-13)
        (0.0, 2 * np.pi),                   # phi: phase
    ]
    
    # Step 1: Global search with differential evolution
    result_de = differential_evolution(
        lppls_residuals,
        bounds=bounds,
        args=(t, log_p),
        maxiter=maxiter,
        seed=seed,
        tol=1e-12,
        atol=1e-12,
        polish=True,
        popsize=30,
        mutation=(0.5, 1.5),
        recombination=0.8,
    )
    
    # Step 2: Local refinement
    result_local = minimize(
        lppls_residuals,
        x0=result_de.x,
        args=(t, log_p),
        method='Nelder-Mead',
        options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-14},
    )
    
    # Choose better result
    if result_local.fun < result_de.fun:
        best_params = result_local.x
        best_sse = result_local.fun
    else:
        best_params = result_de.x
        best_sse = result_de.fun
    
    params_dict = {
        'tc': best_params[0],
        'A': best_params[1],
        'B': best_params[2],
        'C': best_params[3],
        'beta': best_params[4],
        'omega': best_params[5],
        'phi': best_params[6],
    }
    
    residuals = log_p - lppls(t, *best_params)
    
    return params_dict, residuals, best_sse


# ============================================================
# 3. Statistical Testing (ADF on residuals)
# ============================================================

def adf_test(residuals, max_lag=None):
    """
    Augmented Dickey-Fuller test on LPPLS residuals.
    Returns test statistic, p-value, and number of lags.
    
    Critical values (MacKinnon 2010):
        1%: -3.43
        5%: -2.86
        10%: -2.57
    
    If ADF statistic < -3.43 (p < 0.01), residuals are stationary
    => LPPLS signature is statistically significant at 99% level.
    """
    # Auto-select lag by Schwarz (BIC) criterion — same as paper
    result = adfuller(residuals, maxlag=max_lag, autolag='BIC')
    adf_stat = result[0]
    p_value = result[1]
    n_lags = result[2]
    
    if adf_stat < -3.43:
        significance = '***'  # 1% level
    elif adf_stat < -2.86:
        significance = '**'   # 5% level
    elif adf_stat < -2.57:
        significance = '*'    # 10% level
    else:
        significance = ''     # not significant
    
    return {
        'adf_stat': adf_stat,
        'p_value': p_value,
        'n_lags': n_lags,
        'significance': significance,
        'stationary_1pct': adf_stat < -3.43,
        'stationary_5pct': adf_stat < -2.86,
    }


# ============================================================
# 4. Iterative Subsample Robustness Check
# ============================================================

def iterative_calibration(log_p, cutoff=20, n_iterations=20, seed=42):
    """
    Iteratively re-estimate LPPLS by cutting off observations from the beginning.
    
    Following Grobys (2025): in each iteration j, remove 20*j observations from start.
    Store parameter estimates and ADF results for each iteration.
    
    This tests whether tc estimates are stable (robust) or "sloppy" (unreliable).
    """
    results = []
    n = len(log_p)
    
    for j in range(1, n_iterations + 1):
        start_idx = cutoff * j
        if n - start_idx < 200:  # Need minimum data
            break
        
        log_p_sub = log_p[start_idx:]
        t_sub = np.arange(1, len(log_p_sub) + 1, dtype=np.float64)
        t_max = t_sub[-1]
        tc_bounds = (t_max + 1, t_max + 252)
        
        try:
            params, residuals, sse = calibrate_lppls(
                log_p_sub, t=t_sub, tc_bounds=tc_bounds, seed=seed + j
            )
            adf = adf_test(residuals)
            
            results.append({
                'iteration': j,
                'n_obs': len(log_p_sub),
                'tc': params['tc'],
                'tc_days_ahead': params['tc'] - t_max,
                'A': params['A'],
                'B': params['B'],
                'C': params['C'],
                'beta': params['beta'],
                'omega': params['omega'],
                'phi': params['phi'],
                'sse': sse,
                'adf_stat': adf['adf_stat'],
                'adf_pvalue': adf['p_value'],
                'adf_lags': adf['n_lags'],
                'adf_significance': adf['significance'],
                'adf_stationary_1pct': adf['stationary_1pct'],
            })
        except Exception as e:
            results.append({
                'iteration': j,
                'n_obs': n - start_idx,
                'error': str(e),
            })
    
    return pd.DataFrame(results)


# ============================================================
# 5. Bubble Diagnosis (Overall Assessment)
# ============================================================

def diagnose_bubble(log_p, name="Unknown", n_iterations=20):
    """
    Full bubble diagnosis for a single stock.
    
    Returns a comprehensive assessment including:
    - LPPLS calibration on full sample
    - ADF test significance
    - Iterative robustness
    - tc confidence interval
    - Overall bubble risk classification
    """
    t = np.arange(1, len(log_p) + 1, dtype=np.float64)
    t_max = t[-1]
    
    # Full sample calibration
    params, residuals, sse = calibrate_lppls(log_p, t=t)
    adf = adf_test(residuals)
    
    # Iterative robustness
    iter_df = iterative_calibration(log_p, n_iterations=n_iterations)
    
    # Filter successful iterations
    valid_iter = iter_df.dropna(subset=['tc']) if 'tc' in iter_df.columns else pd.DataFrame()
    
    # Compute tc statistics from valid iterations
    tc_stats = {}
    if len(valid_iter) > 0:
        tc_days = valid_iter['tc_days_ahead'].values
        tc_stats = {
            'tc_mean_days': np.mean(tc_days),
            'tc_std_days': np.std(tc_days),
            'tc_min_days': np.min(tc_days),
            'tc_max_days': np.max(tc_days),
            'tc_median_days': np.median(tc_days),
            'n_valid_iterations': len(valid_iter),
        }
        # Count how many iterations are stationary at 1%
        n_stat_1pct = valid_iter['adf_stationary_1pct'].sum() if 'adf_stationary_1pct' in valid_iter.columns else 0
        tc_stats['n_stationary_1pct'] = int(n_stat_1pct)
        tc_stats['stationary_1pct_ratio'] = n_stat_1pct / len(valid_iter) if len(valid_iter) > 0 else 0
    
    # Overall classification
    # CRITICAL: Both conditions must be met (following the paper's framework)
    #   (a) LPPLS signature statistically significant (ADF stationary at 1%)
    #   (b) tc estimates are stable across iterations (not "sloppy")
    
    sig_1pct = adf['stationary_1pct']
    tc_stable = tc_stats.get('tc_std_days', 999) < 60  # std < ~3 months
    stationary_ratio = tc_stats.get('stationary_1pct_ratio', 0)
    
    if sig_1pct and tc_stable and stationary_ratio > 0.5:
        risk_level = 'HIGH'
        risk_desc = 'Strong LPPLS bubble signature: statistically significant and tc estimates stable'
    elif sig_1pct or stationary_ratio > 0.5:
        risk_level = 'MODERATE'
        risk_desc = 'Partial LPPLS signature: some evidence of super-exponential growth'
    elif adf['stationary_5pct']:
        risk_level = 'WATCH'
        risk_desc = 'Weak LPPLS signature at 5% level: monitor closely'
    else:
        risk_level = 'LOW'
        risk_desc = 'No significant LPPLS bubble signature detected'
    
    # tc in calendar terms
    tc_trading_days = params['tc'] - t_max
    tc_date_approx = f"~{tc_trading_days:.0f} trading days ahead"
    
    result = {
        'name': name,
        'n_observations': len(log_p),
        'params': params,
        'sse': sse,
        'adf': adf,
        'tc_stats': tc_stats,
        'iter_results': iter_df,
        'risk_level': risk_level,
        'risk_description': risk_desc,
        'tc_trading_days_ahead': tc_trading_days,
        'tc_approx': tc_date_approx,
        'beta': params['beta'],
        'omega': params['omega'],
        'C': params['C'],
    }
    
    return result


# ============================================================
# 6. Utility: Build Forward-Adjusted Price Series
# ============================================================

def build_qfq_from_pctchg(close_prices, pct_chg):
    """
    Reconstruct forward-adjusted (前复权) price series from raw close + pct_chg.
    
    Uses cumulative product of (1 + pct_chg/100) anchored to the last close.
    This eliminates all ex-dividend jumps while preserving true price returns.
    """
    # Work backwards from the last observation
    n = len(close_prices)
    ratios = 1.0 + np.array(pct_chg) / 100.0
    
    # Anchor: last close is correct (as of most recent date)
    qfq = np.zeros(n)
    qfq[-1] = close_prices[-1]
    
    # Reconstruct backwards
    for i in range(n - 2, -1, -1):
        if ratios[i + 1] != 0:
            qfq[i] = qfq[i + 1] / ratios[i + 1]
        else:
            qfq[i] = qfq[i + 1]
    
    return qfq


if __name__ == "__main__":
    print("LPPLS Bubble Monitor module loaded.")
    print("Use diagnose_bubble(log_p_array, name) for full analysis.")
