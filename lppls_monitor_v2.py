#!/usr/bin/env python3
"""
LPPLS Bubble Monitor v2 — Fixed constraint enforcement.
Uses penalty-based NLLS to strictly enforce:
    A > 0, B < 0, 0 < β < 1, ω > 0, |C| < 1
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# LPPLS Model
# ============================================================

def lppls(t, tc, A, B, C, beta, omega, phi):
    dt = np.maximum(tc - t, 1e-10)
    log_dt = np.log(dt)
    power = np.power(dt, beta)
    oscillation = 1.0 + C * np.cos(omega * log_dt + phi)
    return A + B * power * oscillation


def lppls_penalty(params, t, log_p, penalty_weight=1e6):
    """
    NLLS objective with constraint penalty.
    Constraints: A>0, B<0, 0.1<=β<=0.99, |C|<0.99, ω>0, tc>t_end
    """
    tc, A, B, C, beta, omega, phi = params
    
    pred = lppls(t, tc, A, B, C, beta, omega, phi)
    resid = log_p - pred
    
    if not np.all(np.isfinite(resid)):
        return 1e15
    
    sse = np.sum(resid ** 2)
    
    # Constraint penalties (quadratic barrier)
    penalty = 0.0
    if A <= 0:
        penalty += (A) ** 2
    if B >= 0:
        penalty += (B) ** 2
    if beta < 0.1:
        penalty += (0.1 - beta) ** 2
    if beta > 0.99:
        penalty += (beta - 0.99) ** 2
    if abs(C) >= 0.99:
        penalty += (abs(C) - 0.99) ** 2
    if omega <= 1.0:
        penalty += (1.0 - omega) ** 2
    
    return sse + penalty_weight * penalty


# ============================================================
# Calibration
# ============================================================

def calibrate_lppls_v2(log_p, t=None, tc_bounds=None, seed=42):
    """
    Calibrate LPPLS with strict constraint enforcement.
    Uses differential_evolution (global) + L-BFGS-B (local, bounded).
    """
    n = len(log_p)
    if t is None:
        t = np.arange(1, n + 1, dtype=np.float64)
    
    t_max = t[-1]
    if tc_bounds is None:
        tc_bounds = (t_max + 1, t_max + 252)
    
    log_p_min = float(log_p.min())
    log_p_max = float(log_p.max())
    
    bounds = [
        tc_bounds,                                          # tc
        (max(log_p_min * 0.3, 0.01), log_p_max * 2.5),     # A
        (-5.0, -1e-6),                                      # B
        (-0.95, 0.95),                                      # C
        (0.1, 0.99),                                        # beta
        (2.0, 15.0),                                        # omega
        (0.0, 2 * np.pi),                                   # phi
    ]
    
    # Step 1: Global search
    result_de = differential_evolution(
        lppls_penalty,
        bounds=bounds,
        args=(t, log_p),
        maxiter=500,
        seed=seed,
        tol=1e-12,
        polish=True,
        popsize=25,
        mutation=(0.5, 1.5),
        recombination=0.8,
    )
    
    # Step 2: Local refinement with bounded L-BFGS-B
    best_params = result_de.x
    best_val = result_de.fun
    
    try:
        result_local = minimize(
            lppls_penalty,
            x0=result_de.x,
            args=(t, log_p),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12},
        )
        if result_local.fun < best_val:
            best_val = result_local.fun
            best_params = result_local.x
    except:
        pass
    
    # Multi-start refinement
    np.random.seed(seed)
    for _ in range(5):
        x0 = best_params + np.random.randn(7) * np.array([5, 0.05, 0.05, 0.03, 0.03, 0.3, 0.2])
        for i in range(7):
            x0[i] = np.clip(x0[i], bounds[i][0], bounds[i][1])
        
        try:
            r = minimize(
                lppls_penalty, x0=x0, args=(t, log_p),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12},
            )
            if r.fun < best_val:
                best_val = r.fun
                best_params = r.x
        except:
            pass
    
    # Verify constraints are satisfied
    tc, A, B, C, beta, omega, phi = best_params
    constraints_ok = (
        A > 0 and B < 0 and 0.1 <= beta <= 0.99 
        and abs(C) < 0.99 and omega > 1.0 and tc > t_max
    )
    
    params_dict = {
        'tc': tc, 'A': A, 'B': B, 'C': C,
        'beta': beta, 'omega': omega, 'phi': phi,
    }
    
    residuals = log_p - lppls(t, *best_params)
    sse = np.sum(residuals ** 2)
    
    return params_dict, residuals, sse, constraints_ok


# ============================================================
# ADF Test
# ============================================================

def adf_test(residuals, max_lag=None):
    result = adfuller(residuals, maxlag=max_lag, autolag='BIC')
    adf_stat = result[0]
    p_value = result[1]
    n_lags = result[2]
    
    if adf_stat < -3.43:
        significance, sig_1pct = '***', True
    elif adf_stat < -2.86:
        significance, sig_1pct = '**', False
    elif adf_stat < -2.57:
        significance, sig_1pct = '*', False
    else:
        significance, sig_1pct = '', False
    
    return {
        'adf_stat': adf_stat, 'p_value': p_value, 'n_lags': n_lags,
        'significance': significance,
        'stationary_1pct': sig_1pct,
        'stationary_5pct': adf_stat < -2.86,
    }


# ============================================================
# Iterative Robustness
# ============================================================

def iterative_calibration_v2(log_p, cutoff=20, n_iterations=15, seed=42):
    results = []
    n = len(log_p)
    
    for j in range(1, n_iterations + 1):
        start_idx = cutoff * j
        if n - start_idx < 200:
            break
        
        log_p_sub = log_p[start_idx:]
        t_sub = np.arange(1, len(log_p_sub) + 1, dtype=np.float64)
        t_max_sub = t_sub[-1]
        tc_bounds = (t_max_sub + 1, t_max_sub + 252)
        
        try:
            params, residuals, sse, constraints_ok = calibrate_lppls_v2(
                log_p_sub, t=t_sub, tc_bounds=tc_bounds, seed=seed + j
            )
            adf = adf_test(residuals)
            
            results.append({
                'iteration': j,
                'n_obs': len(log_p_sub),
                'tc_days_ahead': params['tc'] - t_max_sub,
                'A': params['A'],
                'B': params['B'],
                'C': params['C'],
                'beta': params['beta'],
                'omega': params['omega'],
                'sse': sse,
                'adf_stat': adf['adf_stat'],
                'adf_pvalue': adf['p_value'],
                'adf_significance': adf['significance'],
                'adf_stationary_1pct': adf['stationary_1pct'],
                'constraints_ok': constraints_ok,
            })
        except Exception as e:
            pass
    
    return pd.DataFrame(results)


# ============================================================
# Full Diagnosis
# ============================================================

def diagnose_bubble_v2(log_p, name="Unknown", n_iterations=15):
    t = np.arange(1, len(log_p) + 1, dtype=np.float64)
    t_max = t[-1]
    
    # Full sample
    params, residuals, sse, constraints_ok = calibrate_lppls_v2(log_p, t=t)
    adf = adf_test(residuals)
    
    # Iterative
    iter_df = iterative_calibration_v2(log_p, n_iterations=n_iterations)
    
    # Only use iterations where constraints are satisfied
    valid_iter = iter_df[iter_df['constraints_ok'] == True] if 'constraints_ok' in iter_df.columns else iter_df
    
    tc_stats = {}
    if len(valid_iter) > 0:
        tc_days = valid_iter['tc_days_ahead'].values
        n_stat = valid_iter['adf_stationary_1pct'].sum()
        tc_stats = {
            'tc_mean_days': float(np.mean(tc_days)),
            'tc_std_days': float(np.std(tc_days)),
            'tc_min_days': float(np.min(tc_days)),
            'tc_max_days': float(np.max(tc_days)),
            'tc_median_days': float(np.median(tc_days)),
            'n_valid_iterations': len(valid_iter),
            'n_stationary_1pct': int(n_stat),
            'stationary_1pct_ratio': float(n_stat / len(valid_iter)),
        }
    
    # Risk classification
    sig_1pct = adf['stationary_1pct']
    tc_stable = tc_stats.get('tc_std_days', 999) < 60
    stationary_ratio = tc_stats.get('stationary_1pct_ratio', 0)
    beta_reasonable = 0.1 <= params['beta'] <= 0.95
    c_reasonable = abs(params['C']) < 0.9
    
    if sig_1pct and tc_stable and stationary_ratio >= 0.5 and beta_reasonable and c_reasonable:
        risk_level = 'HIGH'
        risk_desc = 'Strong LPPLS bubble signature: significant + tc stable + parameters in valid range'
    elif (sig_1pct and (tc_stable or stationary_ratio >= 0.4)) or (stationary_ratio >= 0.5 and beta_reasonable):
        risk_level = 'MODERATE'
        risk_desc = 'Partial LPPLS signature: some evidence of super-exponential growth'
    elif adf['stationary_5pct']:
        risk_level = 'WATCH'
        risk_desc = 'Weak LPPLS signature at 5% level: monitor'
    else:
        risk_level = 'LOW'
        risk_desc = 'No significant LPPLS bubble signature detected'
    
    tc_trading_days = params['tc'] - t_max
    
    return {
        'name': name,
        'n_observations': len(log_p),
        'params': params,
        'sse': sse,
        'constraints_ok': constraints_ok,
        'adf': adf,
        'tc_stats': tc_stats,
        'risk_level': risk_level,
        'risk_description': risk_desc,
        'tc_trading_days_ahead': tc_trading_days,
        'beta': params['beta'],
        'omega': params['omega'],
        'C': params['C'],
        'iter_df': iter_df,
    }
