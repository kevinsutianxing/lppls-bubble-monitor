# LPPLS Bubble Monitor for A-Share Tech Stocks

A quantitative bubble-detection system based on the **Log-Periodic Power Law Singularity (LPPLS)** model, adapted from:

> Grobys, K. (2025). *Magnificent 7: unsustainable growth and systemic risk*. **Review of Quantitative Finance and Accounting**, 67:437–468. [DOI: 10.1007/s11156-025-01458-6](https://doi.org/10.1007/s11156-025-01458-6)

The original paper applied LPPLS to the US "Magnificent 7" tech stocks. This project adapts the methodology to **A-share (Chinese mainland) large-cap technology stocks**.

---

## What is LPPLS?

The LPPLS model captures **super-exponential (faster-than-exponential) growth** in asset prices — a hallmark of speculative bubbles driven by herding and positive feedback. The model predicts a **finite-time singularity** (`tc`), a point of extreme market fragility where a regime change (crash or correction) becomes increasingly probable.

### The Model

```
ln[p(t)] = A + B(tc - t)^β × [1 + C × cos(ω × ln(tc - t) + φ)]
```

| Parameter | Meaning | Constraint |
|-----------|---------|------------|
| `A` | Expected log-price near `tc` | `A > 0` |
| `B` | Power-law growth amplitude | `B < 0` |
| `β` | Power-law exponent (super-exponential) | `0 < β < 1` |
| `C` | Oscillation amplitude | `\|C\| < 1` |
| `ω` | Angular log-frequency | `ω > 0` |
| `φ` | Phase | unrestricted |
| `tc` | Critical time (finite-time singularity) | `tc > t_end` |

### Validation Pipeline

1. **NLLS Calibration** — Differential evolution (global) + L-BFGS-B (local) with strict parameter constraints
2. **ADF Test** — Augmented Dickey-Fuller on residuals; stationarity at 1% level ⇒ statistically significant LPPLS signature
3. **Iterative Robustness** — Re-calibrate with progressively shorter windows (cut 20 obs per iteration, 10 rounds) to check `tc` stability
4. **Risk Classification** — Combines ADF significance, `tc` stability, and parameter validity into HIGH / MODERATE / WATCH / LOW

---

## Project Structure

```
.
├── lppls_monitor.py        # Core LPPLS module (model, calibration, ADF test, QFQ builder)
├── lppls_monitor_v2.py     # Strict-constraint version (production)
├── run_v2.py               # Parallel runner for A-share tech stocks
├── generate_charts.py      # Visualization (LPPLS fit plots + risk dashboard)
├── requirements.txt
├── examples/
│   ├── magnificent7_paper.pdf   # Source paper (Open Access)
│   ├── bubble_risk_dashboard.png
│   └── chart_*.png              # LPPLS fit charts for HIGH/MODERATE stocks
└── results/
    ├── bubble_monitor_summary_v2.csv
    ├── bubble_monitor_detailed_v2.json
    └── LPPLS_Bubble_Monitor_Report.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- Access to [fmdata](https://github.com/) financial data service (or modify `fetch_stock_data()` to use your own data source)
- `ssh` access to the data host (or adapt the data fetching)

### Install

```bash
pip install numpy scipy statsmodels pandas matplotlib
```

### Run

```bash
# Analyze 12 A-share tech stocks (parallel, ~15 min)
python run_v2.py

# Generate charts for HIGH/MODERATE risk stocks
python generate_charts.py
```

### Customize

Edit the `TECH_STOCKS` list in `run_v2.py` to add or replace stocks:

```python
TECH_STOCKS = [
    ("300750", "宁德时代", "新能源科技"),
    ("002415", "海康威视", "安防/AI"),
    # ... add your own
]
```

---

## Sample Results (2026-07-23)

| Risk | Count | Stocks |
|------|-------|--------|
| 🔴 HIGH | 3 | 京东方A, 中兴通讯, 兆易创新 |
| 🟠 MODERATE | 3 | 宁德时代, 立讯精密, 海康威视 |
| 🟡 WATCH | 5 | 科大讯飞, 东方财富, 同花顺, 中芯国际, 爱尔眼科 |
| 🟢 LOW | 1 | 四维图新 |

---

## Important Disclaimers

- **LPPLS is a diagnostic tool, not a trading signal.** The critical time `tc` represents a phase of extreme fragility, not a deterministic crash date (Grobys 2025, p. 332; Sornette 2017).
- A-share markets have unique characteristics (±10%/20% price limits, policy interventions, T+1) that may affect model applicability.
- This is a research/educational project, **not investment advice**.
- The source paper is Open Access under CC BY 4.0.

## References

- Grobys, K. (2025). Magnificent 7: unsustainable growth and systemic risk. *Rev Quant Finan Acc* 67, 437–468.
- Sornette, D. (2017). *Why Stock Markets Crash*. Princeton University Press.
- Johansen, A. & Sornette, D. (2001). Finite-time singularity in the dynamics of the world population. *Physica A* 294, 465–502.
- Lin, L., Ren, R. & Sornette, D. (2014). The volatility-confined LPPL model. *Int Rev Financial Anal* 33, 210–225.
