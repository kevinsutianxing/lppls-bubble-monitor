#!/usr/bin/env python3
"""
Generate LPPLS visualization charts for all analyzed stocks.
Shows: log-price, fitted LPPLS model, residuals, and risk indicators.
"""

import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib import rcParams

# Chinese font setup
rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['figure.figsize'] = (14, 8)

from lppls_monitor import build_qfq_from_pctchg
from lppls_monitor_v2 import calibrate_lppls_v2, adf_test, lppls

TECH_STOCKS = [
    ("300750", "宁德时代", "新能源科技"),
    ("002475", "立讯精密", "消费电子"),
    ("002415", "海康威视", "安防/AI"),
    ("000725", "京东方A", "半导体显示"),
    ("000063", "中兴通讯", "通信设备"),
    ("002230", "科大讯飞", "AI/语音"),
    ("300059", "东方财富", "金融科技"),
    ("300033", "同花顺", "金融科技/AI"),
    ("688981", "中芯国际", "半导体制造"),
    ("603986", "兆易创新", "半导体设计"),
    ("300015", "爱尔眼科", "医疗服务科技"),
    ("002405", "四维图新", "导航/自动驾驶"),
]

# Load v2 results
with open('bubble_monitor_detailed_v2.json', 'r', encoding='utf-8') as f:
    results_data = json.load(f)

results_map = {r['code']: r for r in results_data}

def fetch_stock_data(code):
    cmd = f"""ssh sz81 "curl -s 'http://127.0.0.1:1934/market/stock-daily?code={code}'" """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    data = json.loads(result.stdout)
    rows = data.get('data', [])
    df = pd.DataFrame(rows)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df

# Generate individual charts for HIGH/MODERATE risk stocks
high_risk_codes = [r['code'] for r in results_data if r['risk_level'] in ['HIGH', 'MODERATE']]

print(f"Generating charts for {len(high_risk_codes)} HIGH/MODERATE risk stocks...")

for code, name, sector in TECH_STOCKS:
    if code not in high_risk_codes:
        continue
    
    r = results_map[code]
    df = fetch_stock_data(code)
    cutoff = df['trade_date'].max() - pd.DateOffset(years=5)
    df = df[df['trade_date'] >= cutoff].copy()
    
    close = df['close'].values.astype(float)
    pct = df['pct_chg'].values.astype(float)
    qfq = build_qfq_from_pctchg(close, pct)
    log_p = np.log(np.maximum(qfq, 0.01))
    
    t = np.arange(1, len(log_p) + 1, dtype=np.float64)
    dates = df['trade_date'].values
    
    # Recalibrate for chart data
    params, residuals, sse, constraints_ok = calibrate_lppls_v2(log_p, t=t)
    
    # Forecast: extend tc range
    t_forecast = np.arange(1, int(params['tc']) + 30, dtype=np.float64)
    pred_full = lppls(t_forecast, params['tc'], params['A'], params['B'], 
                      params['C'], params['beta'], params['omega'], params['phi'])
    
    # Forecast dates
    last_date = df['trade_date'].iloc[-1]
    n_forecast = len(t_forecast) - len(t)
    forecast_dates = pd.date_range(last_date, periods=n_forecast + 1, freq='B')[1:]
    all_dates = np.concatenate([dates, forecast_dates.to_pydatetime()])
    
    # Color by risk
    risk_colors = {'HIGH': '#d32f2f', 'MODERATE': '#f57c00', 'WATCH': '#fbc02d', 'LOW': '#388e3c'}
    risk_color = risk_colors.get(r['risk_level'], '#757575')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top: Log-price + LPPLS fit
    ax1 = axes[0]
    ax1.plot(dates, log_p, color='#1565c0', linewidth=1.2, alpha=0.8, label='Log Price (前复权)', zorder=3)
    ax1.plot(all_dates[:len(t_forecast)], pred_full, color=risk_color, linewidth=1.5, 
             linestyle='--', alpha=0.7, label=f'LPPLS Fit (β={params["beta"]:.3f})', zorder=2)
    
    # Mark tc
    tc_idx = int(params['tc']) - 1
    if tc_idx < len(all_dates):
        tc_date = all_dates[tc_idx]
        ax1.axvline(x=tc_date, color='#d32f2f', linestyle=':', linewidth=1.5, alpha=0.6)
        ax1.annotate(f'tc ≈ {params["tc"] - t[-1]:.0f}d ahead', 
                     xy=(tc_date, pred_full[tc_idx] if tc_idx < len(pred_full) else 0),
                     fontsize=9, color='#d32f2f', fontweight='bold',
                     xytext=(15, -20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='#d32f2f'))
    
    ax1.set_title(f'{code} {name} ({sector}) — LPPLS Bubble Analysis\n'
                  f'Risk: {r["risk_level"]} | ADF={r["adf_stat"]:.2f}{r["adf_significance"]} | '
                  f'tc={r["tc_days_ahead"]:.0f}d ahead', 
                  fontsize=13, fontweight='bold', color=risk_color)
    ax1.set_ylabel('Log Price', fontsize=11)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=dates[-1], color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Bottom: Residuals
    ax2 = axes[1]
    ax2.plot(dates, residuals, color='#6a1b9a', linewidth=0.8, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.fill_between(dates, residuals, 0, alpha=0.15, color='#6a1b9a')
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    adf = adf_test(residuals)
    ax2.set_title(f'Residuals — ADF={adf["adf_stat"]:.4f} {adf["significance"]} (p={adf["p_value"]:.6f}), '
                  f'Stationary@1%: {"YES" if adf["stationary_1pct"] else "NO"}',
                  fontsize=10, color='#6a1b9a')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'chart_{code}_{name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename}")

# Generate summary dashboard
print("\nGenerating summary dashboard...")
fig, ax = plt.subplots(figsize=(16, 9))

risk_order = ['HIGH', 'MODERATE', 'WATCH', 'LOW', 'ERROR']
risk_colors = {'HIGH': '#d32f2f', 'MODERATE': '#f57c00', 'WATCH': '#fbc02d', 'LOW': '#388e3c', 'ERROR': '#9e9e9e'}

sorted_results = sorted(results_data, key=lambda x: (risk_order.index(x['risk_level']) if x['risk_level'] in risk_order else 99, -x.get('adf_stat', 0)))

y_pos = np.arange(len(sorted_results))
colors = [risk_colors.get(r['risk_level'], '#757575') for r in sorted_results]
labels = [f"{r['code']} {r['name']}" for r in sorted_results]

# Bar chart: ADF statistic (more negative = stronger bubble evidence)
adf_vals = [r.get('adf_stat', 0) for r in sorted_results]
bars = ax.barh(y_pos, adf_vals, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.invert_yaxis()

# Reference lines for significance levels
ax.axvline(x=-3.43, color='#d32f2f', linestyle='--', linewidth=1, alpha=0.5, label='1% significance (-3.43)')
ax.axvline(x=-2.86, color='#f57c00', linestyle='--', linewidth=1, alpha=0.5, label='5% significance (-2.86)')
ax.axvline(x=-2.57, color='#fbc02d', linestyle='--', linewidth=1, alpha=0.5, label='10% significance (-2.57)')

# Annotate bars
for i, (bar, val) in enumerate(zip(bars, adf_vals)):
    sig = sorted_results[i].get('adf_significance', '')
    risk = sorted_results[i].get('risk_level', '')
    ax.text(val - 0.05, i, f' {val:.2f}{sig} [{risk}]', va='center', fontsize=9, 
            color=colors[i], fontweight='bold', ha='right')

ax.set_xlabel('ADF Test Statistic (more negative = stronger LPPLS bubble signature)', fontsize=12)
ax.set_title('LPPLS Bubble Risk Monitor — A-Share Large-Cap Tech Stocks\n'
             '(Based on Grobys 2025, Review of Quantitative Finance and Accounting)\n'
             f'Analysis Date: {datetime.now().strftime("%Y-%m-%d")} | Data: Last 5 Years (前复权)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.2, axis='x')
ax.set_xlim(-5.0, 0)

plt.tight_layout()
plt.savefig('bubble_risk_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ bubble_risk_dashboard.png")

print(f"\n✅ All charts generated successfully.")
