#!/usr/bin/env python3
"""
Parallel LPPLS v2 runner with strict constraint enforcement.
"""

import json
import subprocess
import os
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count

from lppls_monitor import build_qfq_from_pctchg
from lppls_monitor_v2 import diagnose_bubble_v2

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

def fetch_stock_data(code):
    cmd = f"""ssh sz81 "curl -s 'http://127.0.0.1:1934/market/stock-daily?code={code}'" """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    data = json.loads(result.stdout)
    rows = data.get('data', [])
    df = pd.DataFrame(rows)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df

def analyze_stock(args):
    code, name, sector = args
    try:
        df = fetch_stock_data(code)
        if len(df) < 500:
            return {'code': code, 'name': name, 'sector': sector, 'error': 'insufficient data', 'risk_level': 'ERROR'}
        
        cutoff = df['trade_date'].max() - pd.DateOffset(years=5)
        df = df[df['trade_date'] >= cutoff].copy()
        
        close = df['close'].values.astype(float)
        pct = df['pct_chg'].values.astype(float)
        qfq = build_qfq_from_pctchg(close, pct)
        log_p = np.log(np.maximum(qfq, 0.01))
        
        valid = np.isfinite(log_p)
        log_p = log_p[valid]
        
        if len(log_p) < 250:
            return {'code': code, 'name': name, 'sector': sector, 'error': 'too few obs', 'risk_level': 'ERROR'}
        
        result = diagnose_bubble_v2(log_p, name=f"{code} {name}", n_iterations=10)
        
        adf = result['adf']
        tc_stats = result.get('tc_stats', {})
        
        return {
            'code': code,
            'name': name,
            'sector': sector,
            'n_obs': len(log_p),
            'date_start': df['trade_date'].iloc[0].strftime('%Y-%m-%d'),
            'date_end': df['trade_date'].iloc[-1].strftime('%Y-%m-%d'),
            'beta': result['beta'],
            'omega': result['omega'],
            'C': result['C'],
            'constraints_ok': result['constraints_ok'],
            'tc_days_ahead': result['tc_trading_days_ahead'],
            'adf_stat': adf['adf_stat'],
            'adf_pvalue': adf['p_value'],
            'adf_significance': adf['significance'],
            'adf_stationary_1pct': adf['stationary_1pct'],
            'tc_mean_days': tc_stats.get('tc_mean_days'),
            'tc_std_days': tc_stats.get('tc_std_days'),
            'iter_stationary_ratio': tc_stats.get('stationary_1pct_ratio', 0),
            'n_valid_iter': tc_stats.get('n_valid_iterations', 0),
            'risk_level': result['risk_level'],
            'risk_description': result['risk_description'],
        }
    except Exception as e:
        import traceback
        return {'code': code, 'name': name, 'sector': sector, 'error': str(e), 'risk_level': 'ERROR',
                'traceback': traceback.format_exc()}

def main():
    print("=" * 72)
    print("LPPLS Bubble Monitor v2 — A-Share Large-Cap Tech Stocks")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Method: LPPLS (Grobys 2025), strict constraints, {cpu_count()} cores")
    print("=" * 72)
    
    print(f"\nAnalyzing {len(TECH_STOCKS)} stocks in parallel...\n")
    
    with Pool(processes=min(4, len(TECH_STOCKS))) as pool:
        results = pool.map(analyze_stock, TECH_STOCKS)
    
    for r in results:
        print(f"{'─'*60}")
        if 'error' in r:
            print(f"❌ {r['code']} {r['name']}: {r['error']}")
            continue
        
        print(f"📊 {r['code']} {r['name']} ({r['sector']})")
        print(f"   Data: {r['date_start']} → {r['date_end']}, {r['n_obs']} obs")
        print(f"   β={r['beta']:.4f}  ω={r['omega']:.4f}  C={r['C']:.4f}  constraints={'✓' if r['constraints_ok'] else '✗'}")
        print(f"   tc={r['tc_days_ahead']:.1f} trading days ahead")
        print(f"   ADF={r['adf_stat']:.4f}{r['adf_significance']} (p={r['adf_pvalue']:.6f})")
        if r.get('tc_mean_days') is not None:
            print(f"   Iter: tc_mean={r['tc_mean_days']:.0f}d, tc_std={r['tc_std_days']:.0f}d, "
                  f"1%_stat={r['iter_stationary_ratio']*100:.0f}% ({r['n_valid_iter']} valid)")
        print(f"   🔴 Risk: {r['risk_level']} — {r['risk_description']}\n")
    
    print(f"{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    
    rows = []
    for r in results:
        if 'error' in r:
            rows.append({'Code': r['code'], 'Name': r['name'], 'Risk': 'ERROR'})
        else:
            rows.append({
                'Code': r['code'],
                'Name': r['name'],
                'Sector': r['sector'],
                'β': f"{r['beta']:.3f}",
                'C': f"{r['C']:.3f}",
                'ADF': f"{r['adf_stat']:.2f}{r['adf_significance']}",
                'tc_days': f"{r['tc_days_ahead']:.0f}",
                'tc_std': f"{r.get('tc_std_days', 0):.0f}" if r.get('tc_std_days') else 'N/A',
                'Iter_1%': f"{r.get('iter_stationary_ratio', 0)*100:.0f}%",
                'Risk': r['risk_level'],
            })
    
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    summary_df.to_csv(os.path.join(output_dir, 'bubble_monitor_summary_v2.csv'), index=False, encoding='utf-8-sig')
    
    with open(os.path.join(output_dir, 'bubble_monitor_detailed_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📁 Results saved: bubble_monitor_summary_v2.csv, bubble_monitor_detailed_v2.json")
    
    risk_counts = pd.Series([r['risk_level'] for r in results]).value_counts()
    print(f"\n📊 Risk Distribution:")
    for level, count in risk_counts.items():
        emoji = {'HIGH': '🔴', 'MODERATE': '🟠', 'WATCH': '🟡', 'LOW': '🟢', 'ERROR': '❌'}.get(level, '⚪')
        print(f"   {emoji} {level}: {count}")

if __name__ == "__main__":
    main()
