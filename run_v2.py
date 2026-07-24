#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from lppls_monitor import CSVPriceProvider, FMDataHTTPProvider
from lppls_monitor.pipeline import analyze_stock
from lppls_monitor.reporting import write_artifacts

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multiscale LPPLS monitor")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fmdata-url", help="fmdata base URL, e.g. http://127.0.0.1:1934")
    source.add_argument("--csv-dir", type=Path, help="directory containing <code>.csv files")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--lookback-years", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = (
        FMDataHTTPProvider(args.fmdata_url)
        if args.fmdata_url
        else CSVPriceProvider(args.csv_dir)
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / run_id
    results = []
    for code, name, sector in TECH_STOCKS:
        try:
            payload = analyze_stock(provider, code, name, args.lookback_years)
            payload["sector"] = sector
            results.append(payload)
            print(
                f"{code} {name}: {payload['risk_level']} "
                f"confidence={payload['positive_bubble_confidence']:.0%} "
                f"valid={payload['n_valid_windows']}/{payload['n_attempted_windows']}"
            )
        except Exception as exc:
            results.append(
                {
                    "code": code,
                    "name": name,
                    "sector": sector,
                    "risk_level": "ERROR",
                    "error": str(exc),
                }
            )
            print(f"{code} {name}: ERROR {exc}")
    summary, detail, report = write_artifacts(results, output_dir)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "linearized_lppls_multiscale_v3",
        "files": [str(summary), str(detail), str(report)],
        "universe": [code for code, _, _ in TECH_STOCKS],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
