#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate charts from monitor_detailed.json")
    parser.add_argument("detail_json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.detail_json.read_text(encoding="utf-8"))
    output_dir = args.output_dir or args.detail_json.parent / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"{row.get('code', '')} {row.get('name', '')}" for row in rows]
    confidence = [float(row.get("positive_bubble_confidence", 0.0)) for row in rows]
    fig, ax = plt.subplots(figsize=(12, max(5, len(rows) * 0.45)))
    y = np.arange(len(rows))
    ax.barh(y, confidence)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Positive-bubble confidence")
    ax.set_title("LPPLS multiscale confidence (invalid boundary fits excluded)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for row in rows:
        latest = row.get("latest_fit")
        if not latest:
            continue
        diagnostics = latest.get("diagnostics", {})
        data = {
            "parameter": ["m", "omega", "C", "tc_days", "relative_rmse", "oscillations"],
            "value": [
                latest.get("m"),
                latest.get("omega"),
                latest.get("C"),
                latest.get("tc", 0) - latest.get("t_end", 0),
                diagnostics.get("relative_rmse"),
                diagnostics.get("oscillation_count"),
            ],
        }
        frame = pd.DataFrame(data).dropna()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(frame["parameter"], frame["value"])
        ax.set_title(
            f"{row.get('code', '')} {row.get('name', '')} — {latest.get('status', '')}"
        )
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"diagnostics_{row.get('code', 'unknown')}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
