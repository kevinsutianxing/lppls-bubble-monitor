from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def write_artifacts(results: Iterable[dict], output_dir: str | Path) -> tuple[Path, Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    detail_path = out / "monitor_detailed.json"
    summary_path = out / "monitor_summary.csv"
    report_path = out / "LPPLS_Monitor_Report.md"
    detail_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_columns = [
        "code", "name", "risk_level", "positive_bubble_confidence",
        "valid_fit_ratio", "boundary_saturation_ratio", "n_valid_windows",
        "n_attempted_windows", "tc_median_days", "tc_p10_days", "tc_p90_days",
        "date_start", "date_end", "data_hash",
    ]
    pd.DataFrame(rows).reindex(columns=summary_columns).to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )
    report_path.write_text(_markdown_report(rows), encoding="utf-8")
    return summary_path, detail_path, report_path


def _markdown_report(rows: list[dict]) -> str:
    lines = [
        "# LPPLS Bubble Monitor Report",
        "",
        "> Risk is assigned only after fit-validity gates. Boundary-saturated fits are rejected.",
        "",
        "| Code | Name | Risk | Confidence | Valid windows | Boundary rate | tc P10–P90 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        p10, p90 = row.get("tc_p10_days"), row.get("tc_p90_days")
        interval = "—" if p10 is None or p90 is None else f"{p10:.0f}–{p90:.0f}d"
        lines.append(
            f"| {row.get('code','')} | {row.get('name','')} | {row.get('risk_level','')} | "
            f"{row.get('positive_bubble_confidence',0):.0%} | "
            f"{row.get('n_valid_windows',0)}/{row.get('n_attempted_windows',0)} | "
            f"{row.get('boundary_saturation_ratio',0):.0%} | {interval} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `tc` is a fragility interval, not a deterministic crash date.",
            "- `LOW` does not imply that an asset is safe or undervalued.",
            "- Use LPPLS together with valuation, liquidity, positioning, policy and fundamental evidence.",
        ]
    )
    return "\n".join(lines) + "\n"
