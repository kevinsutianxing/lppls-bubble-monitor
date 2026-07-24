from __future__ import annotations

import numpy as np
import pandas as pd

from .confidence import analyze_multiscale
from .config import MonitorConfig
from .data import PriceProvider, build_adjusted_close, frame_fingerprint


def analyze_stock(
    provider: PriceProvider,
    code: str,
    name: str,
    lookback_years: int = 5,
    config: MonitorConfig | None = None,
) -> dict:
    df = provider.fetch_daily(code)
    cutoff = df["trade_date"].max() - pd.DateOffset(years=lookback_years)
    df = df[df["trade_date"] >= cutoff].copy()
    adjusted, adjustment_method = build_adjusted_close(df)
    result = analyze_multiscale(np.log(adjusted), name=f"{code} {name}", config=config)
    payload = result.to_dict()
    payload.update(
        {
            "code": code,
            "name": name,
            "date_start": df["trade_date"].iloc[0].strftime("%Y-%m-%d"),
            "date_end": df["trade_date"].iloc[-1].strftime("%Y-%m-%d"),
            "n_obs": int(len(df)),
            "data_hash": frame_fingerprint(df),
            "adjustment_method": adjustment_method,
        }
    )
    return payload
