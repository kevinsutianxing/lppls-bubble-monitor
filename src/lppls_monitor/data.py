from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import requests


REQUIRED_COLUMNS = {"trade_date", "close"}


class PriceProvider(Protocol):
    def fetch_daily(self, code: str) -> pd.DataFrame: ...


@dataclass
class FMDataHTTPProvider:
    base_url: str
    timeout_seconds: float = 30.0

    def fetch_daily(self, code: str) -> pd.DataFrame:
        url = f"{self.base_url.rstrip('/')}/market/stock-daily"
        response = requests.get(url, params={"code": code}, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return normalize_daily_frame(pd.DataFrame(payload.get("data", [])))


@dataclass
class CSVPriceProvider:
    directory: Path

    def fetch_daily(self, code: str) -> pd.DataFrame:
        return normalize_daily_frame(pd.read_csv(self.directory / f"{code}.csv"))


def normalize_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"].astype(str), errors="raise")
    out["close"] = pd.to_numeric(out["close"], errors="raise")
    if "pct_chg" in out.columns:
        out["pct_chg"] = pd.to_numeric(out["pct_chg"], errors="raise")
    if "adj_factor" in out.columns:
        out["adj_factor"] = pd.to_numeric(out["adj_factor"], errors="raise")
    if "pct_chg" not in out.columns and "adj_factor" not in out.columns:
        raise ValueError("daily data requires either pct_chg or adj_factor")
    out = out.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    if out.empty or (out["close"] <= 0).any():
        raise ValueError("daily price data is empty or contains non-positive closes")
    numeric_columns = ["close"] + [c for c in ("pct_chg", "adj_factor") if c in out.columns]
    if not np.isfinite(out[numeric_columns].to_numpy()).all():
        raise ValueError("daily price data contains non-finite values")
    return out.reset_index(drop=True)


def build_qfq_from_pctchg(close_prices: np.ndarray, pct_chg: np.ndarray) -> np.ndarray:
    close = np.asarray(close_prices, dtype=float)
    pct = np.asarray(pct_chg, dtype=float)
    if len(close) != len(pct) or len(close) == 0:
        raise ValueError("close_prices and pct_chg must have the same non-zero length")
    ratios = 1.0 + pct / 100.0
    if np.any(ratios <= 0):
        raise ValueError("pct_chg implies a non-positive gross return")
    qfq = np.empty(len(close), dtype=float)
    qfq[-1] = close[-1]
    for idx in range(len(close) - 2, -1, -1):
        qfq[idx] = qfq[idx + 1] / ratios[idx + 1]
    return qfq


def frame_fingerprint(df: pd.DataFrame) -> str:
    canonical = df.sort_values("trade_date").to_csv(index=False, date_format="%Y-%m-%d")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_adjusted_close(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Prefer official adjustment factors; fall back to return reconstruction."""
    if "adj_factor" in df.columns:
        factors = df["adj_factor"].to_numpy(dtype=float)
        if np.any(factors <= 0):
            raise ValueError("adj_factor must be positive")
        adjusted = df["close"].to_numpy(dtype=float) * factors / factors[-1]
        return adjusted, "adj_factor"
    adjusted = build_qfq_from_pctchg(
        df["close"].to_numpy(dtype=float), df["pct_chg"].to_numpy(dtype=float)
    )
    return adjusted, "pct_chg_backward_reconstruction"
