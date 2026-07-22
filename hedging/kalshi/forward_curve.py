"""GPU compute forward-curve construction & historical reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


@dataclass
class ForwardContract:
    ticker: str
    expiry: pd.Timestamp
    mid: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    open_interest: float = 0.0
    tenor_months: float | None = None


class ForwardCurveStore:
    """
    Persist and reconstruct Kalshi GPU forward curves at arbitrary timestamps.

    Storage layout: parquet partitions by snapshot_ts.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.cache_dir / "forward_curve_history.parquet"
        self._df = self._load()

    def _load(self) -> pd.DataFrame:
        if self._path.exists():
            return pd.read_parquet(self._path)
        return pd.DataFrame(
            columns=[
                "snapshot_ts",
                "ticker",
                "expiry",
                "mid",
                "bid",
                "ask",
                "bid_size",
                "ask_size",
                "open_interest",
                "tenor_months",
            ]
        )

    def save(self) -> None:
        self._df.to_parquet(self._path, index=False)

    def append_snapshot(self, contracts: list[ForwardContract], snapshot_ts: pd.Timestamp | None = None) -> None:
        ts = snapshot_ts or pd.Timestamp.utcnow()
        rows = []
        for c in contracts:
            tenor = (c.expiry - ts).days / 30.44
            rows.append(
                {
                    "snapshot_ts": ts,
                    "ticker": c.ticker,
                    "expiry": c.expiry,
                    "mid": c.mid,
                    "bid": c.bid,
                    "ask": c.ask,
                    "bid_size": c.bid_size,
                    "ask_size": c.ask_size,
                    "open_interest": c.open_interest,
                    "tenor_months": tenor,
                }
            )
        self._df = pd.concat([self._df, pd.DataFrame(rows)], ignore_index=True)
        self.save()

    def curve_at(self, asof: pd.Timestamp) -> pd.DataFrame:
        """Return the forward curve snapshot closest to ``asof`` (not after)."""
        if self._df.empty:
            return self._df.copy()
        df = self._df.copy()
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
        asof = pd.Timestamp(asof)
        if asof.tzinfo is None:
            asof = asof.tz_localize("UTC")
        eligible = df[df["snapshot_ts"] <= asof]
        if eligible.empty:
            eligible = df
        snap = eligible["snapshot_ts"].max()
        curve = eligible[eligible["snapshot_ts"] == snap].sort_values("tenor_months")
        return curve.reset_index(drop=True)

    def interpolate_price(self, asof: pd.Timestamp, tenor_months: float) -> float:
        curve = self.curve_at(asof)
        if curve.empty:
            raise ValueError("No forward curve data available")
        x = curve["tenor_months"].to_numpy(dtype=float)
        y = curve["mid"].to_numpy(dtype=float)
        return float(np.interp(tenor_months, x, y))


def contracts_to_frame(contracts: list[ForwardContract], snapshot_ts: pd.Timestamp) -> pd.DataFrame:
    """Build a clean forward-curve frame from contracts (no persistent cache)."""
    ts = pd.Timestamp(snapshot_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    rows = []
    for c in contracts:
        if c.tenor_months is not None:
            tenor = float(c.tenor_months)
        else:
            tenor = (pd.Timestamp(c.expiry) - ts).days / 30.44
        rows.append(
            {
                "snapshot_ts": ts,
                "ticker": c.ticker,
                "expiry": c.expiry,
                "mid": c.mid,
                "bid": c.bid,
                "ask": c.ask,
                "bid_size": c.bid_size,
                "ask_size": c.ask_size,
                "open_interest": c.open_interest,
                "tenor_months": tenor,
            }
        )
    return pd.DataFrame(rows).sort_values("tenor_months").reset_index(drop=True)


def orderbook_to_contract(ticker: str, expiry: pd.Timestamp, book: dict) -> ForwardContract:
    """Collapse a Kalshi order book into mid/size summary for curve building."""
    yes = book.get("orderbook", book)
    # Kalshi format: yes bids as [[price_cents, qty], ...]; no side mirrors asks
    yes_bids = yes.get("yes", []) or []
    no_bids = yes.get("no", []) or []

    bid = (yes_bids[-1][0] / 100.0) if yes_bids else 0.5
    bid_size = float(yes_bids[-1][1]) if yes_bids else 0.0
    # Best ask approximated from complementary no bid
    ask = (1.0 - no_bids[-1][0] / 100.0) if no_bids else bid
    ask_size = float(no_bids[-1][1]) if no_bids else 0.0
    mid = 0.5 * (bid + ask)
    return ForwardContract(ticker, expiry, mid, bid, ask, bid_size, ask_size)


def simulate_gpu_forward_curve(
    asof: pd.Timestamp,
    spot: float,
    tenors_months: list[int] | None = None,
    # Negative = backwardation: rental rates for a chip generation drift down with obsolescence.
    annual_drift: float = -0.10,
    rng: np.random.Generator | None = None,
) -> list[ForwardContract]:
    """
    Synthetic GPU compute forward curve when live Kalshi markets are unavailable.

    Default shape is mild **backwardation** (~10%/year): far tenors price below
    spot as a given GPU generation loses perf/$ to successors. Prices are
    $/GPU-hour mids (Ornn OCPI units).
    """
    rng = rng or np.random.default_rng(7)
    tenors = tenors_months or [1, 2, 3, 6, 9, 12, 18, 24]
    contracts: list[ForwardContract] = []
    for m in tenors:
        fwd = spot * (1.0 + annual_drift * (m / 12.0)) * (1.0 + rng.normal(0, 0.01))
        fwd = max(float(fwd), spot * 0.35)
        # ~8–20 bps quoted spread, wider with tenor
        spread = fwd * (0.0008 + 0.00005 * m)
        expiry = pd.Timestamp(asof) + pd.DateOffset(months=m)
        ticker = f"GPU-CMP-{expiry.strftime('%y%b').upper()}"
        contracts.append(
            ForwardContract(
                ticker=ticker,
                expiry=expiry,
                mid=float(fwd),
                bid=float(fwd - spread / 2),
                ask=float(fwd + spread / 2),
                bid_size=float(rng.integers(5_000, 25_000)),
                ask_size=float(rng.integers(5_000, 25_000)),
                open_interest=float(rng.integers(20_000, 80_000)),
                tenor_months=float(m),
            )
        )
    return contracts
