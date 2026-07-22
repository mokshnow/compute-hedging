"""Threshold-based execution triggers (avoid continuous hedging churn)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from hedging.risk.margins import margin_spread_vs_forward


class Signal(str, Enum):
    HOLD = "HOLD"
    SELL = "SELL"  # open / add short hedge
    BUY = "BUY"  # unwind / reduce short
    ALERT = "ALERT"


@dataclass
class TriggerConfig:
    min_margin: float = 0.18
    confidence: float = 0.95
    # Sell when projected margin-at-forward falls below floor by this buffer
    breach_buffer: float = 0.02
    # Only re-signal if cushion worsens by this absolute amount
    reentry_delta: float = 0.01
    # Ignore tiny notionals
    min_contracts: int = 1


@dataclass
class TradeSignal:
    signal: Signal
    month: int
    contracts: int
    reason: str
    cushion: float
    margin_at_forward: float


def generate_signals(
    exposure: pd.DataFrame,
    forward_curve: pd.DataFrame,
    monthly_contract_targets: pd.DataFrame,
    config: TriggerConfig | None = None,
) -> list[TradeSignal]:
    """
    Emit sell signals only when projected operating margin breaches the
    confidence interval around the minimum acceptable margin.
    """
    cfg = config or TriggerConfig()
    spread = margin_spread_vs_forward(exposure, forward_curve, cfg.min_margin)
    targets = monthly_contract_targets.set_index("month")["contracts"]

    # Confidence interval around margin_at_forward using rolling residual vol
    mafs = spread["margin_at_forward"].to_numpy()
    vol = float(np.nanstd(np.diff(mafs))) if len(mafs) > 2 else 0.03
    z = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}.get(cfg.confidence, 1.65)
    lower = mafs - z * vol

    signals: list[TradeSignal] = []
    last_sell_cushion = np.inf

    for i, row in spread.iterrows():
        month = int(row["month"])
        contracts = int(targets.get(month, 0))
        maf = float(row["margin_at_forward"])
        cushion = float(row["cushion"])
        lo = float(lower[spread.index.get_loc(i)])

        if contracts < cfg.min_contracts:
            signals.append(TradeSignal(Signal.HOLD, month, 0, "below min contract size", cushion, maf))
            continue

        # Breach of confidence band below floor
        if lo < cfg.min_margin - cfg.breach_buffer:
            if cushion < last_sell_cushion - cfg.reentry_delta:
                signals.append(
                    TradeSignal(
                        Signal.SELL,
                        month,
                        contracts,
                        f"margin CI lower={lo:.3f} < floor={cfg.min_margin:.3f}",
                        cushion,
                        maf,
                    )
                )
                last_sell_cushion = cushion
            else:
                signals.append(TradeSignal(Signal.HOLD, month, contracts, "signal suppressed (no new breach)", cushion, maf))
        elif maf > cfg.min_margin + 2 * cfg.breach_buffer and contracts > 0:
            # Comfortably above floor — allow partial unwind signal
            unwind = max(contracts // 4, 1)
            signals.append(TradeSignal(Signal.BUY, month, unwind, "margin recovered; trim hedge", cushion, maf))
        else:
            signals.append(TradeSignal(Signal.HOLD, month, contracts, "within band", cushion, maf))

    return signals


def signals_to_frame(signals: list[TradeSignal]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "month": s.month,
                "signal": s.signal.value,
                "contracts": s.contracts,
                "reason": s.reason,
                "cushion": s.cushion,
                "margin_at_forward": s.margin_at_forward,
            }
            for s in signals
        ]
    )
