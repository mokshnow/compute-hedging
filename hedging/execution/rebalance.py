"""Dynamic rebalancing: roll / unwind as hardware depreciates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hedging.execution.triggers import Signal, TradeSignal


@dataclass
class Position:
    ticker: str
    contracts: int  # negative = short
    entry_mid: float
    expiry: pd.Timestamp


@dataclass
class Portfolio:
    positions: list[Position] = field(default_factory=list)

    @property
    def net_short(self) -> int:
        """Absolute short contracts (positive number of shorts)."""
        return int(sum(-p.contracts for p in self.positions if p.contracts < 0))

    def total_contracts(self) -> int:
        return int(sum(p.contracts for p in self.positions))


@dataclass
class RebalanceAction:
    action: str  # open_short | unwind | roll
    from_ticker: str | None
    to_ticker: str | None
    contracts: int
    reason: str


def select_front_contract(curve: pd.DataFrame, min_tenor_months: float = 0.5) -> pd.Series:
    eligible = curve[curve["tenor_months"] >= min_tenor_months].sort_values("tenor_months")
    if eligible.empty:
        return curve.sort_values("tenor_months").iloc[0]
    return eligible.iloc[0]


def rebalance(
    portfolio: Portfolio,
    target_short_contracts: int,
    curve: pd.DataFrame,
    asof: pd.Timestamp,
    roll_when_tenor_below: float = 1.0,
) -> tuple[Portfolio, list[RebalanceAction]]:
    """
    Adjust standing short to ``target_short_contracts`` and roll near-expiry legs.
    """
    actions: list[RebalanceAction] = []
    front = select_front_contract(curve)
    new_positions: list[Position] = []

    # Roll aged contracts
    for pos in portfolio.positions:
        row = curve[curve["ticker"] == pos.ticker]
        tenor = float(row.iloc[0]["tenor_months"]) if not row.empty else 0.0
        if tenor < roll_when_tenor_below and pos.contracts < 0:
            qty = -pos.contracts
            actions.append(
                RebalanceAction("roll", pos.ticker, str(front["ticker"]), qty, f"tenor {tenor:.2f}m < {roll_when_tenor_below}")
            )
            new_positions.append(
                Position(str(front["ticker"]), -qty, float(front["mid"]), pd.Timestamp(front["expiry"]))
            )
        else:
            new_positions.append(pos)

    current_short = sum(-p.contracts for p in new_positions if p.contracts < 0)
    delta = target_short_contracts - current_short

    if delta > 0:
        actions.append(RebalanceAction("open_short", None, str(front["ticker"]), delta, "sizing hedge to target"))
        # merge into front position
        merged = False
        for i, p in enumerate(new_positions):
            if p.ticker == front["ticker"] and p.contracts < 0:
                new_positions[i] = Position(p.ticker, p.contracts - delta, p.entry_mid, p.expiry)
                merged = True
                break
        if not merged:
            new_positions.append(Position(str(front["ticker"]), -delta, float(front["mid"]), pd.Timestamp(front["expiry"])))
    elif delta < 0:
        unwind = -delta
        actions.append(RebalanceAction("unwind", str(front["ticker"]), None, unwind, "reduce hedge to target"))
        remaining = unwind
        adjusted: list[Position] = []
        for p in new_positions:
            if remaining <= 0 or p.contracts >= 0:
                adjusted.append(p)
                continue
            short_qty = -p.contracts
            take = min(short_qty, remaining)
            new_qty = p.contracts + take  # toward zero
            remaining -= take
            if new_qty != 0:
                adjusted.append(Position(p.ticker, new_qty, p.entry_mid, p.expiry))
        new_positions = adjusted

    return Portfolio(new_positions), actions


def target_from_signals(signals: list[TradeSignal], month: int) -> int:
    """Standing short size implied by the latest SELL/HOLD target at ``month``."""
    relevant = [s for s in signals if s.month <= month and s.signal in (Signal.SELL, Signal.HOLD) and s.contracts > 0]
    if not relevant:
        return 0
    return relevant[-1].contracts
