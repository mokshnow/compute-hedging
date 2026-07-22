"""End-to-end execution engine: signals → liquidity gate → rebalance."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from hedging.execution.alerts import Alert, block_execution, liquidity_check
from hedging.execution.rebalance import Portfolio, RebalanceAction, rebalance, target_from_signals
from hedging.execution.triggers import TriggerConfig, TradeSignal, generate_signals, signals_to_frame
from hedging.kalshi.forward_curve import ForwardContract, ForwardCurveStore
from hedging.risk.hedge_ratio import HedgeResult, optimal_hedge_ratio


@dataclass
class ExecutionReport:
    hedge: HedgeResult
    signals: pd.DataFrame
    actions: list[RebalanceAction] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    portfolio: Portfolio = field(default_factory=Portfolio)
    executed: bool = False


def run_execution_cycle(
    exposure: pd.DataFrame,
    curve_df: pd.DataFrame,
    contracts: list[ForwardContract],
    min_margin: float = 0.18,
    asof: pd.Timestamp | None = None,
    store: ForwardCurveStore | None = None,
) -> ExecutionReport:
    """
    One cycle of the automated hedging pipeline.

    1. Size optimal hedge
    2. Generate threshold-based signals
    3. Liquidity-check the intended short
    4. Rebalance only if book can absorb size
    """
    asof = asof or pd.Timestamp.utcnow()
    if store is not None:
        store.append_snapshot(contracts, asof)

    hedge = optimal_hedge_ratio(exposure, curve_df, min_margin)
    signals = generate_signals(
        exposure,
        curve_df,
        hedge.monthly_targets,
        TriggerConfig(min_margin=min_margin),
    )
    month = int(exposure["month"].iloc[-1]) if len(exposure) else 0
    target = target_from_signals(signals, month) or hedge.contracts

    front = max(contracts, key=lambda c: c.bid_size) if contracts else None
    alerts: list[Alert] = []
    if front is not None:
        alerts = liquidity_check(front, target)

    portfolio = Portfolio()
    actions: list[RebalanceAction] = []
    executed = False
    if front is not None and not block_execution(alerts):
        portfolio, actions = rebalance(Portfolio(), target, curve_df, asof)
        executed = True

    return ExecutionReport(
        hedge=hedge,
        signals=signals_to_frame(signals),
        actions=actions,
        alerts=alerts,
        portfolio=portfolio,
        executed=executed,
    )
