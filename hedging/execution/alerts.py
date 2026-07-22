"""System health & liquidity alerts before blind execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hedging.kalshi.forward_curve import ForwardContract


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    level: AlertLevel
    code: str
    message: str


@dataclass
class LiquidityConfig:
    max_participation: float = 0.25
    max_slippage_bps: float = 75.0
    min_open_interest: float = 100.0
    min_bid_size: float = 20.0
    allow_slice_hint: bool = True


def estimate_slippage_bps(contracts: int, book_size: float, spread: float, mid: float) -> float:
    if mid <= 0 or book_size <= 0:
        return float("inf")
    participation = max(contracts / book_size, 0.0)
    half_spread_bps = 10_000 * (0.5 * spread / mid)
    # Square-root impact in bps of mid; capped so thin absolute prices don't explode
    impact_bps = min(500.0, 10_000 * 0.35 * (participation**0.5) * max(spread / mid, 1e-4))
    return float(half_spread_bps + impact_bps)


def liquidity_check(
    contract: ForwardContract,
    order_contracts: int,
    config: LiquidityConfig | None = None,
) -> list[Alert]:
    cfg = config or LiquidityConfig()
    alerts: list[Alert] = []

    book = contract.bid_size
    if book < cfg.min_bid_size:
        alerts.append(
            Alert(AlertLevel.CRITICAL, "THIN_BOOK", f"Bid size {book:.0f} < minimum {cfg.min_bid_size:.0f}; skip execution")
        )

    if contract.open_interest < cfg.min_open_interest:
        alerts.append(
            Alert(
                AlertLevel.WARNING,
                "LOW_OI",
                f"Open interest {contract.open_interest:.0f} below institutional threshold {cfg.min_open_interest:.0f}",
            )
        )

    max_slice = int(cfg.max_participation * book) if book > 0 else 0
    if book > 0 and order_contracts > max_slice:
        level = AlertLevel.WARNING if cfg.allow_slice_hint and max_slice >= 1 else AlertLevel.CRITICAL
        alerts.append(
            Alert(
                level,
                "SIZE_VS_LIQUIDITY",
                f"Parent order {order_contracts} exceeds {cfg.max_participation:.0%} of bid "
                f"({max_slice} max slice). "
                + (
                    "Slice across sessions / tenors before sending."
                    if level == AlertLevel.WARNING
                    else "Would induce massive slippage — block."
                ),
            )
        )

    spread = max(contract.ask - contract.bid, 1e-12)
    child = min(order_contracts, max(max_slice, 1))
    child_slip = estimate_slippage_bps(child, max(book, 1.0), spread, contract.mid)
    parent_slip = estimate_slippage_bps(order_contracts, max(book, 1.0), spread, contract.mid)

    if parent_slip > cfg.max_slippage_bps and order_contracts > max_slice >= 1:
        alerts.append(
            Alert(
                AlertLevel.WARNING,
                "SLIPPAGE",
                f"Full-size slippage ~{parent_slip:.0f} bps; child-slice ~{child_slip:.1f} bps "
                f"(limit {cfg.max_slippage_bps:.0f}). Do not dump parent size blindly.",
            )
        )
    elif child_slip > cfg.max_slippage_bps:
        alerts.append(
            Alert(
                AlertLevel.CRITICAL,
                "SLIPPAGE",
                f"Estimated slippage {child_slip:.1f} bps > max {cfg.max_slippage_bps:.1f} bps; do not execute blindly",
            )
        )

    if not alerts:
        alerts.append(Alert(AlertLevel.INFO, "OK", "Order book can absorb hedge size within policy limits"))
    return alerts


def format_alerts(alerts: list[Alert]) -> str:
    return "\n".join(f"[{a.level.value}] {a.code}: {a.message}" for a in alerts)


def block_execution(alerts: list[Alert]) -> bool:
    return any(a.level == AlertLevel.CRITICAL for a in alerts)
