"""Optimal hedge-ratio calculation for GPU forward shorts ($/GPU-hour)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# One contract = this many GPU-hours of compute notional.
CONTRACT_SIZE_GPU_HOURS = 10_000.0


@dataclass
class HedgeResult:
    """Contracts to short to neutralize downside below the margin floor."""

    hedge_ratio: float  # fraction of notional exposure to hedge
    contracts: int
    notional_usd: float
    expected_payoff_if_spot_to_floor: float
    monthly_targets: pd.DataFrame


def optimal_hedge_ratio(
    exposure: pd.DataFrame,
    forward_curve: pd.DataFrame,
    min_margin: float,
    contract_size_gpu_hours: float = CONTRACT_SIZE_GPU_HOURS,
    confidence: float = 0.95,
    max_hedge_ratio: float = 1.0,
) -> HedgeResult:
    """
    Size a short forward position that offsets losses when spot ($/GPU-hr)
    falls below the profitability threshold.
    """
    curve = forward_curve.sort_values("tenor_months")
    tenors = curve["tenor_months"].to_numpy(dtype=float)
    mids = curve["mid"].to_numpy(dtype=float)
    months = exposure["month"].to_numpy(dtype=float)
    fwd = np.interp(months, tenors, mids)

    gpu_h = exposure["gpu_hours"].to_numpy()
    cost = exposure["total_cost"].to_numpy()
    floor_rev = cost / max(1.0 - min_margin, 1e-9)
    projected_rev = gpu_h * fwd
    shortfall = np.maximum(floor_rev - projected_rev, 0.0)

    spot = exposure["spot_per_gpu_hour"].to_numpy()
    vol = float(np.nanstd(np.diff(np.log(np.maximum(spot, 1e-9))))) if len(spot) > 2 else 0.08
    vol = max(vol, 0.08)
    z = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}.get(confidence, 1.65)
    stress_spot = fwd * np.exp(-z * vol * np.sqrt(np.maximum(months, 1) / 12.0))
    stress_rev = gpu_h * stress_spot
    stress_shortfall = np.maximum(floor_rev - stress_rev, 0.0)

    drop = np.maximum(fwd - stress_spot, 1e-12)
    monthly_contracts = stress_shortfall / (drop * contract_size_gpu_hours)

    peak = float(np.nanmax(monthly_contracts)) if len(monthly_contracts) else 0.0
    total_exposure_notional = float(np.nansum(projected_rev))
    hedged_notional = peak * contract_size_gpu_hours * float(np.nanmean(fwd))
    ratio = min(max_hedge_ratio, hedged_notional / total_exposure_notional if total_exposure_notional else 0.0)

    targets = pd.DataFrame(
        {
            "month": months.astype(int),
            "forward_mid": fwd,
            "stress_spot": stress_spot,
            "shortfall": shortfall,
            "stress_shortfall": stress_shortfall,
            "contracts": np.ceil(monthly_contracts).astype(int),
        }
    )

    contracts = int(np.ceil(peak))
    payoff = float(np.nansum(np.minimum(stress_shortfall, contracts * drop * contract_size_gpu_hours)))

    return HedgeResult(
        hedge_ratio=float(ratio),
        contracts=contracts,
        notional_usd=float(contracts * contract_size_gpu_hours * float(np.nanmean(fwd))),
        expected_payoff_if_spot_to_floor=payoff,
        monthly_targets=targets,
    )


def apply_hedge_pnl(
    exposure: pd.DataFrame,
    hedge: HedgeResult,
    realized_spot: np.ndarray,
    entry_forward: float,
    contract_size_gpu_hours: float = CONTRACT_SIZE_GPU_HOURS,
) -> pd.DataFrame:
    """Mark-to-market short hedge PnL in $/GPU-hour units."""
    out = exposure.copy()
    spot = np.asarray(realized_spot, dtype=float)
    standing = max(hedge.contracts, 0)
    monthly = hedge.monthly_targets.set_index("month")["contracts"].reindex(out["month"]).fillna(standing)
    contracts = np.full(len(out), standing, dtype=float)
    hedge_pnl = contracts * contract_size_gpu_hours * (entry_forward - spot)

    baseline_rev = out["gpu_hours"].to_numpy() * entry_forward
    out["hedge_pnl"] = hedge_pnl
    out["hedged_operating_income"] = out["operating_income"] + hedge_pnl
    out["baseline_revenue"] = baseline_rev
    out["hedged_margin"] = np.where(
        baseline_rev > 0,
        out["hedged_operating_income"] / baseline_rev,
        np.nan,
    )
    out["target_contracts_schedule"] = monthly.to_numpy()
    return out
