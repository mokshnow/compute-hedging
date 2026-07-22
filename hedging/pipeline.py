"""Orchestrate a 12–24 month hedged vs unhedged simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import requests

from hedging.execution.engine import ExecutionReport, run_execution_cycle
from hedging.kalshi.forward_curve import contracts_to_frame, simulate_gpu_forward_curve
from hedging.models.datacenter import DataCenterState, build_exposure_frame
from hedging.models.hardware import DEFAULT_FLEETS
from hedging.models.power import DEFAULT_REGIONS
from hedging.ornn import OrnnError, fetch_current_price
from hedging.risk.hedge_ratio import apply_hedge_pnl


@dataclass
class SimulationResult:
    exposure: pd.DataFrame
    hedged: pd.DataFrame
    curve: pd.DataFrame
    execution: ExecutionReport
    summary: dict


def run_pipeline(
    site_id: str = "iad-h100-01",
    gpu_model: str = "H100",
    region: str = "us-east-1",
    horizon_months: int = 12,
    min_margin: float = 0.18,
    seed: int = 42,
    rate_per_kwh: float | None = None,
    n_gpus: int | None = None,
    utilization: float | None = None,
    use_ornn: bool = True,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    if gpu_model not in DEFAULT_FLEETS:
        raise KeyError(
            f"Unknown GPU model {gpu_model!r}; expected one of {sorted(DEFAULT_FLEETS)}"
        )
    fleet = DEFAULT_FLEETS[gpu_model]
    if n_gpus is not None:
        fleet = replace(fleet, n_gpus=max(1, int(n_gpus)))
    if utilization is not None:
        fleet = replace(fleet, utilization=float(np.clip(utilization, 0.01, 1.0)))
    power = DEFAULT_REGIONS[region]
    if rate_per_kwh is not None:
        power = replace(power, rate_per_kwh=float(rate_per_kwh))
    state = DataCenterState(
        site_id=site_id,
        fleet=fleet,
        power=power,
        horizon_months=horizon_months,
        min_operating_margin=min_margin,
    )

    spot_anchor = None
    ornn_meta: dict = {"price_source": "simulated", "ornn_error": None}
    if use_ornn:
        try:
            quote = fetch_current_price(gpu_model)
            spot_anchor = float(quote.price_per_gpu_hour)
            ornn_meta = {
                "price_source": "ornn",
                "ornn_gpu": quote.ornn_name,
                "ornn_price_per_gpu_hour": quote.price_per_gpu_hour,
                "ornn_last_updated": quote.last_updated,
                "ornn_error": None,
            }
        except (OrnnError, requests.RequestException, OSError, ValueError, KeyError, TypeError) as exc:
            ornn_meta = {"price_source": "simulated", "ornn_error": str(exc)}

    exposure = build_exposure_frame(state, spot_anchor=spot_anchor, rng=rng)
    asof = pd.Timestamp.now(tz="UTC")
    spot0 = float(exposure["spot_per_gpu_hour"].iloc[0])
    contracts = simulate_gpu_forward_curve(asof, spot0, rng=rng)
    curve = contracts_to_frame(contracts, asof)

    execution = run_execution_cycle(exposure, curve, contracts, min_margin=min_margin, asof=asof, store=None)
    entry = float(curve.sort_values("tenor_months").iloc[0]["mid"])
    hedged = apply_hedge_pnl(exposure, execution.hedge, exposure["spot_per_gpu_hour"].to_numpy(), entry)

    unhedged_vol = float(exposure["operating_income"].std())
    hedged_vol = float(hedged["hedged_operating_income"].std())
    floor_hit_unhedged = float((exposure["operating_margin"] < min_margin).mean())
    floor_hit_hedged = float((hedged["hedged_margin"] < min_margin).mean())

    summary = {
        "site_id": site_id,
        "gpu_model": gpu_model,
        "region": region,
        "rate_per_kwh": power.rate_per_kwh,
        "n_gpus": fleet.n_gpus,
        "utilization": fleet.utilization,
        "horizon_months": horizon_months,
        "min_margin": min_margin,
        "hedge_ratio": execution.hedge.hedge_ratio,
        "hedge_contracts": execution.hedge.contracts,
        "hedge_notional_usd": execution.hedge.notional_usd,
        "unhedged_total_pnl": float(exposure["operating_income"].sum()),
        "hedged_total_pnl": float(hedged["hedged_operating_income"].sum()),
        "unhedged_income_vol": unhedged_vol,
        "hedged_income_vol": hedged_vol,
        "vol_reduction_pct": 100.0 * (1.0 - hedged_vol / unhedged_vol) if unhedged_vol else 0.0,
        "months_below_floor_unhedged_pct": 100.0 * floor_hit_unhedged,
        "months_below_floor_hedged_pct": 100.0 * floor_hit_hedged,
        "execution_cleared": execution.executed,
        "spot0_per_gpu_hour": spot0,
        "capital_efficiency_note": (
            "Lower P&L volatility and a verified margin floor improve debt-service coverage, "
            "supporting tighter infrastructure financing spreads."
        ),
        **ornn_meta,
    }
    return SimulationResult(exposure, hedged, curve, execution, summary)


if __name__ == "__main__":
    result = run_pipeline()
    print("=== AI Data Center Hedging Pipeline ===")
    for k, v in result.summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.4f}" if abs(v) < 1e6 else f"  {k}: {v:,.0f}")
        else:
            print(f"  {k}: {v}")
    print("\nAlerts:")
    for a in result.execution.alerts:
        print(f"  [{a.level.value}] {a.code}: {a.message}")
