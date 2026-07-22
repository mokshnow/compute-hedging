"""Data-center balance-sheet state: hardware + power + operating margins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hedging.models.hardware import FleetSpec, fleet_book_value_matrix
from hedging.models.power import PowerContract, power_cost_matrix


@dataclass
class DataCenterState:
    """Simulated balance-sheet path for one site over a hedging horizon."""

    site_id: str
    fleet: FleetSpec
    power: PowerContract
    horizon_months: int = 24
    opex_other_monthly: float = 650_000.0
    min_operating_margin: float = 0.18  # target floor on (rev - cost) / rev


def build_exposure_frame(
    state: DataCenterState,
    spot_price_per_gpu_hour: np.ndarray | None = None,
    spot_anchor: float | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Construct the unhedged P&L path.

    Maps hardware GPU-hours against spot compute prices in **$/GPU-hour**
    (Ornn OCPI units).

    If ``spot_anchor`` is set (e.g. live Ornn $/GPU-hr), the simulated path
    is rescaled so month-0 matches that live level.
    """
    rng = rng or np.random.default_rng(42)
    hw = fleet_book_value_matrix(state.fleet, state.horizon_months)
    power = power_cost_matrix(state.power, state.horizon_months, hw["power_kw"].to_numpy())

    hours = 730.0
    # Util + age-adjusted GPU count implied by effective TFLOPS / peak TFLOPS.
    effective_gpus = hw["effective_tflops"].to_numpy() / max(state.fleet.peak_tflops, 1e-9)
    gpu_hours = effective_gpus * hours

    if spot_price_per_gpu_hour is None:
        # Simulated $/GPU-hr path (H100-ish levels); Ornn anchor rescales when present.
        n = state.horizon_months + 1
        base = 2.35
        noise = rng.normal(0, 0.05, size=n)
        drift = -0.01 * np.arange(n)
        jumps = rng.choice([0.0, -0.12, -0.22], size=n, p=[0.82, 0.12, 0.06])
        spot = np.maximum(base + np.cumsum(noise) * 0.4 + drift + np.cumsum(jumps) * 0.2, 0.80)
        if spot_anchor is not None and spot[0] > 0:
            spot = np.maximum(spot * (float(spot_anchor) / float(spot[0])), 1e-6)
    else:
        spot = np.asarray(spot_price_per_gpu_hour, dtype=float)

    revenue = gpu_hours * spot
    purchase_total = state.fleet.purchase_price_per_gpu * state.fleet.n_gpus
    residual = purchase_total * state.fleet.residual_value_frac
    months_life = max(state.fleet.lifecycle_months, 1)
    depreciation = np.full(len(hw), (purchase_total - residual) / months_life)
    power_cost = power["total_power_cost"].to_numpy()
    other = np.full(len(hw), state.opex_other_monthly)
    total_cost = depreciation + power_cost + other
    operating_income = revenue - total_cost
    margin = np.where(revenue > 0, operating_income / revenue, np.nan)

    out = hw.copy()
    out["spot_per_gpu_hour"] = spot
    out["gpu_hours"] = gpu_hours
    out["revenue"] = revenue
    out["depreciation"] = depreciation
    out["power_cost"] = power_cost
    out["other_opex"] = other
    out["total_cost"] = total_cost
    out["operating_income"] = operating_income
    out["operating_margin"] = margin
    out["margin_vs_floor"] = margin - state.min_operating_margin
    out["site_id"] = state.site_id
    out["region"] = state.power.region
    return out
