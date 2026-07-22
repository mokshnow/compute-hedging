"""Contracted power capacity matrices ($/kWh per rack)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PowerContract:
    """Long-term power purchase agreement for a site."""

    region: str
    contracted_kw: float
    rate_per_kwh: float  # base contracted rate
    escalation_annual: float = 0.025
    demand_charge_per_kw_month: float = 12.0
    pue: float = 1.25  # power usage effectiveness


DEFAULT_REGIONS: dict[str, PowerContract] = {
    "us-east-1": PowerContract("us-east-1", contracted_kw=25_000, rate_per_kwh=0.065, pue=1.22),
    "us-west-2": PowerContract("us-west-2", contracted_kw=18_000, rate_per_kwh=0.055, pue=1.18),
    "eu-north-1": PowerContract("eu-north-1", contracted_kw=12_000, rate_per_kwh=0.078, pue=1.15),
    "us-central": PowerContract("us-central", contracted_kw=30_000, rate_per_kwh=0.042, pue=1.20),
}


def power_cost_matrix(
    contract: PowerContract,
    horizon_months: int,
    load_kw: float | np.ndarray,
    hours_per_month: float = 730.0,
) -> pd.DataFrame:
    """
    Build a month × cost-component matrix for contracted power.

    ``load_kw`` may be a scalar or length ``horizon_months + 1`` array of
    IT load; facility draw is load × PUE, capped at contracted capacity.
    """
    months = np.arange(horizon_months + 1)
    load = np.broadcast_to(np.asarray(load_kw, dtype=float), months.shape)
    facility_kw = np.minimum(load * contract.pue, contract.contracted_kw)

    # Escalating energy rate
    annual_factor = (1.0 + contract.escalation_annual) ** (months / 12.0)
    energy_rate = contract.rate_per_kwh * annual_factor

    energy_kwh = facility_kw * hours_per_month
    energy_cost = energy_kwh * energy_rate
    demand_cost = facility_kw * contract.demand_charge_per_kw_month
    # Take-or-pay haircut on unused contracted capacity (10% of demand charge)
    unused_kw = np.maximum(contract.contracted_kw - facility_kw, 0.0)
    take_or_pay = unused_kw * contract.demand_charge_per_kw_month * 0.10

    return pd.DataFrame(
        {
            "month": months,
            "region": contract.region,
            "facility_kw": facility_kw,
            "energy_rate_per_kwh": energy_rate,
            "energy_cost": energy_cost,
            "demand_cost": demand_cost,
            "take_or_pay": take_or_pay,
            "total_power_cost": energy_cost + demand_cost + take_or_pay,
            "cost_per_kwh_effective": (energy_cost + demand_cost + take_or_pay) / np.maximum(energy_kwh, 1.0),
        }
    )


def multi_region_power_cube(
    contracts: dict[str, PowerContract],
    horizon_months: int,
    load_by_region: dict[str, float],
) -> pd.DataFrame:
    """Stack regional power matrices into a single tidy frame."""
    frames = [
        power_cost_matrix(c, horizon_months, load_by_region.get(region, c.contracted_kw * 0.7))
        for region, c in contracts.items()
    ]
    return pd.concat(frames, ignore_index=True)
