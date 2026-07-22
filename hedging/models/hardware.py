"""Vectorized HPC hardware depreciation curves (H100 / H200 fleets)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

GpuModel = Literal["H100", "H200", "A100"]


@dataclass(frozen=True)
class FleetSpec:
    """Physical fleet composition for a single data-center site."""

    model: GpuModel
    n_gpus: int
    purchase_price_per_gpu: float
    residual_value_frac: float = 0.08
    lifecycle_months: int = 36
    tdp_watts: float = 700.0
    peak_tflops: float = 989.0  # FP16 Tensor for H100-class
    utilization: float = 0.72


# Typical list economics (illustrative; tune per contract).
DEFAULT_FLEETS: dict[GpuModel, FleetSpec] = {
    "H100": FleetSpec("H100", n_gpus=8_192, purchase_price_per_gpu=30_000.0, tdp_watts=700.0, peak_tflops=989.0),
    "H200": FleetSpec("H200", n_gpus=4_096, purchase_price_per_gpu=40_000.0, tdp_watts=700.0, peak_tflops=1979.0),
    "A100": FleetSpec("A100", n_gpus=4_096, purchase_price_per_gpu=12_000.0, tdp_watts=400.0, peak_tflops=312.0),
}


def depreciation_curve(
    months: int,
    purchase_price: float,
    residual_frac: float = 0.08,
    method: Literal["declining_balance", "straight_line", "double_declining"] = "declining_balance",
    declining_rate: float = 0.045,
) -> np.ndarray:
    """
    Return book value path of length ``months + 1`` (t=0 .. t=months).

    Declining-balance defaults approximate rapid GPU obsolescence under
    successive architecture releases.
    """
    t = np.arange(months + 1, dtype=float)
    residual = purchase_price * residual_frac

    if method == "straight_line":
        values = purchase_price - (purchase_price - residual) * (t / months)
    elif method == "double_declining":
        rate = 2.0 / months
        values = purchase_price * (1.0 - rate) ** t
        values = np.maximum(values, residual)
    else:  # declining_balance — smooth exponential toward residual
        values = residual + (purchase_price - residual) * np.exp(-declining_rate * t)

    return values.astype(float)


def fleet_book_value_matrix(
    fleet: FleetSpec,
    horizon_months: int | None = None,
    method: Literal["declining_balance", "straight_line", "double_declining"] = "declining_balance",
) -> pd.DataFrame:
    """
    Vectorized book-value time series for the full fleet.

    Columns: month, book_value_per_gpu, fleet_book_value, cumulative_depreciation,
    remaining_life_frac, effective_compute_capacity (utilization-adjusted TFLOPS).
    """
    months = horizon_months or fleet.lifecycle_months
    per_gpu = depreciation_curve(
        months,
        fleet.purchase_price_per_gpu,
        fleet.residual_value_frac,
        method=method,
    )
    total = per_gpu * fleet.n_gpus
    purchase_total = fleet.purchase_price_per_gpu * fleet.n_gpus

    # Effective capacity decays gently with age (thermal / reliability haircut).
    age_haircut = 1.0 - 0.15 * (np.arange(months + 1) / max(months, 1))
    effective_tflops = fleet.n_gpus * fleet.peak_tflops * fleet.utilization * age_haircut

    return pd.DataFrame(
        {
            "month": np.arange(months + 1),
            "book_value_per_gpu": per_gpu,
            "fleet_book_value": total,
            "cumulative_depreciation": purchase_total - total,
            "remaining_life_frac": per_gpu / fleet.purchase_price_per_gpu,
            "effective_tflops": effective_tflops,
            "power_kw": fleet.n_gpus * fleet.tdp_watts / 1000.0,
        }
    )


def monthly_depreciation_expense(book_values: pd.DataFrame) -> pd.Series:
    """First difference of fleet book value (positive = expense)."""
    return (-book_values["fleet_book_value"].diff()).fillna(0.0)
