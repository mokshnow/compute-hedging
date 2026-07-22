"""Margin threshold mapping: hardware/power overhead vs GPU forward curve."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cost_breakeven_spot(exposure: pd.DataFrame, min_margin: float) -> pd.Series:
    """
    Minimum spot ($/GPU-hour) required to hit ``min_margin``.

    revenue = gpu_hours * spot
    (rev - cost) / rev >= min_margin  =>  spot >= cost / (gpu_hours * (1 - min_margin))
    """
    denom = exposure["gpu_hours"] * (1.0 - min_margin)
    return exposure["total_cost"] / denom.replace(0, np.nan)


def margin_spread_vs_forward(
    exposure: pd.DataFrame,
    forward_curve: pd.DataFrame,
    min_margin: float,
) -> pd.DataFrame:
    """
    For each month, compare required breakeven spot to the matching forward mid.

    Positive ``cushion`` means the forward locks a price above the margin floor.
    """
    be = cost_breakeven_spot(exposure, min_margin)
    curve = forward_curve.sort_values("tenor_months")
    tenors = curve["tenor_months"].to_numpy(dtype=float)
    mids = curve["mid"].to_numpy(dtype=float)

    months = exposure["month"].to_numpy(dtype=float)
    fwd = np.interp(months, tenors, mids)

    gpu_hours = exposure["gpu_hours"].to_numpy()
    out = exposure[["month", "spot_per_gpu_hour", "operating_margin", "revenue", "total_cost"]].copy()
    out["breakeven_spot"] = be.to_numpy()
    out["forward_mid"] = fwd
    out["cushion"] = fwd - be.to_numpy()
    out["margin_at_forward"] = np.where(
        gpu_hours * fwd > 0,
        1.0 - exposure["total_cost"].to_numpy() / (gpu_hours * fwd),
        np.nan,
    )
    out["breach"] = out["cushion"] < 0
    return out
