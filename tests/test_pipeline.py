"""Smoke tests for the hedging pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hedging.kalshi.forward_curve import ForwardCurveStore, simulate_gpu_forward_curve
from hedging.models.datacenter import DataCenterState, build_exposure_frame
from hedging.models.hardware import DEFAULT_FLEETS, depreciation_curve, fleet_book_value_matrix
from hedging.models.power import DEFAULT_REGIONS, power_cost_matrix
from hedging.pipeline import run_pipeline
from hedging.execution.alerts import block_execution, liquidity_check
from hedging.kalshi.forward_curve import ForwardContract


def test_depreciation_monotonic():
    v = depreciation_curve(36, 30_000)
    assert v[0] == 30_000
    assert np.all(np.diff(v) <= 0)
    assert v[-1] < v[0]


def test_power_matrix_shape():
    c = DEFAULT_REGIONS["us-east-1"]
    df = power_cost_matrix(c, 12, load_kw=10_000)
    assert len(df) == 13
    assert (df["total_power_cost"] > 0).all()


def test_exposure_frame():
    state = DataCenterState("t", DEFAULT_FLEETS["H100"], DEFAULT_REGIONS["us-east-1"], horizon_months=12)
    exp = build_exposure_frame(state)
    assert "operating_margin" in exp.columns
    assert len(exp) == 13


def test_forward_store_roundtrip(tmp_path):
    store = ForwardCurveStore(tmp_path)
    asof = pd.Timestamp("2026-01-01", tz="UTC")
    contracts = simulate_gpu_forward_curve(asof, 2.45)
    store.append_snapshot(contracts, asof)
    curve = store.curve_at(asof + pd.Timedelta(hours=1))
    assert len(curve) == len(contracts)
    px = store.interpolate_price(asof, 6)
    assert px > 0


def test_hedge_pipeline():
    result = run_pipeline(horizon_months=12, seed=1, use_ornn=False)
    assert result.execution.hedge.contracts >= 0
    assert "vol_reduction_pct" in result.summary
    assert "hedge_ratio" in result.summary
    assert result.summary["price_source"] == "simulated"


def test_pipeline_fleet_overrides():
    result = run_pipeline(
        horizon_months=12,
        seed=1,
        use_ornn=False,
        n_gpus=1_024,
        utilization=0.50,
    )
    assert result.summary["n_gpus"] == 1_024
    assert abs(result.summary["utilization"] - 0.50) < 1e-9
    # Halving util vs default 0.72 should shrink month-0 gpu_hours vs a default run
    baseline = run_pipeline(horizon_months=12, seed=1, use_ornn=False, n_gpus=1_024)
    assert float(result.exposure["gpu_hours"].iloc[0]) < float(baseline.exposure["gpu_hours"].iloc[0])


def test_ornn_mapping():
    from hedging.ornn import map_gpu

    assert map_gpu("H100") == "H100 SXM"
    assert map_gpu("H200") == "H200"
    assert map_gpu("A100") == "A100 SXM4"


def test_exposure_uses_gpu_hour_units():
    state = DataCenterState("t", DEFAULT_FLEETS["H100"], DEFAULT_REGIONS["us-east-1"], horizon_months=12)
    exp = build_exposure_frame(state, spot_anchor=2.74)
    assert "gpu_hours" in exp.columns
    assert "spot_per_gpu_hour" in exp.columns
    assert abs(float(exp["spot_per_gpu_hour"].iloc[0]) - 2.74) < 1e-9
    assert float(exp["spot_per_gpu_hour"].iloc[0]) > 1.0  # $/GPU-hr, not $/TFLOP-hr


def test_liquidity_blocks_huge_order():
    c = ForwardContract("T", pd.Timestamp("2027-01-01"), 2.40, 2.39, 2.41, bid_size=10, ask_size=10, open_interest=50)
    alerts = liquidity_check(c, order_contracts=500)
    # Thin book / tiny slice capacity should still block blind full-size sends
    assert any(a.code in {"THIN_BOOK", "SLIPPAGE", "SIZE_VS_LIQUIDITY"} for a in alerts)
    assert block_execution(alerts) or any(a.level.value == "WARNING" for a in alerts)
