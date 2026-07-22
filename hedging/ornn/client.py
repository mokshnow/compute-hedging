"""Ornn Compute Price Index (OCPI) client — public free-tier API."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import requests

BASE_URL = "https://api.ornnai.com"

# App GPU labels → Ornn free-tier identifiers
GPU_TO_ORNN: dict[str, str] = {
    "H100": "H100 SXM",
    "H200": "H200",
    "A100": "A100 SXM4",
}


@dataclass(frozen=True)
class OrnnQuote:
    gpu_model: str
    ornn_name: str
    price_per_gpu_hour: float
    last_updated: str
    source: str = "ornn"


class OrnnError(RuntimeError):
    pass


def map_gpu(gpu_model: str) -> str:
    raw = gpu_model.strip()
    if raw in GPU_TO_ORNN:
        return GPU_TO_ORNN[raw]
    if raw in GPU_TO_ORNN.values():
        return raw
    key = raw.upper().replace(" ", "")
    for app_name, ornn_name in GPU_TO_ORNN.items():
        if key == app_name.upper() or key == ornn_name.upper().replace(" ", ""):
            return ornn_name
    raise OrnnError(f"No Ornn mapping for GPU model {gpu_model!r}")


def fetch_current_price(gpu_model: str, timeout: float = 12.0) -> OrnnQuote:
    """
    Latest OCPI $/GPU-hour for a mapped GPU (free tier, no API key).

    Docs: https://dashboard.ornnai.com/docs
    """
    ornn_name = map_gpu(gpu_model)
    url = f"{BASE_URL}/api/gpu/{quote(ornn_name)}"
    resp = requests.get(url, timeout=timeout)
    if resp.status_code >= 400:
        raise OrnnError(f"Ornn API {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    if not payload.get("success"):
        raise OrnnError(f"Ornn API error payload: {payload}")
    data = payload["data"]
    return OrnnQuote(
        gpu_model=gpu_model,
        ornn_name=str(data.get("gpu_name", ornn_name)),
        price_per_gpu_hour=float(data["index_value"]),
        last_updated=str(data.get("last_updated", "")),
    )


def fetch_daily_index(gpu_model: str, timeout: float = 12.0) -> OrnnQuote:
    """Most recent settled daily OCPI close."""
    ornn_name = map_gpu(gpu_model)
    resp = requests.get(
        f"{BASE_URL}/api/daily-index",
        params={"gpu": ornn_name},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise OrnnError(f"Ornn daily-index {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    if not payload.get("success"):
        raise OrnnError(f"Ornn daily-index error: {payload}")
    data = payload["data"]
    return OrnnQuote(
        gpu_model=gpu_model,
        ornn_name=str(data.get("gpu_type", ornn_name)),
        price_per_gpu_hour=float(data["index_value"]),
        last_updated=str(data.get("date", "")),
        source="ornn-daily",
    )
