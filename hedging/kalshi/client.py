"""Kalshi REST API connector for GPU compute forward markets."""

from __future__ import annotations

import base64
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiAuthError(RuntimeError):
    pass


class KalshiClient:
    """
    Thin authenticated client for Kalshi Trade API v2.

    Auth: RSA-PSS signed requests when ``KALSHI_API_KEY_ID`` + private key
    are present; otherwise operates in public/read-only mode where allowed,
    or raises on private endpoints.
    """

    def __init__(
        self,
        api_key_id: str | None = None,
        private_key_path: str | None = None,
        base_url: str | None = None,
        demo: bool | None = None,
        timeout: float = 20.0,
    ) -> None:
        demo = demo if demo is not None else os.getenv("KALSHI_DEMO", "true").lower() == "true"
        self.base_url = (base_url or os.getenv("KALSHI_BASE_URL") or (DEMO_BASE if demo else DEFAULT_BASE)).rstrip("/")
        self.api_key_id = api_key_id or os.getenv("KALSHI_API_KEY_ID") or ""
        key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH") or ""
        self._private_key = self._load_private_key(key_path) if key_path else None
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})

    @staticmethod
    def _load_private_key(path: str):
        try:
            from cryptography.hazmat.primitives import serialization

            pem = Path(path).read_bytes()
            return serialization.load_pem_private_key(pem, password=None)
        except Exception as exc:  # noqa: BLE001
            raise KalshiAuthError(f"Failed to load Kalshi private key at {path}: {exc}") from exc

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        if self._private_key is None:
            raise KalshiAuthError("Private key required for authenticated requests")
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        # Strip query string for signature path per Kalshi docs
        path_only = path.split("?")[0]
        message = f"{timestamp_ms}{method.upper()}{path_only}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _headers(self, method: str, path: str) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key_id and self._private_key is not None:
            ts = str(int(time.time() * 1000))
            headers["KALSHI-ACCESS-KEY"] = self.api_key_id
            headers["KALSHI-ACCESS-TIMESTAMP"] = ts
            headers["KALSHI-ACCESS-SIGNATURE"] = self._sign(ts, method, path)
        return headers

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url_path = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}{url_path}"
        headers = self._headers(method, url_path)
        resp = self.session.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(f"Kalshi API {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def get_markets(self, series_ticker: str | None = None, status: str = "open", limit: int = 200) -> pd.DataFrame:
        params: dict[str, Any] = {"limit": limit, "status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker
        data = self.request("GET", "/markets", params=params)
        markets = data.get("markets", data.get("market", []))
        return pd.DataFrame(markets)

    def get_orderbook(self, ticker: str, depth: int = 20) -> dict[str, Any]:
        return self.request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self.request("GET", f"/markets/{ticker}")

    def get_trades(self, ticker: str, limit: int = 1000) -> pd.DataFrame:
        data = self.request("GET", "/markets/trades", params={"ticker": ticker, "limit": limit})
        return pd.DataFrame(data.get("trades", []))

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> pd.DataFrame:
        """Historical candlesticks for reconstructing a forward curve snapshot."""
        path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
        data = self.request(
            "GET",
            path,
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        candles = data.get("candlesticks", [])
        return pd.DataFrame(candles)

    @property
    def authenticated(self) -> bool:
        return bool(self.api_key_id and self._private_key is not None)

    def fingerprint(self) -> str:
        raw = f"{self.base_url}|{self.api_key_id}|{self.authenticated}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
