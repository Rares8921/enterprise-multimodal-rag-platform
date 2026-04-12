from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class PromQuery:
    name: str
    promql: str


class PrometheusClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def query(self, promql: str) -> Dict[str, Any]:
        if self._client is None:
            raise RuntimeError("PrometheusClient not started")

        resp = await self._client.get(
            f"{self.base_url}/api/v1/query",
            params={"query": promql},
        )
        resp.raise_for_status()
        return resp.json()


def _scalar_from_result(data: Dict[str, Any]) -> Optional[float]:
    try:
        result = data["data"]["result"]
        if not result:
            return None
        # If it's a vector with a single sample, value is [ts, value]
        value = result[0].get("value")
        if not value:
            return None
        return float(value[1])
    except Exception:
        return None


async def run_prometheus_queries(
    prom: PrometheusClient,
    queries: list[PromQuery],
    *,
    on_scalar: callable,
) -> None:
    """Run one polling iteration against Prometheus.

    on_scalar(name, value) is called for each query where a scalar can be extracted.
    """

    for q in queries:
        try:
            data = await prom.query(q.promql)
            val = _scalar_from_result(data)
            if val is not None:
                on_scalar(q.name, val)
        except Exception as e:
            logger.warning("Prometheus query failed (%s): %s", q.name, e)
