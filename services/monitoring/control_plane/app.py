from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

import aiohttp
import asyncpg
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from services.monitoring.control_plane.config import Settings
from services.monitoring.event_store.db import MonitoringEvent, ensure_schema, insert_event, update_feedback_by_request_id
from services.monitoring.integrations.logging_utils import configure_logging
from services.monitoring.metrics_exporter.metrics import (
    CACHE_HIT_RATE,
    CONCEPT_DRIFT_DETECTED,
    COST_PER_REQUEST,
    DATA_DRIFT_PSI,
    DATA_FRESHNESS_P95_S,
    DB_ERROR_RATE_24H,
    DB_P95_LATENCY_MS_24H,
    DB_UP,
    EMBEDDING_CENTROID_DISTANCE,
    EVENTS_INGESTED,
    EVENTS_INGEST_FAILED,
    FALLBACK_RATE,
    FEEDBACK_SCORE_AVG,
    INPUT_MISSING_RATE,
    LOW_CONFIDENCE_RATE,
    MONITOR_LOOP_DURATION,
    OUT_OF_RANGE_RATE,
    PREDICTION_ENTROPY_AVG,
    RAG_CITATION_COVERAGE_AVG_24H,
    RAG_CITATIONS_PER_ANSWER_AVG_24H,
    RAG_GROUNDEDNESS_AVG_24H,
    RAG_HAS_CITATIONS_RATE_24H,
    RAG_UNGROUNDED_RATE_24H,
    SCHEMA_HASH_DISTINCT,
    SLO_ERROR_RATE,
    SLO_P95_LATENCY_S,
    SLO_REQUEST_RATE,
    TENANT_FEEDBACK_SPREAD,
)
from services.monitoring.integrations.prometheus_collector import PrometheusClient, PromQuery, run_prometheus_queries
from services.monitoring.integrations.otel import init_otel
from services.monitoring.modules.rag_monitoring.signals import extract_rag_fields

logger = logging.getLogger(__name__)

app = FastAPI(title="Monitoring Control Plane", version="1.0.0")
settings = Settings()

# Expose /metrics
app.mount("/metrics", make_asgi_app())

# Global state
pg_pool: Optional[asyncpg.Pool] = None


async def send_slack_alert(text: str) -> None:
    if not settings.slack_webhook_url:
        return

    try:
        async with aiohttp.ClientSession() as session:
            await session.post(settings.slack_webhook_url, json={"text": text})
    except Exception as e:
        logger.warning("Failed to send Slack alert: %s", e)


def _compute_entropy_from_confidence(conf: float) -> float:
    # Proxy entropy for a binary-ish confidence. Not perfect, but stable.
    p = min(max(conf, 1e-6), 1 - 1e-6)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


@app.on_event("startup")
async def startup() -> None:
    global pg_pool

    configure_logging()
    init_otel()

    # Optionally instrument FastAPI if OTel is enabled
    try:
        instrumentor = getattr(init_otel, "instrument_fastapi", None)
        if instrumentor is not None:
            instrumentor().instrument_app(app)
    except Exception:
        pass

    pg_pool = await asyncpg.create_pool(settings.postgres_url)
    await ensure_schema(pg_pool)

    DB_UP.set(1)

    # Start background tasks
    asyncio.create_task(_drift_and_quality_loop())
    asyncio.create_task(_prometheus_slo_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    global pg_pool
    if pg_pool is not None:
        await pg_pool.close()
        pg_pool = None


@app.get("/health")
async def health() -> Dict[str, Any]:
    ok = True
    details: Dict[str, Any] = {"service": "monitoring", "db": "unknown"}

    try:
        if pg_pool is None:
            raise RuntimeError("pool is None")
        async with pg_pool.acquire() as conn:
            await conn.execute("SELECT 1")
        details["db"] = "ok"
    except Exception as e:
        ok = False
        details["db"] = f"down: {e}"

    return {"status": "healthy" if ok else "degraded", **details}


@app.post("/events")
async def ingest_event(payload: Dict[str, Any]):
    if pg_pool is None:
        raise HTTPException(status_code=503, detail="DB not ready")

    # Minimal validation (kept flexible intentionally)
    try:
        tenant_id = str(payload["tenant_id"])
        service = str(payload.get("service", "unknown"))
        event_type = str(payload.get("event_type", "custom"))

        rag = extract_rag_fields(payload)

        ev = MonitoringEvent(
            tenant_id=tenant_id,
            service=service,
            event_type=event_type,
            request_id=payload.get("request_id"),
            doc_type=payload.get("doc_type"),
            query_length=payload.get("query_length"),
            embedding=payload.get("embedding"),
            user_feedback_score=payload.get("user_feedback_score"),
            confidence_score=payload.get("confidence_score"),
            model_used=payload.get("model_used"),
            latency_ms=payload.get("latency_ms"),
            retrieval_count=payload.get("retrieval_count"),
            cache_hit=payload.get("cache_hit"),
            error_type=payload.get("error_type"),
            input_missing_rate=payload.get("input_missing_rate"),
            out_of_range_rate=payload.get("out_of_range_rate"),
            schema_hash=payload.get("schema_hash"),
            data_freshness_seconds=payload.get("data_freshness_seconds"),
            prediction_entropy=payload.get("prediction_entropy"),
            low_confidence=payload.get("low_confidence"),
            fallback_used=payload.get("fallback_used"),
            cost_dollars=payload.get("cost_dollars"),
            rag_answer_sentences=rag.get("rag_answer_sentences"),
            rag_citations_count=rag.get("rag_citations_count"),
            rag_has_citations=rag.get("rag_has_citations"),
            rag_citation_coverage=rag.get("rag_citation_coverage"),
            rag_groundedness_score=rag.get("rag_groundedness_score"),
            tags=payload.get("tags"),
        )

        await insert_event(pg_pool, ev)
        EVENTS_INGESTED.labels(service=service, event_type=event_type).inc()
        return JSONResponse({"status": "ok"})

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        EVENTS_INGEST_FAILED.labels(service=str(payload.get("service", "unknown"))).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(payload: Dict[str, Any]):
    if pg_pool is None:
        raise HTTPException(status_code=503, detail="DB not ready")

    try:
        tenant_id = str(payload["tenant_id"])
        request_id = str(payload["request_id"])
        score = float(payload["user_feedback_score"])
        tags = payload.get("tags")

        updated = await update_feedback_by_request_id(
            pg_pool,
            tenant_id=tenant_id,
            request_id=request_id,
            user_feedback_score=score,
            tags=tags,
        )
        if not updated:
            # Insert as standalone feedback event
            ev = MonitoringEvent(
                tenant_id=tenant_id,
                service=str(payload.get("service", "inference-api")),
                event_type="feedback",
                request_id=request_id,
                user_feedback_score=score,
                tags=tags,
            )
            await insert_event(pg_pool, ev)

        return JSONResponse({"status": "ok", "updated": updated})

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _drift_and_quality_loop() -> None:
    """Compute drift + data quality + prediction behavior + fairness proxy from monitoring_events."""

    # Import DriftDetector directly from drift_detection package (best-effort)
    try:
        from drift_detection.DriftDetector import DriftDetector  # type: ignore
    except Exception:
        DriftDetector = None  # type: ignore

    detector = DriftDetector() if DriftDetector else None

    while True:
        with MONITOR_LOOP_DURATION.labels(loop="db_analytics").time():
            try:
                if pg_pool is None:
                    await asyncio.sleep(5)
                    continue

                async with pg_pool.acquire() as conn:
                    tenants = await conn.fetch(
                        """
                        SELECT DISTINCT tenant_id
                        FROM monitoring_events
                        WHERE timestamp >= NOW() - INTERVAL '24 hours'
                        """
                    )

                    # Fairness proxy: spread in avg feedback across tenants
                    feedback_avgs = []

                    for t in tenants:
                        tenant_id = t["tenant_id"]

                        rows = await conn.fetch(
                            """
                            SELECT
                              confidence_score,
                              user_feedback_score,
                              latency_ms,
                              cache_hit,
                              fallback_used,
                              event_type,
                              error_type,
                              input_missing_rate,
                              out_of_range_rate,
                              data_freshness_seconds,
                              schema_hash,
                              rag_answer_sentences,
                              rag_citations_count,
                              rag_has_citations,
                              rag_citation_coverage,
                              rag_groundedness_score
                            FROM monitoring_events
                            WHERE tenant_id = $1
                              AND timestamp >= NOW() - INTERVAL '24 hours'
                            """,
                            tenant_id,
                        )

                        # Prediction behavior
                        confs = [r["confidence_score"] for r in rows if r["confidence_score"] is not None]
                        if confs:
                            low_rate = float(
                                sum(1 for c in confs if c < settings.low_confidence_threshold) / len(confs)
                            )
                            LOW_CONFIDENCE_RATE.labels(tenant_id=tenant_id).set(low_rate)

                            ent = float(np.mean([_compute_entropy_from_confidence(c) for c in confs]))
                            PREDICTION_ENTROPY_AVG.labels(tenant_id=tenant_id).set(ent)

                        # Performance (from events)
                        lat_ms = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
                        if lat_ms:
                            DB_P95_LATENCY_MS_24H.labels(tenant_id=tenant_id).set(float(np.percentile(lat_ms, 95)))

                        # Error rate proxy (from events)
                        total = len(rows)
                        if total:
                            err = sum(
                                1
                                for r in rows
                                if (r["event_type"] in {"error", "inference_error"}) or (r["error_type"] is not None)
                            )
                            DB_ERROR_RATE_24H.labels(tenant_id=tenant_id).set(float(err / total))

                        # Cache hit rate (if present)
                        cache_vals = [r["cache_hit"] for r in rows if r["cache_hit"] is not None]
                        if cache_vals:
                            CACHE_HIT_RATE.labels(tenant_id=tenant_id).set(float(sum(1 for v in cache_vals if v) / len(cache_vals)))

                        # Fallback rate (if present)
                        fb_vals = [r["fallback_used"] for r in rows if r["fallback_used"] is not None]
                        if fb_vals:
                            FALLBACK_RATE.labels(tenant_id=tenant_id).set(float(sum(1 for v in fb_vals if v) / len(fb_vals)))

                        # Data quality
                        miss_vals = [r["input_missing_rate"] for r in rows if r["input_missing_rate"] is not None]
                        if miss_vals:
                            INPUT_MISSING_RATE.labels(tenant_id=tenant_id).set(float(np.mean(miss_vals)))

                        oor_vals = [r["out_of_range_rate"] for r in rows if r["out_of_range_rate"] is not None]
                        if oor_vals:
                            OUT_OF_RANGE_RATE.labels(tenant_id=tenant_id).set(float(np.mean(oor_vals)))

                        fresh_vals = [r["data_freshness_seconds"] for r in rows if r["data_freshness_seconds"] is not None]
                        if fresh_vals:
                            DATA_FRESHNESS_P95_S.labels(tenant_id=tenant_id).set(float(np.percentile(fresh_vals, 95)))

                        schema_vals = [r["schema_hash"] for r in rows if r["schema_hash"]]
                        if schema_vals:
                            SCHEMA_HASH_DISTINCT.labels(tenant_id=tenant_id).set(float(len(set(schema_vals))))

                        # RAG / groundedness & citations (task-specific)
                        cov_vals = [r["rag_citation_coverage"] for r in rows if r["rag_citation_coverage"] is not None]
                        if cov_vals:
                            cov_avg = float(np.mean(cov_vals))
                            RAG_CITATION_COVERAGE_AVG_24H.labels(tenant_id=tenant_id).set(cov_avg)
                            if (
                                settings.alert_on_rag
                                and len(cov_vals) >= settings.rag_min_samples_for_alert
                                and cov_avg < settings.rag_citation_coverage_threshold
                            ):
                                await send_slack_alert(
                                    f"RAG ALERT: low citation coverage tenant={tenant_id} avg_24h={cov_avg:.3f}"
                                )

                        has_vals = [r["rag_has_citations"] for r in rows if r["rag_has_citations"] is not None]
                        if has_vals:
                            has_rate = float(sum(1 for v in has_vals if v) / len(has_vals))
                            RAG_HAS_CITATIONS_RATE_24H.labels(tenant_id=tenant_id).set(has_rate)

                        cit_counts = [r["rag_citations_count"] for r in rows if r["rag_citations_count"] is not None]
                        if cit_counts:
                            RAG_CITATIONS_PER_ANSWER_AVG_24H.labels(tenant_id=tenant_id).set(float(np.mean(cit_counts)))

                        g_vals = [r["rag_groundedness_score"] for r in rows if r["rag_groundedness_score"] is not None]
                        if g_vals:
                            g_avg = float(np.mean(g_vals))
                            RAG_GROUNDEDNESS_AVG_24H.labels(tenant_id=tenant_id).set(g_avg)
                            unground_rate = float(
                                sum(1 for v in g_vals if v < settings.rag_groundedness_threshold) / len(g_vals)
                            )
                            RAG_UNGROUNDED_RATE_24H.labels(tenant_id=tenant_id).set(unground_rate)
                            if (
                                settings.alert_on_rag
                                and len(g_vals) >= settings.rag_min_samples_for_alert
                                and unground_rate > 0.2
                            ):
                                await send_slack_alert(
                                    f"RAG ALERT: ungrounded rate tenant={tenant_id} rate_24h={unground_rate:.3f} (thr={settings.rag_groundedness_threshold})"
                                )

                        # Fairness proxy input
                        fbs = [r["user_feedback_score"] for r in rows if r["user_feedback_score"] is not None]
                        if fbs:
                            feedback_avgs.append(float(np.mean(fbs)))

                    if feedback_avgs:
                        TENANT_FEEDBACK_SPREAD.set(float(max(feedback_avgs) - min(feedback_avgs)))

                # Optional: run drift detection using existing drift logic and alerting
                if detector is not None and settings.enable_drift_detection:
                    await _run_drift_cycle(detector)

            except Exception as e:
                logger.warning("Monitoring analytics loop error: %s", e)

        await asyncio.sleep(settings.analytics_interval_seconds)


async def _run_drift_cycle(detector) -> None:
    """Run data/concept/embedding drift using the existing DriftDetector implementation."""

    if pg_pool is None:
        return

    async with pg_pool.acquire() as conn:
        tenants = await conn.fetch(
            "SELECT DISTINCT tenant_id FROM monitoring_events WHERE timestamp >= NOW() - INTERVAL '24 hours'"
        )

        for tenant_record in tenants:
            tenant_id = tenant_record["tenant_id"]

            recent = await conn.fetch(
                """
                SELECT query_length, embedding, user_feedback_score
                FROM monitoring_events
                WHERE tenant_id = $1 AND timestamp >= NOW() - INTERVAL '24 hours'
                """,
                tenant_id,
            )
            baseline = await conn.fetch(
                """
                SELECT query_length, embedding, user_feedback_score
                FROM monitoring_events
                WHERE tenant_id = $1
                  AND timestamp >= NOW() - INTERVAL '8 days'
                  AND timestamp < NOW() - INTERVAL '24 hours'
                """,
                tenant_id,
            )

            # Data drift (query_length)
            recent_lengths = [r["query_length"] for r in recent if r["query_length"] is not None]
            baseline_lengths = [b["query_length"] for b in baseline if b["query_length"] is not None]
            if recent_lengths and baseline_lengths:
                data_drift = await detector.detect_data_drift(
                    "query_length",
                    recent_lengths,
                    baseline_lengths,
                )
                DATA_DRIFT_PSI.labels(feature="query_length", tenant_id=tenant_id).set(float(data_drift.get("psi", 0.0)))
                if data_drift.get("drift_detected") and settings.alert_on_drift:
                    await send_slack_alert(
                        f"DRIFT ALERT [Data Drift] tenant={tenant_id} psi={data_drift.get('psi')} severity={data_drift.get('severity')}"
                    )

            # Concept drift (feedback)
            recent_fb = [r["user_feedback_score"] for r in recent if r["user_feedback_score"] is not None]
            baseline_fb = [b["user_feedback_score"] for b in baseline if b["user_feedback_score"] is not None]
            if recent_fb:
                FEEDBACK_SCORE_AVG.labels(tenant_id=tenant_id).set(float(np.mean(recent_fb)))
            if recent_fb and baseline_fb:
                concept = await detector.detect_concept_drift(
                    float(np.mean(recent_fb)),
                    float(np.mean(baseline_fb)),
                    settings.concept_drift_threshold,
                )
                if concept.get("drift_detected"):
                    CONCEPT_DRIFT_DETECTED.labels(tenant_id=tenant_id).inc()
                    if settings.alert_on_drift:
                        await send_slack_alert(
                            f"DRIFT ALERT [Concept Drift] tenant={tenant_id} accuracy_drop={concept.get('accuracy_drop')} severity={concept.get('severity')}"
                        )

            # Embedding drift (sample to cap cost)
            def _parse_embedding(x: Any) -> Optional[list[float]]:
                if not x:
                    return None
                try:
                    import json

                    return json.loads(x)
                except Exception:
                    return None

            recent_emb = [_parse_embedding(r["embedding"]) for r in recent]
            base_emb = [_parse_embedding(b["embedding"]) for b in baseline]
            recent_emb = [e for e in recent_emb if e is not None]
            base_emb = [e for e in base_emb if e is not None]

            if recent_emb and base_emb:
                # cap to 200 each
                recent_arr = np.array(recent_emb[:200], dtype=np.float32)
                base_arr = np.array(base_emb[:200], dtype=np.float32)
                emb = await detector.detect_embedding_drift(recent_arr, base_arr)
                EMBEDDING_CENTROID_DISTANCE.labels(tenant_id=tenant_id).set(float(emb.get("centroid_distance", 0.0)))
                if emb.get("drift_detected") and settings.alert_on_drift:
                    await send_slack_alert(
                        f"DRIFT ALERT [Embedding Drift] tenant={tenant_id} centroid_distance={emb.get('centroid_distance')}"
                    )


async def _prometheus_slo_loop() -> None:
    if not settings.prometheus_url:
        return

    prom = PrometheusClient(settings.prometheus_url, timeout_s=settings.prometheus_timeout_seconds)

    queries = [
        PromQuery(
            name="inference_rps",
            promql="sum(rate(query_requests_total[5m]))",
        ),
        PromQuery(
            name="inference_error_rate",
            promql="(sum(rate(query_failures_total[5m])) / clamp_min(sum(rate(query_requests_total[5m])), 1))",
        ),
        PromQuery(
            name="inference_p95_latency",
            promql="histogram_quantile(0.95, sum(rate(query_duration_seconds_bucket[5m])) by (le))",
        ),
        PromQuery(
            name="llm_cost_per_req_mistral",
            promql="(sum(rate(llm_cost_dollars{model=\"mistral\"}[5m])) / clamp_min(sum(rate(llm_requests_total{model=\"mistral\"}[5m])), 1))",
        ),
        PromQuery(
            name="llm_cost_per_req_gemini",
            promql="(sum(rate(llm_cost_dollars{model=\"gemini\"}[5m])) / clamp_min(sum(rate(llm_requests_total{model=\"gemini\"}[5m])), 1))",
        ),
    ]

    def on_scalar(name: str, value: float) -> None:
        if name == "inference_rps":
            SLO_REQUEST_RATE.labels(service="inference-api").set(value)
        elif name == "inference_error_rate":
            SLO_ERROR_RATE.labels(service="inference-api").set(value)
            if value > settings.error_rate_alert_threshold and settings.alert_on_slo:
                asyncio.create_task(send_slack_alert(f"SLO ALERT: inference-api error_rate_5m={value:.3f}"))
        elif name == "inference_p95_latency":
            SLO_P95_LATENCY_S.labels(service="inference-api").set(value)
            if value > settings.p95_latency_seconds_alert_threshold and settings.alert_on_slo:
                asyncio.create_task(send_slack_alert(f"SLO ALERT: inference-api p95_latency_5m={value:.3f}s"))
        elif name == "llm_cost_per_req_mistral":
            COST_PER_REQUEST.labels(model="mistral").set(value)
        elif name == "llm_cost_per_req_gemini":
            COST_PER_REQUEST.labels(model="gemini").set(value)

    await prom.start()
    try:
        while True:
            with MONITOR_LOOP_DURATION.labels(loop="prometheus_slo").time():
                await run_prometheus_queries(
                    prom,
                    queries,
                    on_scalar=on_scalar,
                )
            await asyncio.sleep(settings.prometheus_poll_interval_seconds)
    finally:
        await prom.stop()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("MONITORING_PORT", str(settings.monitoring_port)))
    uvicorn.run(app, host="0.0.0.0", port=port)
