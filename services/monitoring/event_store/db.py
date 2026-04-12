from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import asyncpg


@dataclass
class MonitoringEvent:
    tenant_id: str
    service: str
    event_type: str

    request_id: Optional[str] = None
    doc_type: Optional[str] = None

    query_length: Optional[int] = None
    embedding: Optional[list[float]] = None

    # Labels / outcomes
    user_feedback_score: Optional[float] = None
    confidence_score: Optional[float] = None
    model_used: Optional[str] = None

    # Service performance
    latency_ms: Optional[float] = None
    retrieval_count: Optional[int] = None
    cache_hit: Optional[bool] = None
    error_type: Optional[str] = None

    # Data quality
    input_missing_rate: Optional[float] = None
    out_of_range_rate: Optional[float] = None
    schema_hash: Optional[str] = None
    data_freshness_seconds: Optional[float] = None

    # Prediction behavior
    prediction_entropy: Optional[float] = None
    low_confidence: Optional[bool] = None
    fallback_used: Optional[bool] = None

    # Cost
    cost_dollars: Optional[float] = None

    # RAG/task-specific monitoring
    rag_answer_sentences: Optional[int] = None
    rag_citations_count: Optional[int] = None
    rag_has_citations: Optional[bool] = None
    rag_citation_coverage: Optional[float] = None
    rag_groundedness_score: Optional[float] = None

    # Extensible
    tags: Optional[Dict[str, Any]] = None


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS monitoring_events (
  id UUID PRIMARY KEY,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  tenant_id TEXT NOT NULL,
  service TEXT NOT NULL,
  event_type TEXT NOT NULL,

  request_id TEXT,
  doc_type TEXT,

  query_length INTEGER,
  embedding TEXT,

  user_feedback_score DOUBLE PRECISION,
  confidence_score DOUBLE PRECISION,
  model_used TEXT,

  latency_ms DOUBLE PRECISION,
  retrieval_count INTEGER,
  cache_hit BOOLEAN,
  error_type TEXT,

  input_missing_rate DOUBLE PRECISION,
  out_of_range_rate DOUBLE PRECISION,
  schema_hash TEXT,
  data_freshness_seconds DOUBLE PRECISION,

  prediction_entropy DOUBLE PRECISION,
  low_confidence BOOLEAN,
  fallback_used BOOLEAN,

  cost_dollars DOUBLE PRECISION,

  rag_answer_sentences INTEGER,
  rag_citations_count INTEGER,
  rag_has_citations BOOLEAN,
  rag_citation_coverage DOUBLE PRECISION,
  rag_groundedness_score DOUBLE PRECISION,

  tags JSONB
);

CREATE INDEX IF NOT EXISTS idx_monitoring_events_ts ON monitoring_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_monitoring_events_tenant_ts ON monitoring_events(tenant_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_monitoring_events_request_id ON monitoring_events(request_id);
"""


MIGRATIONS_SQL = [
    "ALTER TABLE monitoring_events ADD COLUMN IF NOT EXISTS rag_answer_sentences INTEGER;",
    "ALTER TABLE monitoring_events ADD COLUMN IF NOT EXISTS rag_citations_count INTEGER;",
    "ALTER TABLE monitoring_events ADD COLUMN IF NOT EXISTS rag_has_citations BOOLEAN;",
    "ALTER TABLE monitoring_events ADD COLUMN IF NOT EXISTS rag_citation_coverage DOUBLE PRECISION;",
    "ALTER TABLE monitoring_events ADD COLUMN IF NOT EXISTS rag_groundedness_score DOUBLE PRECISION;",
]


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLE_SQL)
        for sql in MIGRATIONS_SQL:
            await conn.execute(sql)


def _embedding_to_text(embedding: Optional[list[float]]) -> Optional[str]:
    if embedding is None:
        return None
    return json.dumps(embedding)


async def insert_event(pool: asyncpg.Pool, ev: MonitoringEvent) -> uuid.UUID:
    event_id = uuid.uuid4()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO monitoring_events (
              id, tenant_id, service, event_type,
              request_id, doc_type,
              query_length, embedding,
              user_feedback_score, confidence_score, model_used,
              latency_ms, retrieval_count, cache_hit, error_type,
              input_missing_rate, out_of_range_rate, schema_hash, data_freshness_seconds,
              prediction_entropy, low_confidence, fallback_used,
              cost_dollars,
              rag_answer_sentences, rag_citations_count, rag_has_citations, rag_citation_coverage, rag_groundedness_score,
              tags
            ) VALUES (
              $1, $2, $3, $4,
              $5, $6,
              $7, $8,
              $9, $10, $11,
              $12, $13, $14, $15,
              $16, $17, $18, $19,
              $20, $21, $22,
              $23,
              $24, $25, $26, $27, $28,
              $29
            )
            """,
            event_id,
            ev.tenant_id,
            ev.service,
            ev.event_type,
            ev.request_id,
            ev.doc_type,
            ev.query_length,
            _embedding_to_text(ev.embedding),
            ev.user_feedback_score,
            ev.confidence_score,
            ev.model_used,
            ev.latency_ms,
            ev.retrieval_count,
            ev.cache_hit,
            ev.error_type,
            ev.input_missing_rate,
            ev.out_of_range_rate,
            ev.schema_hash,
            ev.data_freshness_seconds,
            ev.prediction_entropy,
            ev.low_confidence,
            ev.fallback_used,
            ev.cost_dollars,
            ev.rag_answer_sentences,
            ev.rag_citations_count,
            ev.rag_has_citations,
            ev.rag_citation_coverage,
            ev.rag_groundedness_score,
            ev.tags,
        )

    return event_id


async def update_feedback_by_request_id(
    pool: asyncpg.Pool,
    *,
    tenant_id: str,
    request_id: str,
    user_feedback_score: float,
    tags: Optional[Dict[str, Any]] = None,
) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE monitoring_events
            SET user_feedback_score = $1,
                tags = COALESCE(tags, '{}'::jsonb) || COALESCE($2::jsonb, '{}'::jsonb)
            WHERE tenant_id = $3 AND request_id = $4
            """,
            user_feedback_score,
            json.dumps(tags) if tags is not None else None,
            tenant_id,
            request_id,
        )

    # asyncpg returns "UPDATE <n>"
    try:
        updated = int(result.split()[-1])
    except Exception:
        updated = 0
    return updated > 0
