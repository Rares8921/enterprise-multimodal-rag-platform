from prometheus_client import Counter, Gauge, Histogram

# Event ingestion
EVENTS_INGESTED = Counter(
    "monitoring_events_ingested_total",
    "Monitoring events ingested",
    ["service", "event_type"],
)
EVENTS_INGEST_FAILED = Counter(
    "monitoring_events_ingest_failures_total",
    "Monitoring event ingest failures",
    ["service"],
)

DB_UP = Gauge("monitoring_db_up", "Monitoring DB connectivity")

# Derived / aggregated signals (from DB or Prometheus)
SLO_ERROR_RATE = Gauge(
    "slo_error_rate_5m",
    "Error rate over 5m (best-effort, derived)",
    ["service"],
)
SLO_P95_LATENCY_S = Gauge(
    "slo_p95_latency_seconds_5m",
    "p95 latency over 5m in seconds (best-effort, derived)",
    ["service"],
)
SLO_REQUEST_RATE = Gauge(
    "slo_request_rate_rps_5m",
    "Requests/sec over 5m (best-effort, derived)",
    ["service"],
)

# ML behavior metrics
LOW_CONFIDENCE_RATE = Gauge(
    "prediction_low_confidence_rate_5m",
    "Share of low-confidence predictions over recent window",
    ["tenant_id"],
)
PREDICTION_ENTROPY_AVG = Gauge(
    "prediction_entropy_avg_5m",
    "Average prediction entropy over recent window",
    ["tenant_id"],
)

# Cost
COST_PER_REQUEST = Gauge(
    "llm_cost_per_request_dollars_5m",
    "Approx. cost per request over 5m",
    ["model"],
)

# Fairness / disparity proxy (tenant spread)
TENANT_FEEDBACK_SPREAD = Gauge(
    "tenant_feedback_spread_24h",
    "(max-min) of average feedback across tenants over 24h",
)

# Data quality
INPUT_MISSING_RATE = Gauge(
    "input_missing_rate_24h",
    "Average missing-rate across monitored inputs over 24h",
    ["tenant_id"],
)
OUT_OF_RANGE_RATE = Gauge(
    "input_out_of_range_rate_24h",
    "Average out-of-range rate across monitored inputs over 24h",
    ["tenant_id"],
)
DATA_FRESHNESS_P95_S = Gauge(
    "data_freshness_p95_seconds_24h",
    "p95 data freshness (seconds) over 24h",
    ["tenant_id"],
)
SCHEMA_HASH_DISTINCT = Gauge(
    "schema_hash_distinct_24h",
    "Number of distinct schema hashes observed over 24h",
    ["tenant_id"],
)

# Service/prediction behavior
CACHE_HIT_RATE = Gauge(
    "cache_hit_rate_24h",
    "Cache hit-rate over 24h (from events)",
    ["tenant_id"],
)
FALLBACK_RATE = Gauge(
    "llm_fallback_rate_24h",
    "Share of requests that used a fallback path (from events)",
    ["tenant_id"],
)

# RAG/task-specific quality
RAG_HAS_CITATIONS_RATE_24H = Gauge(
    "rag_has_citations_rate_24h",
    "Share of RAG answers that contain citations (best-effort)",
    ["tenant_id"],
)
RAG_CITATION_COVERAGE_AVG_24H = Gauge(
    "rag_citation_coverage_avg_24h",
    "Average sentence-level citation coverage over 24h (best-effort)",
    ["tenant_id"],
)
RAG_CITATIONS_PER_ANSWER_AVG_24H = Gauge(
    "rag_citations_per_answer_avg_24h",
    "Average number of citations per answer over 24h (best-effort)",
    ["tenant_id"],
)
RAG_GROUNDEDNESS_AVG_24H = Gauge(
    "rag_groundedness_score_avg_24h",
    "Average groundedness score over 24h (if provided by producer)",
    ["tenant_id"],
)
RAG_UNGROUNDED_RATE_24H = Gauge(
    "rag_ungrounded_rate_24h",
    "Share of answers below groundedness threshold over 24h (best-effort)",
    ["tenant_id"],
)

# Drift / performance from events
DATA_DRIFT_PSI = Gauge(
    "data_drift_psi",
    "PSI drift score",
    ["feature", "tenant_id"],
)
EMBEDDING_CENTROID_DISTANCE = Gauge(
    "embedding_centroid_distance",
    "Cosine distance between embedding centroids",
    ["tenant_id"],
)
FEEDBACK_SCORE_AVG = Gauge(
    "model_user_feedback_score_avg_24h",
    "Average user feedback score over 24h",
    ["tenant_id"],
)
CONCEPT_DRIFT_DETECTED = Counter(
    "concept_drift_detected_total",
    "Concept drift detections",
    ["tenant_id"],
)

DB_ERROR_RATE_24H = Gauge(
    "inference_error_rate_24h",
    "Error rate over 24h (from events)",
    ["tenant_id"],
)
DB_P95_LATENCY_MS_24H = Gauge(
    "inference_p95_latency_ms_24h",
    "p95 latency (ms) over 24h (from events)",
    ["tenant_id"],
)

# Monitoring loop duration
MONITOR_LOOP_DURATION = Histogram(
    "monitoring_loop_duration_seconds",
    "Duration of monitoring background loops",
    ["loop"],
)
