from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Core deps
    redis_url: str = "redis://localhost:6379"
    postgres_url: str = "postgresql://admin:changeme@localhost:5432/doc_intel"

    # Integrations
    mlflow_tracking_uri: str = "http://localhost:5000"
    prometheus_url: str = "http://localhost:9090"

    # Monitoring service
    monitoring_port: int = 9104

    # Analytics from DB events
    analytics_interval_seconds: int = 60
    low_confidence_threshold: float = 0.4

    # Drift detection (DB-based)
    enable_drift_detection: bool = True
    drift_check_interval: int = 3600  # 1 hour (kept for backwards compatibility)
    data_drift_threshold: float = 0.2  # PSI threshold
    concept_drift_threshold: float = 0.05  # User feedback score drop
    embedding_drift_threshold: float = 0.15  # Centroid cosine distance limit

    # RAG/task-specific thresholds (best-effort)
    rag_groundedness_threshold: float = 0.6
    rag_citation_coverage_threshold: float = 0.3
    rag_min_samples_for_alert: int = 20
    alert_on_rag: bool = False

    # Prometheus polling (SLO/SLI + cost)
    prometheus_timeout_seconds: float = 5.0
    prometheus_poll_interval_seconds: int = 30

    # Alerting
    slack_webhook_url: str = ""
    alert_on_drift: bool = True
    alert_on_slo: bool = True

    # SLO thresholds (defaults)
    error_rate_alert_threshold: float = 0.05
    p95_latency_seconds_alert_threshold: float = 0.4

    class Config:
        env_file = ".env"