from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"

    postgres_url: str = "postgresql://admin:changeme@localhost:5432/doc_intel"

    mlflow_tracking_uri: str = "http://localhost:5000"

    prometheus_url: str = "http://localhost:9090"

    drift_check_interval: int = 3600  # 1 hour
    data_drift_threshold: float = 0.2  # PSI threshold
    concept_drift_threshold: float = 0.05  # User feedback score drop
    embedding_drift_threshold: float = 0.15  # Centroid cosine distance limit

    # Alerting
    slack_webhook_url: str = ""
    alert_on_drift: bool = True

    class Config:
        env_file = ".env"