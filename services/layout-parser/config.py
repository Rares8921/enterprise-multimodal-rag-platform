from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Redis
    redis_url: str = "redis://localhost:6379"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"

    # Model
    model_name: str = "microsoft/layoutlmv3-base"
    model_version: str = "1.0.0"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Processing
    batch_size: int = 1
    max_seq_length: int = 512

    class Config:
        env_file = ".env"
        extra = "ignore"