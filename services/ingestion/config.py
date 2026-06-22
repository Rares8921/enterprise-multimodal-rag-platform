from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis
    redis_url: str = "redis://localhost:6379"

    postgres_url: str = "postgresql+asyncpg://admin:changeme@localhost:5432/doc_intel"

    # Process topology
    enable_workers: bool = True

    # MinIO/S3
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"

    ocr_engine: str = "easyocr"
    ocr_languages: str = "eng"
    ocr_dpi: int = 300
    prefer_text_pdf_extraction: bool = True
    ocr_gpu: bool = False

    # Processing
    max_file_size_mb: int = 100
    worker_concurrency: int = 4

    class Config:
        env_file = ".env"
        extra = "ignore"