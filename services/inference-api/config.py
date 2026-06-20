"""
Config file.
"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"

    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index: str = "doc-intelligence"
    pinecone_timeout: int = 10

    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    llm_orchestrator_url: str = "http://llm-orchestrator:8002"
    llm_timeout: int = 30
    llm_retry_attempts: int = 3

    ingestion_service_url: str = "http://ingestion:8001"
    documents_authority_url: str = ""

    rate_limit_per_minute: int = 60

    max_top_k: int = 50
    max_query_length: int = 2000
    min_retrieval_score: float = 0.7
    max_concurrent_requests: int = 100
    max_context_chars: int = 15000
    request_timeout: int = 30

    cache_ttl: int = 3600
    cache_ttl_high_confidence: int = 7200
    cache_ttl_low_confidence: int = 1800

    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 30

    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost:8080"]

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "documents"
    minio_secure: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"
