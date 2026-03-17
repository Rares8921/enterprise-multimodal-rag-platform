"""
Config file.
"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"

    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index: str = "doc-intelligence"

    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    llm_orchestrator_url: str = "http://llm-orchestrator:8002"

    cache_ttl: int = 3600

    rate_limit_per_minute: int = 60

    class Config:
        env_file = ".env"