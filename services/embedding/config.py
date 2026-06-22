from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"

    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index: str = "doc-intelligence"

    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"

    chunk_size: int = 512
    chunk_overlap: int = 50

    batch_size: int = 32

    class Config:
        env_file = ".env"
        extra = "ignore"
