from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"

    gemini_api_key: str
    gemini_model_name: str = "gemini-2.5-flash"
    mistral_api_url: str = "http://mistral:8000"

    mlflow_tracking_uri: str = "http://localhost:5000"

    # model routing
    complexity_threshold: float = 0.7

    cache_ttl: int = 3600 # 1 hour

    class Config:
        env_file = ".env"
        extra = "ignore"
