from typing import Optional
import httpx
import redis.asyncio as aioredis
import mlflow

from complexity_analyzer import QueryComplexityAnalyzer
from model_wrapper import GeminiLLM, MistralLLM
from utils import ModelRouter

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from prometheus_client import Counter, Histogram
# Metrics
LLM_REQUESTS = Counter('llm_requests_total', 'Total LLM requests', ['model', 'doc_type'])
LLM_DURATION = Histogram('llm_inference_seconds', 'LLM inference duration', ['model'])
LLM_TOKENS = Counter('llm_tokens_total', 'Total tokens processed', ['model', 'type'])
LLM_COST = Counter('llm_cost_dollars', 'Estimated LLM cost in dollars', ['model'])

from fastapi import FastAPI
app = FastAPI(title="LLM Orchestrator", version="1.0.0")

from config import Settings
settings = Settings()

# Global state
redis_client: Optional[aioredis.Redis] = None
complexity_analyzer: Optional[QueryComplexityAnalyzer] = None
gemini_model = None
mistral_client: Optional[httpx.AsyncClient] = None
mistral_model = None
model_router = None


@app.on_event("startup")
async def startup():
    global redis_client, complexity_analyzer, gemini_model, mistral_client, mistral_model, model_router

    # Redis
    redis_client = await aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True
    )

    # Complexity analyzer
    complexity_analyzer = QueryComplexityAnalyzer()

    # Initialize Router
    model_router = ModelRouter(settings, complexity_analyzer)

    # Initialize LLMs
    gemini_model = GeminiLLM(settings.gemini_api_key)

    mistral_client = httpx.AsyncClient()
    mistral_model = MistralLLM(settings.mistral_api_url, mistral_client)

    # MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    logger.info("LLM Orchestrator started")


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()
    if mistral_client:
        await mistral_client.aclose()

@app.get("/health/live")
async def health_live():
    """Liveness probe"""
    return {"status": "alive"}


@app.get("/health")
@app.get("/health/ready")
async def health_ready():
    deps = {
        "redis": "down",
        "gemini": "down",
        "mistral": "down"
    }
    status = "healthy"

    # Check Redis
    if redis_client:
        try:
            await redis_client.ping()
            deps["redis"] = "ok"
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            status = "degraded"
    else:
        logger.error("Redis client is None")
        status = "degraded"

    # Check Gemini
    if gemini_model and gemini_model.model:
        deps["gemini"] = "ok"
    else:
        logger.error("Gemini model is not initialized")
        status = "degraded"

    # Check Mistral
    if mistral_model and mistral_model.client:
        deps["mistral"] = "ok"
    else:
        logger.error("Mistral model client is not initialized")
        status = "degraded"

    return {
        "status": status,
        "dependencies": deps
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
