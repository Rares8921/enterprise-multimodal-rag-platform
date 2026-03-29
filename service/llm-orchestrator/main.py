from typing import Optional
import httpx
import redis.asyncio as aioredis
from complexity_analyzer import QueryComplexityAnalyzer

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
