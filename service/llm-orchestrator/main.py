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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
