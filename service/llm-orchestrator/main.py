from typing import Optional
import httpx
import redis.asyncio as aioredis
import mlflow
from typing import Dict, List, Any
import re

from prompt_manager import PromptManager
prompt_manager = PromptManager()

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


def build_rag_prompt(query: str, context: List[Dict[str, Any]], doc_type: str) -> str:
    template = prompt_manager.get_prompt_template(doc_type)

    # Format context chunks
    context_parts = []
    for i, ctx in enumerate(context[:10], 1):  # Top 10 chunks
        context_parts.append(
            f"[{i}] (Page {ctx.get('page', 'N/A')}, {ctx.get('type', 'text')})\n{ctx.get('text', '')}"
        )

    context_str = '\n\n'.join(context_parts)

    prompt = template.format(
        context=context_str,
        query=query
    )

    return prompt


def calculate_confidence(answer: str, citations: List[Dict]) -> float:
    # Calculate confidence score based on
    # 1. Presence of citations
    # 2. Answer length (not too short/long)
    # 3. Absence of hedging language

    score = 0.5  # Base score

    # Citations boost confidence
    if citations:
        score += min(0.3, len(citations) * 0.1)

    # Reasonable length using token approximation
    tokens = int(len(answer) / 4)
    if 50 < tokens < 500:
        score += 0.1

    # Hedging language reduces confidence
    hedging_words = ['maybe', 'perhaps', 'might', 'could be', 'unclear', 'not sure']
    hedge_count = sum(1 for word in hedging_words if word in answer.lower())
    score -= hedge_count * 0.05

    return max(0.0, min(1.0, score))


def extract_citations(answer: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Simple citation extraction based on [N] markers
    citation_pattern = r'\[(\d+)\]'
    cited_indices = set(int(m) for m in re.findall(citation_pattern, answer))

    citations = []
    for idx in cited_indices:
        if 0 < idx <= len(context):
            ctx = context[idx - 1]
            citations.append({
                'index': idx,
                'page': ctx.get('page'),
                'text': ctx.get('text', '')[:200],
                'doc_id': ctx.get('doc_id'),
                'filename': ctx.get('filename')
            })

    return citations


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
