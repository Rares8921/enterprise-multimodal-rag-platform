from typing import Optional
import httpx
import redis.asyncio as aioredis
import time
from typing import Dict, List, Any
import re
import hashlib, json

from prompt_manager import PromptManager
prompt_manager = PromptManager()

from complexity_analyzer import QueryComplexityAnalyzer

from model_wrapper import GeminiLLM, MistralLLM
from utils import ModelRouter, QueryResponse, QueryRequest, ModelChoice

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from prometheus_client import Counter, Histogram, make_asgi_app
# Metrics
LLM_REQUESTS = Counter('llm_requests_total', 'Total LLM requests', ['model', 'doc_type'])
LLM_DURATION = Histogram('llm_inference_seconds', 'LLM inference duration', ['model'])
LLM_TOKENS = Counter('llm_tokens_total', 'Total tokens processed', ['model', 'type'])
LLM_COST = Counter('llm_cost_dollars', 'Estimated LLM cost in dollars', ['model'])

from fastapi import FastAPI, HTTPException
app = FastAPI(title="LLM Orchestrator", version="1.0.0")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

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
    gemini_model = GeminiLLM(settings.gemini_api_key, settings.gemini_model_name)

    mistral_client = httpx.AsyncClient()
    mistral_model = MistralLLM(settings.mistral_api_url, mistral_client)

    # MLflow is only needed at service startup, so import it lazily to keep
    # lightweight unit tests from paying the import cost.
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    except Exception as e:
        logger.warning(f"MLflow tracking setup skipped: {str(e)}")

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


def build_rag_prompt(query: str, context: List[Dict[str, Any]], doc_type: str, agent: str = "default") -> str:
    template = prompt_manager.get_prompt_template(doc_type, agent=agent)

    # Format context chunks (Top 10)
    context_parts = []
    for i, ctx in enumerate(context[:10], 1):
        source = ctx.get('filename') or ctx.get('doc_id') or 'N/A'
        context_parts.append(
            f"[{i}] ({source}, Page {ctx.get('page', 'N/A')}, {ctx.get('type', 'text')})\n{ctx.get('text', '')}"
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


def _safe_token_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def normalize_generation_result(result: Dict[str, Any]) -> tuple[str, Dict[str, int]]:
    if not isinstance(result, dict):
        raise ValueError("Model response must be a dictionary")

    answer = result.get('text')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Model response missing non-empty text")

    raw_usage = result.get('usage')
    if raw_usage is None:
        raw_usage = {}
    if not isinstance(raw_usage, dict):
        raise ValueError("Model response usage must be a dictionary")

    input_tokens = _safe_token_count(raw_usage.get('input_tokens'))
    output_tokens = _safe_token_count(raw_usage.get('output_tokens'))
    total_tokens = _safe_token_count(raw_usage.get('total_tokens'))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return answer, {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens
    }


@app.post("/generate", response_model=QueryResponse)
async def generate_response(request: QueryRequest):
    if not all([redis_client, model_router, gemini_model, mistral_model]):
        logger.error("Global components not initialized")
        raise HTTPException(status_code=503, detail="Service unavailable: components not initialized")

    start_time = time.time()

    try:
        # Optimize context hashing & token estimation
        context_dump = json.dumps(request.context, sort_keys=True)
        context_tokens = int(len(context_dump) / 4)

        # Model routing
        selected_model = model_router.select_model(
            request.query,
            context_tokens,
            request.doc_type,
            force_model=request.model_choice.value if request.model_choice != ModelChoice.AUTO else None
        )

        # Check cache with robust key
        hash_input = f"{request.tenant_id}|{request.query}|{context_dump}|{request.doc_type}|{request.agent}|{selected_model}|{request.temperature}|{request.max_tokens}".encode(
            'utf-8')
        stable_hash = hashlib.sha256(hash_input).hexdigest()
        cache_key = f"llm_cache:{request.tenant_id}:{request.doc_type}:{request.agent}:{selected_model}:{stable_hash}"

        cached_data = await redis_client.get(cache_key)
        if cached_data:
            logger.info("Cache HIT. Returning cached response.")
            cached_response = json.loads(cached_data)
            return QueryResponse(**cached_response)

        context_for_prompt = request.context[:10]

        # Build prompt
        prompt = build_rag_prompt(
            query=request.query,
            context=context_for_prompt,
            doc_type=request.doc_type,
            agent=request.agent
        )

        try:
            with LLM_DURATION.labels(model=selected_model).time():
                if selected_model == 'gemini':
                    result = await gemini_model.generate(
                        prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    )
                else:
                    result = await mistral_model.generate(
                        prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    )
        except Exception as e:
            if selected_model == 'gemini':
                logger.warning(f"Fallback activated. Gemini error: {str(e)}. Falling back to Mistral.")
                selected_model = 'mistral'

                # Update cache key for fallback model
                hash_input = f"{request.tenant_id}|{request.query}|{context_dump}|{request.doc_type}|{request.agent}|{selected_model}|{request.temperature}|{request.max_tokens}".encode(
                    'utf-8')
                stable_hash = hashlib.sha256(hash_input).hexdigest()
                cache_key = f"llm_cache:{request.tenant_id}:{request.doc_type}:{request.agent}:{selected_model}:{stable_hash}"

                try:
                    with LLM_DURATION.labels(model=selected_model).time():
                        result = await mistral_model.generate(
                            prompt,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens
                        )
                except Exception as fallback_e:
                    logger.error(f"Mistral fallback also failed: {str(fallback_e)}")
                    raise HTTPException(status_code=500, detail="All models failed. Fallback exhausted.")
            else:
                logger.error(f"Mistral generation failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Model generation failed: {str(e)}")

        logger.info(f"Successfully generated response using model: {selected_model}")

        try:
            answer, usage = normalize_generation_result(result)
        except ValueError as e:
            logger.error(f"Malformed model response from {selected_model}: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Malformed model response: {str(e)}")

        # Extract citations from answer (align indices with prompt context)
        citations = extract_citations(answer, context_for_prompt)

        confidence = calculate_confidence(answer, citations)

        LLM_REQUESTS.labels(model=selected_model, doc_type=request.doc_type).inc()
        LLM_TOKENS.labels(model=selected_model, type='input').inc(usage['input_tokens'])
        LLM_TOKENS.labels(model=selected_model, type='output').inc(usage['output_tokens'])

        # Track cost
        cost = model_router.estimate_cost(selected_model, usage['input_tokens'], usage['output_tokens'])
        LLM_COST.labels(model=selected_model).inc(cost)

        # Latency calc & return ans
        latency_ms = (time.time() - start_time) * 1000

        response_obj = QueryResponse(
            answer=answer,
            model_used=selected_model,
            citations=citations,
            confidence_score=confidence,
            tokens_used=usage['total_tokens'],
            latency_ms=latency_ms
        )

        # Store in cache
        await redis_client.setex(
            cache_key,
            settings.cache_ttl,
            response_obj.model_dump_json() if hasattr(response_obj, 'model_dump_json') else response_obj.json()
        )

        return response_obj

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
