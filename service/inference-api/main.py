import logging, random, asyncio, time, json, uuid, hashlib, re
from typing import Counter, Dict

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import redis.asyncio as aioredis
from pinecone import Pinecone
from minio import Minio
from sentence_transformers import SentenceTransformer

from config import Settings
from auth import verify_tenant
from QueryModel import QueryRequest, QueryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document API", version="1.0.0")
settings = Settings()

CACHE_HITS = Counter('cache_hits_total', 'Cache hits', ['tenant_id'])
QUERY_FAILURES = Counter('query_failures_total', 'Failed queries', ['error_type', 'tenant_id'])
RATE_LIMIT_EXCEEDED = Counter('rate_limit_exceeded_total', 'Rate limit hits', ['tenant_id'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

redis_client: Optional[aioredis.Redis] = None
redis_pubsub_client: Optional[aioredis.Redis] = None
pinecone_client: Optional[Pinecone] = None
pinecone_index = None

minio_client: Optional[Minio] = None

request_semaphore: Optional[asyncio.Semaphore] = None
embedding_semaphore: Optional[asyncio.Semaphore] = None

embedding_model: Optional[SentenceTransformer] = None
embedding_model_cpu: Optional[SentenceTransformer] = None
embedding_device: str = "cpu"
embedding_semaphore: Optional[asyncio.Semaphore] = None

tenant_semaphores: Dict[str, asyncio.Semaphore] = {}
tenant_request_counts: Dict[str, int] = {}


def sanitize_context(text: str) -> str:
    # TODO: add more obviously
    dangerous_patterns = [
        r"ign[o0]re\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+previous",
        r"forget\s+everything",
        r"new\s+instructions?:",
        r"system\s*:",
        r"assistant\s*:",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"<\|system\|>",
        r"<\|user\|>",
        r"<\|assistant\|>",
        r"promptize",
        r"jailbreak",
        r"you\s+are\s+now",
        r"act\s+as\s+if",
        r"pretend\s+to\s+be",
        r"simulate\s+that",
        r"override\s+your",
        r"bypass\s+your"
    ]

    sanitized = text
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)

    # Remove control characters and excessive whitespace
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\t')
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    # Truncate extremely long single chunks
    if len(sanitized) > 8000:
        sanitized = sanitized[:8000] + "...[truncated]"

    return sanitized


def get_tenant_semaphore(tenant_id: str, max_concurrent_per_tenant: int = 5) -> asyncio.Semaphore:
    """Get or create semaphore for a tenant"""
    if tenant_id not in tenant_semaphores:
        tenant_semaphores[tenant_id] = asyncio.Semaphore(max_concurrent_per_tenant)
        tenant_request_counts[tenant_id] = 0
    return tenant_semaphores[tenant_id]


async def acquire_with_fairness(tenant_id: str, global_sem: asyncio.Semaphore, tenant_sem: asyncio.Semaphore, request_id: str):
    """global first, then tenant-specific"""
    # Acquire global semaphore first (to limit total concurrency)
    await global_sem.acquire()

    try:
        # Then acquire tenant-specific semaphore
        try:
            await asyncio.wait_for(tenant_sem.acquire(), timeout=30.0)
            tenant_request_counts[tenant_id] = tenant_request_counts.get(tenant_id, 0) + 1
            return True
        except asyncio.TimeoutError:
            logger.warning(f"[{request_id}] Tenant {tenant_id} semaphore timeout")
            global_sem.release()
            raise HTTPException(status_code=503, detail="Service busy, please retry later")
    except:
        global_sem.release()
        raise


def release_with_fairness(tenant_id: str, global_sem: asyncio.Semaphore, tenant_sem: asyncio.Semaphore):
    """Release both semaphores"""
    tenant_request_counts[tenant_id] = max(0, tenant_request_counts.get(tenant_id, 1) - 1)
    tenant_sem.release()
    global_sem.release()


async def check_rate_limit(tenant_id: str) -> None:
    # Use hash tags for Redis Cluster to ensure all tenant keys on same shard
    key = f"{{tenant_id:{tenant_id}}}:rate:window"
    now = time.time()
    window_start = now - 60  # 60 seconds window

    lua_script = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window_start = tonumber(ARGV[2])
    local max_requests = tonumber(ARGV[3])
    local request_id = ARGV[4]

    -- Remove old entries outside the window
    redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

    -- Count current requests in window
    local current_count = redis.call('ZCARD', key)

    if current_count >= max_requests then
        return -1
    end

    -- Add current request
    redis.call('ZADD', key, now, request_id)
    redis.call('EXPIRE', key, 120) -- Keep key for 2 minutes

    return current_count + 1
    """

    request_id = str(uuid.uuid4())
    count = await redis_client.eval(
        lua_script,
        1,
        key,
        str(now),
        str(window_start),
        str(settings.rate_limit_per_minute),
        request_id
    )

    if count == -1:
        RATE_LIMIT_EXCEEDED.labels(tenant_id=tenant_id).inc()
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {settings.rate_limit_per_minute} requests per minute."
        )



async def check_circuit_breaker() -> bool:
    cb_key = "circuit_breaker:llm"
    is_open = await redis_client.get(cb_key)
    return is_open == "1"


def build_cache_key(tenant_id: str, request: QueryRequest) -> str:
    cache_params = {
        'q': request.query,
        'doc_type': request.doc_type,
        'doc_id': request.doc_id,
        'model': request.model_choice,
        'top_k': request.top_k,
        'embedding_model': settings.embedding_model
    }

    def _stable_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    param_hash = _stable_hash(json.dumps(cache_params, sort_keys=True))
    return f"query_cache:{tenant_id}:{param_hash}"


async def encode_with_fallback(text: str, request_id: str, timeout: float = 10.0):
    """Encode text with GPU, fallback to CPU on failure"""
    loop = asyncio.get_event_loop()

    # Try GPU first if available
    if embedding_device == "cuda":
        try:
            embedding = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: embedding_model.encode(
                        text,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                ),
                timeout=timeout
            )
            return embedding
        except asyncio.TimeoutError:
            logger.warning(f"[{request_id}] GPU embedding timeout, falling back to CPU")
        except Exception as e:
            logger.warning(f"[{request_id}] GPU embedding failed: {e}, falling back to CPU")

    # CPU fallback
    try:
        embedding = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: embedding_model_cpu.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            ),
            timeout=timeout * 2  # Give CPU more time
        )
        return embedding
    except Exception as e:
        logger.error(f"[{request_id}] CPU embedding also failed: {e}")
        raise HTTPException(status_code=503, detail="Embedding service unavailable")


async def retry_delete_document(doc_id: str, tenant_id: str, object_name: Optional[str] = None, max_retries: int = 3) -> bool:
    """Retry deleting document from Pinecone and MinIO with exponential backoff"""

    for attempt in range(max_retries):
        try:
            # Try Pinecone delete
            pinecone_success = False
            try:
                pinecone_index.delete(
                    filter={'doc_id': doc_id},
                    namespace=tenant_id
                )
                pinecone_success = True
                logger.info(f"Deleted from Pinecone: {doc_id} (attempt {attempt + 1})")
            except Exception as pinecone_error:
                logger.warning(f"Pinecone deletion failed (attempt {attempt + 1}): {pinecone_error}")

            # Try MinIO delete
            minio_success = False
            if object_name:
                try:
                    minio_client.remove_object(settings.minio_bucket, object_name)
                    minio_success = True
                    logger.info(f"Deleted from MinIO: {object_name} (attempt {attempt + 1})")
                except Exception as minio_error:
                    logger.warning(f"MinIO deletion failed (attempt {attempt + 1}): {minio_error}")
            else:
                minio_success = True  # No object to delete

            # Both succeeded
            if pinecone_success and minio_success:
                return True

            # Wait before retry
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)

        except Exception as e:
            logger.error(f"Retry delete failed for {doc_id} (attempt {attempt + 1}): {e}")

    return False


async def add_to_delete_dlq(doc_id: str, doc_data: dict, failure_reason: str):
    dlq_key = "delete_dlq"
    task_data = {
        'doc_id': doc_id,
        'tenant_id': doc_data.get('tenant_id'),
        'object_name': doc_data.get('object_name'),
        'timestamp': time.time(),
        'failure_reason': failure_reason,
        'retry_count': 0
    }

    await redis_client.lpush(dlq_key, json.dumps(task_data))
    logger.warning(f"Added document {doc_id} to delete DLQ: {failure_reason}")


def query_pinecone(vector_np, top_k: int, namespace: str, filter_dict: dict) -> dict:
    try:
        vector = vector_np.tolist() if hasattr(vector_np, 'tolist') else vector_np
        index = pinecone_client.Index(settings.pinecone_index)
        return index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
    except Exception as e:
        logger.error(f"Pinecone query failed: {e}")
        return {'matches': []}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, x_tenant_id: str = Header(...)):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    tenant_id = None

    async def _process():
        nonlocal tenant_id

        logger.info(f"[{request_id}] Query from tenant {x_tenant_id}: {request.query[:50]}")

        # Verify tenant before acquiring semaphores
        tenant_id = await verify_tenant(x_tenant_id=x_tenant_id)

        if tenant_id != request.tenant_id:
            raise HTTPException(status_code=403, detail="Tenant ID mismatch")

        # Get tenant-specific semaphore
        tenant_sem = get_tenant_semaphore(tenant_id, max_concurrent_per_tenant=5)

        # Acquire with fairness (global + tenant-specific)
        await acquire_with_fairness(tenant_id, request_semaphore, tenant_sem, request_id)

        try:
            await check_rate_limit(tenant_id)

            cache_key = build_cache_key(tenant_id, request)
            cached_response = await redis_client.get(cache_key)

            if cached_response:
                logger.info(f"[{request_id}] Cache hit")
                CACHE_HITS.labels(tenant_id=tenant_id).inc()
                response_data = json.loads(cached_response)
                response_data['latency_ms'] = (time.time() - start_time) * 1000
                response_data['metadata']['cache_hit'] = True
                response_data['metadata']['request_id'] = request_id
                return QueryResponse(**response_data)

            async with embedding_semaphore:
                query_embedding = await encode_with_fallback(
                    request.query,
                    request_id
                )

            filter_dict = {}
            if request.doc_type:
                filter_dict['doc_type'] = request.doc_type
            if request.doc_id:
                filter_dict['doc_id'] = request.doc_id

            context = []
            retrieval_time = 0.0
            retrieval_start = time.time()

            try:
                loop = asyncio.get_event_loop()
                retrieval_task = loop.run_in_executor(
                    None,
                    lambda: query_pinecone(
                        query_embedding,
                        request.top_k,
                        tenant_id,
                        filter_dict
                    )
                )

                retrieval_results = await asyncio.wait_for(
                    retrieval_task,
                    timeout=settings.pinecone_timeout
                )

                for match in retrieval_results.get('matches', []):
                    if match.get('score', 0) < settings.min_retrieval_score:
                        continue

                    metadata = match.get('metadata') or {}
                    raw_text = metadata.get('text', '')
                    sanitized_text = sanitize_context(raw_text)

                    context.append({
                        'text': sanitized_text,
                        'page': metadata.get('page', 1),
                        'type': metadata.get('type', 'text'),
                        'doc_id': metadata.get('doc_id', ''),
                        'filename': metadata.get('filename', ''),
                        'score': match.get('score', 0.0)
                    })

                context = limit_context_size(context)

                retrieval_time = time.time() - retrieval_start
                RETRIEVAL_LATENCY.labels(tenant_id=tenant_id).observe(retrieval_time)

                total_context_chars = sum(len(c['text']) for c in context)
                CONTEXT_SIZE.labels(tenant_id=tenant_id).observe(total_context_chars)

                logger.info(
                    f"[{request_id}] Retrieved {len(context)} chunks, {total_context_chars} chars in {retrieval_time:.2f}s")

            except Exception as e:
                retrieval_time = time.time() - retrieval_start
                logger.warning(
                    f"[{request_id}] Pinecone query failed after {retrieval_time:.2f}s, continuing without retrieval: {e}")
                context = []

            if not context:
                logger.info(f"[{request_id}] No context available, proceeding with LLM only")

            llm_start = time.time()
            llm_data = await call_llm_with_retry(
                request.query,
                context,
                request.doc_type or 'generic',
                tenant_id,
                request.model_choice,
                request_id
            )
            llm_time = time.time() - llm_start
            LLM_LATENCY.labels(tenant_id=tenant_id).observe(llm_time)

            answer = llm_data['answer']
            if len(answer) > 5000:
                logger.warning(f"[{request_id}] LLM response too long ({len(answer)} chars), truncating")
                answer = answer[:5000] + "..."

            latency_ms = (time.time() - start_time) * 1000

            response_data = {
                'answer': answer,
                'citations': llm_data.get('citations', []) if request.include_citations else [],
                'model_used': llm_data['model_used'],
                'confidence_score': llm_data['confidence_score'],
                'latency_ms': latency_ms,
                'metadata': {
                    'retrieval_count': len(context),
                    'tokens_used': llm_data.get('tokens_used', 0),
                    'cache_hit': False,
                    'request_id': request_id
                }
            }

            should_cache = (
                llm_data['confidence_score'] > 0.5 and
                len(context) > 0
            )

            if should_cache:
                ttl = get_adaptive_cache_ttl(llm_data['confidence_score'])
                await redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(response_data)
                )

            QUERY_REQUESTS.labels(
                tenant_id=tenant_id,
                doc_type=request.doc_type or 'all'
            ).inc()
            QUERY_SUCCESS.labels(tenant_id=tenant_id).inc()
            QUERY_DURATION.labels(tenant_id=tenant_id).observe(latency_ms / 1000)

            logger.info(
                f"[{request_id}] Query processed: {latency_ms:.0f}ms, context={len(context)}, llm={llm_time:.2f}s, retrieval={retrieval_time:.2f}s, cached={should_cache}")

            return QueryResponse(**response_data)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Query processing error: {str(e)}")
            QUERY_FAILURES.labels(
                error_type=type(e).__name__,
                tenant_id=tenant_id or request.tenant_id
            ).inc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Always release semaphores
            release_with_fairness(tenant_id, request_semaphore, tenant_sem)

    try:
        return await asyncio.wait_for(_process(), timeout=settings.request_timeout)
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Request timeout after {settings.request_timeout}s")
        QUERY_FAILURES.labels(
            error_type='TimeoutError',
            tenant_id=tenant_id or request.tenant_id
        ).inc()
        raise HTTPException(status_code=504, detail="Request timeout")

@app.get("/documents/{tenant_id}")
async def list_documents(tenant_id: str, x_tenant_id: str = Header(...)):
    verified_tenant = await verify_tenant(x_tenant_id=x_tenant_id)

    if verified_tenant != tenant_id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    try:
        doc_ids = await redis_client.smembers(f"tenant_docs:{tenant_id}")

        documents = []
        for doc_id in doc_ids:
            doc_data = await redis_client.hgetall(f"doc:{doc_id}")
            if doc_data:
                documents.append({
                    'doc_id': doc_data.get('doc_id'),
                    'filename': doc_data.get('filename'),
                    'doc_type': doc_data.get('doc_type'),
                    'status': doc_data.get('status'),
                    'uploaded_at': doc_data.get('uploaded_at')
                })

        return {'documents': documents, 'total': len(documents)}

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, x_tenant_id: str = Header(...)):
    tenant_id = await verify_tenant(x_tenant_id=x_tenant_id)

    try:
        doc_data = await redis_client.hgetall(f"doc:{doc_id}")
        if not doc_data:
            raise HTTPException(status_code=404, detail="Document not found")

        if doc_data.get('tenant_id') != tenant_id:
            raise HTTPException(status_code=403, detail="Unauthorized")

        # Mark as pending_delete
        await redis_client.hset(f"doc:{doc_id}", "status", "pending_delete")

        # Try immediate delete
        delete_success = await retry_delete_document(
            doc_id,
            tenant_id,
            doc_data.get('object_name'),
            max_retries=2
        )

        if delete_success:
            # Clean up Redis metadata immediately
            await redis_client.delete(f"doc:{doc_id}")
            await redis_client.delete(f"doc:{doc_id}:ocr")
            await redis_client.delete(f"doc:{doc_id}:layout")
            await redis_client.delete(f"doc:{doc_id}:embeddings")
            await redis_client.srem(f"tenant_docs:{tenant_id}", doc_id)

            logger.info(f"Document deleted successfully: {doc_id}")
            return {'status': 'deleted', 'doc_id': doc_id}
        else:
            # Add to DLQ for later retry
            await add_to_delete_dlq(doc_id, doc_data, "Initial delete attempt failed")
            logger.warning(f"Document marked for async deletion: {doc_id}")
            return {
                'status': 'pending_deletion',
                'doc_id': doc_id,
                'message': 'Document deletion in progress, in the mean time you can access the app.'
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check(deep: bool = False):
    circuit_breaker_open = await check_circuit_breaker()

    checks = {
        'api': 'healthy',
        'redis': 'unknown',
        'pinecone': 'unknown',
        'embedding_model': 'loaded' if embedding_model else 'not_loaded',
        'circuit_breaker': 'open' if circuit_breaker_open else 'closed'
    }

    try:
        await redis_client.ping()
        await redis_client.set('health_check', '1', ex=10)
        checks['redis'] = 'connected'
    except Exception as e:
        checks['redis'] = f'disconnected: {str(e)}'

    try:
        stats = pinecone_index.describe_index_stats()
        checks['pinecone'] = 'connected'
        checks['vector_count'] = stats.get('total_vector_count', 0)
        checks['namespaces'] = len(stats.get('namespaces', {}))
    except Exception as e:
        checks['pinecone'] = f'disconnected: {str(e)}'

    if deep and embedding_model:
        try:
            test_text = "health check test query"
            async with embedding_semaphore:
                test_embedding = await encode_with_fallback(
                    test_text,
                    "health_check",
                    timeout=5.0
                )
            checks['embedding_test'] = 'passed' if test_embedding is not None else 'failed'
        except Exception as e:
            checks['embedding_test'] = f'failed: {str(e)}'

    overall_healthy = (
        checks['redis'] == 'connected' and
        checks['pinecone'] == 'connected' and
        checks['embedding_model'] == 'loaded' and
        not circuit_breaker_open
    )

    checks['status'] = 'healthy' if overall_healthy else 'degraded'
    status_code = 200 if overall_healthy else 503

    return JSONResponse(content=checks, status_code=status_code)


@app.get("/stats/{tenant_id}")
async def get_stats(tenant_id: str, x_tenant_id: str = Header(...)):
    verified_tenant = await verify_tenant(x_tenant_id=x_tenant_id)

    if verified_tenant != tenant_id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    try:
        doc_ids = await redis_client.smembers(f"tenant_docs:{tenant_id}")
        doc_count = len(doc_ids)

        total_pages = 0
        for doc_id in doc_ids:
            ocr_data = await redis_client.hgetall(f"doc:{doc_id}:ocr")
            total_pages += int(ocr_data.get('total_pages', 0) or 0)

        index_stats = pinecone_index.describe_index_stats()
        namespace_stats = index_stats.get('namespaces', {}).get(tenant_id, {})
        vector_count = namespace_stats.get('vector_count', 0)

        return {
            'tenant_id': tenant_id,
            'documents': doc_count,
            'total_pages': total_pages,
            'vector_chunks': vector_count
        }

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
