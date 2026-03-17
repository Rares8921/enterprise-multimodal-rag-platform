import logging, random, asyncio, time, json

from fastapi import FastAPI, Header, HTTPException
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import redis.asyncio as aioredis
from pinecone import Pinecone
from minio import Minio

from config import Settings
from auth import verify_tenant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document API", version="1.0.0")
settings = Settings()

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
    """Add failed delete to Dead Letter Queue"""
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
