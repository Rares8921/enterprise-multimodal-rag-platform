import logging

from fastapi import FastAPI, Header, HTTPException
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import redis.asyncio as aioredis
from pinecone import Pinecone

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
