import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis
from minio import Minio
from prometheus_client import Counter, Histogram

from .config import Settings
from .db import get_db, engine
from .models import Document, DocumentStatus, Base
from .ocr_engine import OCREngine
from .queue import TaskQueue
from .worker import WorkerManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DOCUMENTS_UPLOADED = Counter('documents_uploaded_total', 'Total documents uploaded', ['tenant_id', 'doc_type'])
INGESTION_DURATION = Histogram('ingestion_duration_seconds', 'Total ingestion duration')

app = FastAPI(title="Document Ingestion Service", version="1.0.0")
settings = Settings()

# Global state
redis_client: Optional[aioredis.Redis] = None
task_queue: Optional[TaskQueue] = None
minio_client: Optional[Minio] = None
ocr_engine: Optional[OCREngine] = None
worker_manager: Optional[WorkerManager] = None

VALID_MIME_TYPES = {
    'application/pdf',
    'image/jpeg',
    'image/jpg',
    'image/png'
}


@app.on_event("startup")
async def startup():
    global redis_client, task_queue, minio_client, ocr_engine, worker_manager

    # Initialize DB schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    redis_client = await aioredis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
    task_queue = TaskQueue(redis_client)

    minio_client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=False
    )

    if not minio_client.bucket_exists(settings.minio_bucket):
        minio_client.make_bucket(settings.minio_bucket)

    ocr_engine = OCREngine(settings)

    # Initialize and start concurrent background workers
    worker_manager = WorkerManager(task_queue, redis_client, minio_client, ocr_engine, settings)
    await worker_manager.start()

    logger.info("Ingestion service and workers started successfully")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    if worker_manager:
        await worker_manager.stop()
    if redis_client:
        await redis_client.close()
    await engine.dispose()
    logger.info("Ingestion service shut down")


@app.get("/health")
async def health_check():
    return {
        'status': 'healthy',
        'service': 'ingestion',
        'redis': 'connected' if redis_client else 'disconnected'
    }