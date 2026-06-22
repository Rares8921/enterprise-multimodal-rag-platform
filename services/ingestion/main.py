import logging
import io
from datetime import datetime
import json, hashlib, uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis
from minio import Minio
from prometheus_client import Counter, Histogram, make_asgi_app

from .config import Settings
from .db import get_db, engine
from .models import Document, DocumentStatus, Base
from .task_queue import TaskQueue

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DOCUMENTS_UPLOADED = Counter('documents_uploaded_total', 'Total documents uploaded', ['tenant_id', 'doc_type'])
INGESTION_DURATION = Histogram('ingestion_duration_seconds', 'Total ingestion duration')

app = FastAPI(title="Document Ingestion Service", version="1.0.0")
settings = Settings()

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Global state
redis_client: Optional[aioredis.Redis] = None
task_queue: Optional[TaskQueue] = None
minio_client: Optional[Minio] = None
ocr_engine: Optional[object] = None
worker_manager: Optional[object] = None

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

    if settings.enable_workers:
        from .ocr_engine import OCREngine
        from .worker import WorkerManager

        ocr_engine = OCREngine(settings)
        worker_manager = WorkerManager(task_queue, redis_client, minio_client, ocr_engine, settings)
        await worker_manager.start()
        logger.info("Ingestion service and workers started successfully")
    else:
        logger.info("Ingestion service started (workers disabled via ENABLE_WORKERS)")


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


@app.get("/documents/{doc_id}/status")
async def get_document_status(doc_id: str, db: AsyncSession = Depends(get_db)):
    # DB for persistence, fallback to Redis
    doc = await db.get(Document, doc_id)
    if not doc:
        # Fallback to Redis just in case
        doc_data = await redis_client.hgetall(f"doc:{doc_id}")
        if not doc_data:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            'doc_id': doc_id,
            'status': doc_data.get('status'),
            'filename': doc_data.get('filename'),
            'uploaded_at': doc_data.get('uploaded_at'),
            'error': doc_data.get('error')
        }

    return {
        'doc_id': doc.doc_id,
        'status': doc.status.value,
        'filename': doc.filename,
        'uploaded_at': doc.uploaded_at.isoformat() if doc.uploaded_at else None,
        'error': doc.last_error
    }


@app.post("/documents/upload")
async def upload_document(
        file: UploadFile = File(...),
        tenant_id: str = Form(...),
        doc_type: str = Form(...),
        metadata: Optional[str] = Form(None),
        db: AsyncSession = Depends(get_db)
):
    with INGESTION_DURATION.time():
        try:
            # Input Validation
            if file.content_type not in VALID_MIME_TYPES:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

            if doc_type not in ['legal_contract', 'financial_report']:
                raise HTTPException(status_code=400, detail="Invalid document type")

            parsed_metadata = None
            if metadata:
                try:
                    parsed_metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid JSON format for metadata")

            # Safely read and enforce size limit
            max_bytes = getattr(settings, 'max_file_size_mb', 100) * 1024 * 1024
            content = await file.read()
            if len(content) > max_bytes:
                raise HTTPException(status_code=413,
                                    detail=f"File exceeds maximum size of {settings.max_file_size_mb}MB")
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty file")

            # Content deduplication hash
            content_hash = hashlib.sha256(content).hexdigest()
            doc_id = str(uuid.uuid4())

            # Save to MinIO
            object_name = f"{tenant_id}/{doc_id}/{file.filename}"
            minio_client.put_object(
                settings.minio_bucket,
                object_name,
                io.BytesIO(content),
                length=len(content),
                content_type=file.content_type
            )

            # persist to PostgreSQL
            new_doc = Document(
                doc_id=doc_id,
                tenant_id=tenant_id,
                doc_type=doc_type,
                filename=file.filename,
                bucket_name=settings.minio_bucket,
                object_name=object_name,
                content_hash=content_hash,
                file_size=len(content),
                status=DocumentStatus.UPLOADED,
                document_metadata=parsed_metadata,
                ocr_engine=settings.ocr_engine
            )
            db.add(new_doc)
            await db.flush()  # Ensure it flushes to DB safely

            # Save to Redis
            doc_meta_dict = {
                'doc_id': doc_id,
                'tenant_id': tenant_id,
                'doc_type': doc_type,
                'filename': file.filename,
                'object_name': object_name,
                'file_size': len(content),
                'status': DocumentStatus.UPLOADED.value,
                'uploaded_at': datetime.utcnow().isoformat()
            }
            await redis_client.hset(f"doc:{doc_id}", mapping=doc_meta_dict)

            # Queue for OCR
            await task_queue.enqueue('ocr_processing', doc_meta_dict)
            DOCUMENTS_UPLOADED.labels(tenant_id=tenant_id, doc_type=doc_type).inc()
            logger.info(f"Document uploaded: {doc_id} for tenant {tenant_id}")

            return {
                'doc_id': doc_id,
                'status': DocumentStatus.UPLOADED.value,
                'message': 'Document queued for processing'
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error during upload")