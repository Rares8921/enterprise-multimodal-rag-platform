import asyncio
import logging
import os
import tempfile
from datetime import datetime
from PIL import Image
from pdf2image import pdfinfo_from_path, convert_from_path
from sqlalchemy import select

from .models import Document, DocumentStatus
from .db import get_worker_db
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

DOCUMENTS_PROCESSED = Counter('documents_processed_total', 'Total documents processed', ['status'])
OCR_DURATION = Histogram('ocr_processing_seconds', 'OCR processing duration')


async def process_document_task(task: dict, redis_client, minio_client, ocr_engine, settings):
    doc_id = task['doc_id']
    tenant_id = task['tenant_id']
    object_name = task['object_name']
    filename = task['filename']

    logger.info(f"Starting processing for document {doc_id}")

    # Update status to PROCESSING
    await redis_client.hset(f"doc:{doc_id}", "status", DocumentStatus.PROCESSING.value)
    async with get_worker_db() as db:
        doc = await db.get(Document, doc_id)
        if doc:
            doc.status = DocumentStatus.PROCESSING
            doc.updated_at = datetime.utcnow()

    try:
        with OCR_DURATION.time():
            # Download from MinIO to a temporary file (Safe streaming)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
                response = minio_client.get_object(settings.minio_bucket, object_name)
                for chunk in response.stream(32768):
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name

            try:
                pages_text = []

                # Process based on file type (Batching for PDFs)
                if filename.lower().endswith('.pdf'):
                    info = pdfinfo_from_path(tmp_path)
                    total_pages = info["Pages"]
                    batch_size = 5  # Configurable batch size for memory safety

                    for start_page in range(1, total_pages + 1, batch_size):
                        end_page = min(start_page + batch_size - 1, total_pages)
                        images = convert_from_path(tmp_path, first_page=start_page, last_page=end_page,
                                                   dpi=settings.ocr_dpi)

                        for i, image in enumerate(images):
                            text = await ocr_engine.process_image(image)
                            pages_text.append({
                                'page_number': start_page + i,
                                'text': text,
                                'word_count': len(text.split())
                            })
                else:
                    # Image processing
                    image = Image.open(tmp_path)
                    text = await ocr_engine.process_image(image)
                    pages_text.append({
                        'page_number': 1,
                        'text': text,
                        'word_count': len(text.split())
                    })

                ocr_result = {
                    'total_pages': len(pages_text),
                    'pages': pages_text,
                    'full_text': '\n\n'.join([p['text'] for p in pages_text])
                }
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Save results to Redis
        await redis_client.hset(f"doc:{doc_id}:ocr", mapping={
            'total_pages': ocr_result['total_pages'],
            'word_count': sum(p['word_count'] for p in ocr_result['pages']),
            'full_text': ocr_result['full_text']
        })

        # Update status to OCR_COMPLETE
        await redis_client.hset(f"doc:{doc_id}", "status", DocumentStatus.OCR_COMPLETE.value)
        async with get_worker_db() as db:
            doc = await db.get(Document, doc_id)
            if doc:
                doc.status = DocumentStatus.OCR_COMPLETE
                doc.ocr_completed_at = datetime.utcnow()
                doc.updated_at = datetime.utcnow()

        DOCUMENTS_PROCESSED.labels(status='success').inc()
        logger.info(f"Document {doc_id} OCR completed: {ocr_result['total_pages']} pages")
        return ocr_result

    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {str(e)}")
        await redis_client.hset(f"doc:{doc_id}", "status", DocumentStatus.FAILED.value)
        await redis_client.hset(f"doc:{doc_id}", "error", str(e))

        async with get_worker_db() as db:
            doc = await db.get(Document, doc_id)
            if doc:
                doc.status = DocumentStatus.FAILED
                doc.last_error = str(e)
                doc.updated_at = datetime.utcnow()

        DOCUMENTS_PROCESSED.labels(status='failed').inc()
        raise


async def worker_loop(worker_id: int, task_queue, redis_client, minio_client, ocr_engine, settings):
    logger.info(f"Worker {worker_id} started")

    while True:
        try:
            task = await task_queue.dequeue('ocr_processing', timeout=5)
            if not task:
                continue

            try:
                # Task execution with timeout protection
                ocr_result = await asyncio.wait_for(
                    process_document_task(task, redis_client, minio_client, ocr_engine, settings),
                    timeout=3600  # 1 hour max per document
                )

                # Acknowledge completion
                await task_queue.ack('ocr_processing', task)

                # Enqueue for downstream layout parsing
                await task_queue.enqueue('layout_parsing', {
                    **task,
                    'ocr_result': ocr_result
                })

            except Exception as e:
                logger.error(f"Worker {worker_id} failed on task {task.get('task_id')}: {str(e)}")
                # Handle fail and push to retry/DLQ
                await task_queue.fail('ocr_processing', task)

        except Exception as e:
            logger.error(f"Worker {worker_id} encountered fatal error: {str(e)}")
            await asyncio.sleep(5)


class WorkerManager:
    """Manages concurrent worker execution"""

    def __init__(self, task_queue, redis_client, minio_client, ocr_engine, settings):
        self.task_queue = task_queue
        self.redis_client = redis_client
        self.minio_client = minio_client
        self.ocr_engine = ocr_engine
        self.settings = settings
        self.tasks = []

    async def start(self):
        concurrency = getattr(self.settings, 'worker_concurrency', 4)
        logger.info(f"Starting {concurrency} OCR processing workers")
        for i in range(concurrency):
            task = asyncio.create_task(
                worker_loop(i, self.task_queue, self.redis_client, self.minio_client, self.ocr_engine, self.settings)
            )
            self.tasks.append(task)

    async def stop(self):
        logger.info("Stopping OCR processing workers")
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)