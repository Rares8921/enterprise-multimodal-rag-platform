import asyncio
import redis.asyncio as aioredis
from minio import Minio
import time, json, os
import mlflow
from config import Settings
from PIL import Image
import tempfile
from ..ingestion.queue import TaskQueue

import logging
from LayoutParser import LayoutParser
from pdf2image import pdfinfo_from_path, convert_from_path

# Settings
settings = Settings()

# Global state
redis_client: aioredis.Redis = None
task_queue: TaskQueue = None
minio_client: Minio = None
model = None
processor = None
device = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
from prometheus_client import Counter, Histogram

LAYOUTS_PARSED = Counter('layouts_parsed_total', 'Total layouts parsed', ['doc_type'])
LAYOUT_DURATION = Histogram('layout_parsing_seconds', 'Layout parsing duration')


async def process_layout(layout_parser: LayoutParser, task_data: dict):
    if not isinstance(task_data, dict) or 'doc_id' not in task_data or 'object_name' not in task_data:
        raise ValueError("Invalid task payload missing required fields")

    doc_id = task_data['doc_id']
    filename = task_data.get('filename', '')
    doc_type = task_data.get('doc_type', 'unknown')

    try:
        logger.info(f"Parsing layout for document {doc_id}")

        with LAYOUT_DURATION.time():
            # ocr context from redis
            ocr_data = await redis_client.hgetall(f"doc:{doc_id}:ocr")

            words_raw = ocr_data.get(b'words', b'[]')
            boxes_raw = ocr_data.get(b'boxes', b'[]')

            all_words = json.loads(words_raw.decode('utf-8')) if isinstance(words_raw, bytes) else json.loads(words_raw)
            all_boxes = json.loads(boxes_raw.decode('utf-8')) if isinstance(boxes_raw, bytes) else json.loads(boxes_raw)

            # payload to tempbuffer
            response = minio_client.get_object(settings.minio_bucket, task_data['object_name'])

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if filename.lower().endswith(
                    '.pdf') else ".img") as tmp_file:
                tmp_file.write(response.read())
                tmp_path = tmp_file.name

            response.close()
            response.release_conn()

            all_structures = []

            try:
                if filename.lower().endswith('.pdf'):
                    info = pdfinfo_from_path(tmp_path)
                    total_pages = info["Pages"]

                    for page_num in range(1, total_pages + 1):
                        # convert single page
                        images = convert_from_path(tmp_path, dpi=200, first_page=page_num, last_page=page_num)
                        if not images:
                            continue

                        # extract OCR bounds
                        page_idx = page_num - 1
                        page_words = all_words[page_idx] if all_words and isinstance(all_words[0], list) else all_words
                        page_boxes = all_boxes[page_idx] if all_boxes and isinstance(all_boxes[0], list) else all_boxes

                        if not page_words or not page_boxes:
                            logger.warning(f"No OCR data found for doc {doc_id} page {page_num}, skipping inference")
                            continue

                        # sync model in thread pool
                        structure = await asyncio.wait_for(
                            asyncio.to_thread(layout_parser.parse_document, images[0], page_words, page_boxes),
                            timeout=60.0
                        )

                        structure['page_number'] = page_num
                        all_structures.append(structure)
                else:
                    image = Image.open(tmp_path)
                    if all_words and all_boxes:
                        structure = await asyncio.wait_for(
                            asyncio.to_thread(layout_parser.parse_document, image, all_words, all_boxes),
                            timeout=60.0
                        )
                        structure['page_number'] = 1
                        all_structures.append(structure)

            finally:
                os.unlink(tmp_path)

            # store results locally
            layout_result = {
                'doc_id': doc_id,
                'total_pages': len(all_structures),
                'structures': all_structures
            }

            await redis_client.set(
                f"doc:{doc_id}:layout",
                json.dumps(layout_result)
            )

            # lightweight goes to Q
            await task_queue.enqueue('embedding', task_data)

            LAYOUTS_PARSED.labels(doc_type=doc_type).inc()
            mlflow.log_metric("layout_pages_processed", len(all_structures))

            logger.info(f"Layout parsing completed for {doc_id}")
    except Exception as e:
        logger.error(f"Error parsing layout for {doc_id}: {str(e)}")
        await redis_client.hset(f"doc:{doc_id}", "error", f"Layout parsing failed: {str(e)}")
        raise


async def process_layouts_worker(layout_parser: LayoutParser):
    #Background worker
    logger.info("Starting layout parsing worker")

    while True:
        try:
            task = await task_queue.dequeue('layout_parsing', timeout=5)

            if task:
                next_retry = task.get('next_retry_at', 0)
                if next_retry > time.time():
                    continue

                try:
                    await process_layout(layout_parser, task)
                    await task_queue.ack('layout_parsing', task)
                except Exception as e:
                    logger.error(f"Task processing error: {str(e)}")
                    await task_queue.fail('layout_parsing', task)
            else:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            await asyncio.sleep(5)


async def startup():
    global redis_client, task_queue, minio_client, model, processor

    redis_client = await aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False
    )

    task_queue = TaskQueue(redis_client)

    minio_client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=False
    )

    layout_parser = LayoutParser(settings.model_name)
    model = layout_parser.model
    processor = layout_parser.processor

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    logger.info("Layout parser service started")

    asyncio.create_task(process_layouts_worker(layout_parser))

if __name__ == "__main__":
    asyncio.run(startup())