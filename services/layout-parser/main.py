import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
import sys
from typing import Any

import redis.asyncio as aioredis
from minio import Minio

try:
    import mlflow
except Exception:  # pragma: no cover - optional in lightweight local runs
    mlflow = None

# Ensure local imports work when running as a script in Docker
HERE = Path(__file__).resolve().parent
SERVICES_DIR = HERE.parent
REPO_ROOT = SERVICES_DIR.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SERVICES_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import Settings
from services.ingestion.task_queue import TaskQueue

settings = Settings()

redis_client: aioredis.Redis = None
task_queue: TaskQueue = None
minio_client: Minio = None
model = None
processor = None
_layout_parser = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from prometheus_client import Counter, Histogram, start_http_server

LAYOUTS_PARSED = Counter('layouts_parsed_total', 'Total layouts parsed', ['doc_type'])
LAYOUT_DURATION = Histogram('layout_parsing_seconds', 'Layout parsing duration')

start_http_server(int(os.getenv("METRICS_PORT", "9103")))


async def process_layout(layout_parser: Any, task_data: dict):
    if not isinstance(task_data, dict) or 'doc_id' not in task_data or 'object_name' not in task_data:
        raise ValueError("Invalid task payload missing required fields")

    doc_id = task_data['doc_id']
    filename = task_data.get('filename', '')
    doc_type = task_data.get('doc_type', 'unknown')

    try:
        logger.info(f"Parsing layout for document {doc_id}")

        with LAYOUT_DURATION.time():
            ocr_data = await redis_client.hgetall(f"doc:{doc_id}:ocr")
            all_words = _json_redis_value(ocr_data, 'words', [])
            all_boxes = _json_redis_value(ocr_data, 'boxes', [])
            ocr_pages = _json_redis_value(ocr_data, 'pages', [])
            full_text = _redis_text(ocr_data, 'full_text')

            all_structures = []
            layout_method = "layoutlmv3"

            if not all_words or not all_boxes:
                all_structures = _text_only_structures(ocr_pages, full_text)
                layout_method = "text_only_from_ocr_pages"
                logger.info(f"Using text-only layout fallback for document {doc_id}")
            else:
                all_structures = await _layoutlm_structures(layout_parser, filename, task_data, all_words, all_boxes)

            layout_result = {
                'doc_id': doc_id,
                'total_pages': len(all_structures),
                'layout_method': layout_method,
                'structures': all_structures,
            }

            await redis_client.set(f"doc:{doc_id}:layout", json.dumps(layout_result))
            await task_queue.enqueue('embedding', task_data)

            LAYOUTS_PARSED.labels(doc_type=doc_type).inc()
            if mlflow is not None:
                try:
                    mlflow.log_metric("layout_pages_processed", len(all_structures))
                except Exception as exc:
                    logger.warning(f"MLflow metric logging skipped: {exc}")

            logger.info(f"Layout parsing completed for {doc_id} via {layout_method}")
    except Exception as e:
        logger.error(f"Error parsing layout for {doc_id}: {str(e)}")
        await redis_client.hset(f"doc:{doc_id}", "error", f"Layout parsing failed: {str(e)}")
        raise


async def _layoutlm_structures(layout_parser: Any, filename: str, task_data: dict, all_words: list, all_boxes: list) -> list[dict[str, Any]]:
    parser = layout_parser or _get_layout_parser()
    response = minio_client.get_object(settings.minio_bucket, task_data['object_name'])
    suffix = ".pdf" if filename.lower().endswith('.pdf') else ".img"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(response.read())
        tmp_path = tmp_file.name
    response.close()
    response.release_conn()

    try:
        if filename.lower().endswith('.pdf'):
            from pdf2image import pdfinfo_from_path, convert_from_path

            info = pdfinfo_from_path(tmp_path)
            total_pages = info["Pages"]
            structures = []
            for page_num in range(1, total_pages + 1):
                images = convert_from_path(tmp_path, dpi=200, first_page=page_num, last_page=page_num)
                if not images:
                    continue
                page_idx = page_num - 1
                page_words = all_words[page_idx] if all_words and isinstance(all_words[0], list) else all_words
                page_boxes = all_boxes[page_idx] if all_boxes and isinstance(all_boxes[0], list) else all_boxes
                if not page_words or not page_boxes:
                    continue
                structure = await asyncio.wait_for(
                    asyncio.to_thread(parser.parse_document, images[0], page_words, page_boxes),
                    timeout=60.0,
                )
                structure['page_number'] = page_num
                structures.append(structure)
            return structures

        from PIL import Image

        image = Image.open(tmp_path)
        structure = await asyncio.wait_for(
            asyncio.to_thread(parser.parse_document, image, all_words, all_boxes),
            timeout=60.0,
        )
        structure['page_number'] = 1
        return [structure]
    finally:
        os.unlink(tmp_path)


def _get_layout_parser() -> Any:
    global _layout_parser, model, processor
    if _layout_parser is None:
        from LayoutParser import LayoutParser

        _layout_parser = LayoutParser(settings.model_name)
        model = _layout_parser.model
        processor = _layout_parser.processor
    return _layout_parser


def _text_only_structures(ocr_pages: list[dict[str, Any]], full_text: str) -> list[dict[str, Any]]:
    if not ocr_pages and full_text:
        ocr_pages = [{"page_number": 1, "text": full_text}]
    structures = []
    for page in ocr_pages:
        page_number = int(page.get('page_number') or 1)
        text = str(page.get('text') or '').strip()
        if not text:
            continue
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
        body_text = [{"text": paragraph, "bbox": None} for paragraph in paragraphs]
        structures.append({
            "page_number": page_number,
            "titles": [],
            "sections": [],
            "tables": [],
            "lists": [],
            "headers": [],
            "footers": [],
            "signatures": [],
            "body_text": body_text,
        })
    return structures


def _json_redis_value(data: dict, key: str, default: Any) -> Any:
    value = _redis_text(data, key)
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _redis_text(data: dict, key: str) -> str:
    value = data.get(key)
    if value is None:
        value = data.get(key.encode('utf-8'))
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value or ""


async def process_layouts_worker(layout_parser: Any):
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
    global redis_client, task_queue, minio_client

    redis_client = await aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
    )

    task_queue = TaskQueue(redis_client)

    minio_client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=False,
    )

    if mlflow is not None:
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        except Exception as exc:
            logger.warning(f"MLflow tracking setup skipped: {exc}")

    logger.info("Layout parser service started")

    await process_layouts_worker(None)


if __name__ == "__main__":
    asyncio.run(startup())