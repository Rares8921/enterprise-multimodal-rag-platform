import asyncio
import redis.asyncio as aioredis
from minio import Minio
import mlflow
from config import Settings
from ..ingestion.queue import TaskQueue

import logging

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