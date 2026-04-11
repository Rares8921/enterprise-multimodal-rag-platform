import asyncio
import logging

import redis.asyncio as aioredis
from minio import Minio

from config import Settings
from db import engine
from models import Base
from ocr_engine import OCREngine
from queue import TaskQueue
from worker import WorkerManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    settings = Settings()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    redis_client = await aioredis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
    task_queue = TaskQueue(redis_client)

    minio_client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=False,
    )

    ocr_engine = OCREngine(settings)

    worker_manager = WorkerManager(task_queue, redis_client, minio_client, ocr_engine, settings)
    await worker_manager.start()
    logger.info("Worker service started successfully")

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
