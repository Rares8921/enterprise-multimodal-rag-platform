import asyncio
import redis.asyncio as aioredis
import logging
import time

from config import Settings
from service.embedding.EmbeddingGenerator import EmbeddingGenerator
from service.embedding.VectorStore import VectorStore

settings = Settings()

from ..ingestion.queue import TaskQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_embeddings_worker(embedding_gen: EmbeddingGenerator, vector_store: VectorStore):
    """Background worker for embedding generation"""
    logger.info("Starting embedding worker")

    while True:
        try:
            # Dequeue task
            task = await task_queue.dequeue('embedding', timeout=5)

            if task:
                next_retry = task.get('next_retry_at', 0)
                if next_retry > time.time():
                    continue

                try:
                    await process_embedding(embedding_gen, vector_store, task)
                    await task_queue.ack('embedding', task)
                except Exception as e:
                    logger.error(f"Task processing error: {str(e)}")
                    await task_queue.fail('embedding', task)
            else:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            await asyncio.sleep(5)


async def startup():
    """Initialize service"""
    global redis_client, task_queue, embedding_model, pinecone_client, pinecone_index

    # Redis
    redis_client = await aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False
    )

    # Task queue
    task_queue = TaskQueue(redis_client)

    # Embedding model
    embedding_generator = EmbeddingGenerator(settings.embedding_model)
    embedding_model = embedding_generator

    # Pinecone vector store
    vector_store = VectorStore(
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_environment,
        index_name=settings.pinecone_index,
        dimension=embedding_generator.embedding_dim
    )
    pinecone_client = vector_store
    pinecone_index = vector_store.index

    logger.info("Embedding service started")

    # Start worker
    asyncio.create_task(process_embeddings_worker(embedding_generator, vector_store))


if __name__ == "__main__":
    asyncio.run(startup())