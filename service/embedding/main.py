import asyncio
import redis.asyncio as aioredis
import logging
import time, json

from config import Settings
from service.embedding.EmbeddingGenerator import EmbeddingGenerator
from service.embedding.VectorStore import VectorStore

settings = Settings()

from ..ingestion.queue import TaskQueue

# Global state
redis_client: aioredis.Redis = None
task_queue: TaskQueue = None
embedding_model = None
pinecone_client = None
pinecone_index = None

# Metrics
from prometheus_client import Counter, Histogram
EMBEDDINGS_GENERATED = Counter('embeddings_generated_total', 'Total embeddings generated', ['doc_type'])
EMBEDDINGS_INDEXED = Counter('embeddings_indexed_total', 'Total embeddings indexed')
EMBEDDING_DURATION = Histogram('embedding_generation_seconds', 'Embedding generation duration')
INDEXING_DURATION = Histogram('vector_indexing_seconds', 'Vector indexing duration')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_embedding(embedding_gen: EmbeddingGenerator, vector_store: VectorStore, task_data: dict):
    # Process embeddings for a given document
    if not isinstance(task_data, dict) or 'doc_id' not in task_data or 'tenant_id' not in task_data:
        raise ValueError("Invalid task payload missing required fields")

    doc_id = task_data['doc_id']
    tenant_id = task_data['tenant_id']
    doc_type = task_data.get('doc_type', 'unknown')

    try:
        logger.info(f"Generating embeddings for document {doc_id}")

        status_raw = await redis_client.hget(f"doc:{doc_id}", "status")
        if status_raw:
            status_str = status_raw.decode('utf-8') if isinstance(status_raw, bytes) else status_raw
            if status_str == "indexed":
                logger.info(f"Document {doc_id} already indexed, skipping")
                return

        with EMBEDDING_DURATION.time():
            # layout structure
            layout_json = await redis_client.get(f"doc:{doc_id}:layout")
            layout_structure = json.loads(layout_json) if layout_json else {}

            # ocr text
            ocr_data = await redis_client.hgetall(f"doc:{doc_id}:ocr")

            full_text_raw = ocr_data.get(b'full_text', ocr_data.get('full_text', b''))
            full_text = full_text_raw.decode('utf-8') if isinstance(full_text_raw, bytes) else full_text_raw

            # chunking doc
            chunks = embedding_gen.chunk_document(full_text, layout_structure)

            if not chunks:
                logger.warning(f"No valid chunks generated for document {doc_id}")
                return

            logger.info(f"Document chunked into {len(chunks)} chunks")

            # generate embeddings
            chunk_texts = [chunk['text'] for chunk in chunks]

            embeddings = await asyncio.wait_for(
                asyncio.to_thread(embedding_gen.generate_embeddings, chunk_texts),
                timeout=120.0
            )

            # pinecone vectors
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{doc_id}_{i}"
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'doc_id': doc_id,
                        'tenant_id': tenant_id,
                        'doc_type': doc_type,
                        'chunk_id': i,
                        'page': chunk.get('page', 1),
                        'type': chunk.get('type', 'text'),
                        'text': chunk['text'][:1000], # Store truncated text for display
                        'filename': task_data.get('filename', '')
                    }
                })

            with INDEXING_DURATION.time():
                # upsert to pinecone
                UPSERT_BATCH_SIZE = 100
                for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
                    batch = vectors[i:i+UPSERT_BATCH_SIZE]
                    vector_store.upsert_vectors(batch, namespace=tenant_id)

            # store metadata
            embedding_metadata = {
                'doc_id': doc_id,
                'num_chunks': len(chunks),
                'embedding_model': settings.embedding_model,
                'indexed_at': int(time.time())
            }

            await redis_client.set(f"doc:{doc_id}:embeddings", json.dumps(embedding_metadata))

            # doc status
            await redis_client.hset(f"doc:{doc_id}", "status", "indexed")

            EMBEDDINGS_GENERATED.labels(doc_type=doc_type).inc()
            EMBEDDINGS_INDEXED.inc(len(vectors))

            logger.info(f"Embeddings indexed for {doc_id}: {len(vectors)} chunks")
    except Exception as e:
        logger.error(f"Error generating embeddings for {doc_id}: {str(e)}")
        await redis_client.hset(f"doc:{doc_id}", "error", f"Embedding failed: {str(e)}")
        raise

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