import redis.asyncio as aioredis
import logging
from typing import Dict, Any, Optional
import json
import time
import asyncio

logger = logging.getLogger(__name__)

class TaskQueue:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    @staticmethod
    def _queue_key(queue_name: str) -> str:
        return f"queue:{queue_name}"

    @staticmethod
    def _processing_key(queue_name: str) -> str:
        return f"queue:{queue_name}:processing"

    @staticmethod
    def _dlq_key(queue_name: str) -> str:
        return f"queue:{queue_name}:dlq"

    @staticmethod
    def _serialize(task_data: Dict[str, Any]) -> str:
        payload = {k: v for k, v in task_data.items() if k != "_raw"}
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    async def get_queue_size(self, queue_name: str) -> int:
        return await self.redis.llen(self._queue_key(queue_name))

    async def enqueue(self, queue_name: str, task_data: Dict[str, Any]) -> None:
        raw_task = self._serialize(task_data)
        # LPUSH to head, BRPOPLPUSH takes from tail -> FIFO
        await self.redis.lpush(self._queue_key(queue_name), raw_task)

    async def dequeue(self, queue_name: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
        source = self._queue_key(queue_name)
        destination = self._processing_key(queue_name)

        # atomic move in case of failure
        raw_task = await self.redis.brpoplpush(source, destination, timeout)

        if not raw_task:
            return None

        try:
            task_str = raw_task.decode('utf-8') if isinstance(raw_task, bytes) else raw_task
            task_data = json.loads(task_str)

            # 100% payload match
            task_data['_raw'] = task_str

            # eject tasks that are not ready yet
            if task_data.get('next_retry_at', 0) > time.time():
                await self.redis.lrem(destination, 1, raw_task)
                await self.redis.lpush(source, raw_task)
                await asyncio.sleep(0.5)  # no cpu spin
                return None

            return task_data

        except Exception as e:
            logger.error(f"Task payload decoding failed: {e}")
            # move corrupt task(s)
            await self.redis.lrem(destination, 1, raw_task)
            await self.redis.lpush(self._dlq_key(queue_name), raw_task)
            return None

    async def ack(self, queue_name: str, task_data: Dict[str, Any]) -> None:
        raw_task = task_data.get('_raw', self._serialize(task_data))
        await self.redis.lrem(self._processing_key(queue_name), 1, raw_task)


    async def fail(self, queue_name: str, task_data: Dict[str, Any], max_retries: int = 3) -> None:
        raw_task = task_data.get('_raw', self._serialize(task_data))
        destination = self._processing_key(queue_name)

        # delete the task from processing
        removed = await self.redis.lrem(destination, 1, raw_task)
        if not removed:
            return

        retries = task_data.get('retries', 0)

        # retry
        if retries < max_retries:
            task_data['retries'] = retries + 1
            # Exponential backoff 5s, 10s, 20s
            task_data['next_retry_at'] = int(time.time()) + (2 ** retries) * 5

            new_raw = self._serialize(task_data)
            await self.redis.lpush(self._queue_key(queue_name), new_raw)
            logger.warning(f"Task failed, requeued for retry {task_data['retries']}/{max_retries}")
        else:
            await self.redis.lpush(self._dlq_key(queue_name), raw_task)
            logger.error(f"Task max retries reached, moved to DLQ: {self._dlq_key(queue_name)}")