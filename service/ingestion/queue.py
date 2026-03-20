import redis.asyncio as aioredis
import logging
from typing import Dict, Any
import json

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