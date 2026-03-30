from typing import Callable, Awaitable, Optional
from fastapi import HTTPException, status
from .models import AuthContext
from .stores import get_quota_store

RateLimitCallback = Callable[[str, str], Awaitable[bool]]
_rate_limiter: Optional[RateLimitCallback] = None

def set_rate_limiter(callback: RateLimitCallback):
    global _rate_limiter
    _rate_limiter = callback

async def check_rate_limit(key: str, limit_type: str) -> bool:
    if _rate_limiter:
        return await _rate_limiter(key, limit_type)
    return True

async def enforce_quota(auth_context: AuthContext):
    quota_store = get_quota_store()
    quota = await quota_store.get_quota(auth_context.tenant_id)
    if not quota:
        return

    if quota.requests_per_hour:
        current_usage = await quota_store.get_current_usage(auth_context.tenant_id, "hour")
        if current_usage >= quota.requests_per_hour:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Hourly quota exceeded: {current_usage}/{quota.requests_per_hour}"
            )

    if quota.requests_per_day:
        current_usage = await quota_store.get_current_usage(auth_context.tenant_id, "day")
        if current_usage >= quota.requests_per_day:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Daily quota exceeded: {current_usage}/{quota.requests_per_day}"
            )

    await quota_store.increment_usage(auth_context.tenant_id)