import secrets
import hashlib
import hmac
import asyncio
from typing import Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException, status

from .models import Permission, APIKeyMetadata
from .config import get_auth_config
from .stores import get_api_key_store


def generate_api_key() -> str:
    prefix = "mlops"
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}_{random_part}"

def generate_api_key_with_id(key_id: str) -> str:
    prefix = "mlops_v1"
    signature = secrets.token_urlsafe(32)
    return f"{prefix}_{key_id}_{signature}"

def parse_api_key(api_key: str) -> Optional[Tuple[str, str]]:
    prefix = "mlops_v1_"
    if not api_key.startswith(prefix):
        return None

    rest = api_key[len(prefix):]
    key_id, sep, signature = rest.partition("_")
    if not sep or not key_id or not signature:
        return None

    return key_id, signature

def hash_api_key(api_key: str, salt: str, pepper: str) -> str:
    combined = f"{pepper}{api_key}{salt}".encode('utf-8')
    return hashlib.sha256(combined).hexdigest()

def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)

async def create_api_key(
    tenant_id: str,
    name: str,
    scopes: List[Permission],
    expires_in_days: Optional[int] = None,
    rate_limit_per_minute: Optional[int] = None,
    rate_limit_per_hour: Optional[int] = None,
) -> Tuple[str, APIKeyMetadata]:
    config = get_auth_config()
    store = get_api_key_store()

    key_id = f"key{secrets.token_hex(8)}"
    raw_key = generate_api_key_with_id(key_id)
    salt = secrets.token_hex(16)
    key_hash = hash_api_key(raw_key, salt, config.api_key_pepper)

    now = datetime.now(timezone.utc)
    metadata = APIKeyMetadata(
        key_id=key_id,
        key_hash=key_hash,
        salt=salt,
        tenant_id=tenant_id,
        name=name,
        scopes=scopes,
        created_at=now,
        expires_at=now + timedelta(days=expires_in_days) if expires_in_days else None,
        rate_limit_per_minute=rate_limit_per_minute,
        rate_limit_per_hour=rate_limit_per_hour,
    )

    await store.create_key(metadata)
    return raw_key, metadata


async def verify_api_key_internal(api_key: str) -> APIKeyMetadata:
    config = get_auth_config()
    store = get_api_key_store()

    parsed = parse_api_key(api_key)
    if not parsed:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key format")

    key_id, signature = parsed
    metadata = await store.get_key_metadata(key_id)
    if not metadata:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    computed_hash = hash_api_key(api_key, metadata.salt, config.api_key_pepper)
    if not constant_time_compare(computed_hash, metadata.key_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    if metadata.status == "revoked":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key has been revoked")
    if metadata.status == "expired":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key has expired")

    now = datetime.now(timezone.utc)
    if metadata.expires_at and now > metadata.expires_at:
        metadata.status = "expired"
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key has expired")

    asyncio.create_task(store.update_last_used(key_id))
    return metadata