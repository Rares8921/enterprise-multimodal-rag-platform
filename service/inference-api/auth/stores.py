from abc import ABC, abstractmethod
from typing import Optional, Dict, Set
from datetime import datetime, timezone
from .models import APIKeyMetadata, TenantQuota

class APIKeyStore(ABC):
    @abstractmethod
    async def get_key_metadata(self, key_id: str) -> Optional[APIKeyMetadata]:
        pass

    @abstractmethod
    async def find_key_by_hash(self, key_hash: str) -> Optional[APIKeyMetadata]:
        pass

    @abstractmethod
    async def create_key(self, metadata: APIKeyMetadata) -> None:
        pass

    @abstractmethod
    async def revoke_key(self, key_id: str) -> bool:
        pass

    @abstractmethod
    async def update_last_used(self, key_id: str) -> None:
        pass

class JWTBlacklist(ABC):
    @abstractmethod
    async def is_revoked(self, jti: str) -> bool:
        pass

    @abstractmethod
    async def revoke(self, jti: str, expires_at: datetime) -> None:
        pass

class QuotaStore(ABC):
    @abstractmethod
    async def get_quota(self, tenant_id: str) -> Optional[TenantQuota]:
        pass

    @abstractmethod
    async def increment_usage(self, tenant_id: str, amount: int = 1) -> int:
        pass

    @abstractmethod
    async def get_current_usage(self, tenant_id: str, window: str) -> int:
        pass


class InMemoryAPIKeyStore(APIKeyStore):
    def __init__(self):
        self._keys: Dict[str, APIKeyMetadata] = {}
        self._hash_index: Dict[str, str] = {}

    async def get_key_metadata(self, key_id: str) -> Optional[APIKeyMetadata]:
        return self._keys.get(key_id)

    async def find_key_by_hash(self, key_hash: str) -> Optional[APIKeyMetadata]:
        key_id = self._hash_index.get(key_hash)
        if key_id:
            return self._keys.get(key_id)
        return None

    async def create_key(self, metadata: APIKeyMetadata) -> None:
        self._keys[metadata.key_id] = metadata
        self._hash_index[metadata.key_hash] = metadata.key_id

    async def revoke_key(self, key_id: str) -> bool:
        if key_id in self._keys:
            self._keys[key_id].status = "revoked"
            return True
        return False

    async def update_last_used(self, key_id: str) -> None:
        if key_id in self._keys:
            self._keys[key_id].last_used_at = datetime.now(timezone.utc)


class InMemoryJWTBlacklist(JWTBlacklist):
    def __init__(self):
        self._blacklist: Set[str] = set()

    async def is_revoked(self, jti: str) -> bool:
        return jti in self._blacklist

    async def revoke(self, jti: str, expires_at: datetime) -> None:
        self._blacklist.add(jti)


class InMemoryQuotaStore(QuotaStore):
    def __init__(self):
        self._quotas: Dict[str, TenantQuota] = {}
        self._usage: Dict[str, int] = {}

    async def get_quota(self, tenant_id: str) -> Optional[TenantQuota]:
        return self._quotas.get(tenant_id)

    async def increment_usage(self, tenant_id: str, amount: int = 1) -> int:
        current = self._usage.get(tenant_id, 0)
        self._usage[tenant_id] = current + amount
        return self._usage[tenant_id]

    async def get_current_usage(self, tenant_id: str, window: str) -> int:
        return self._usage.get(tenant_id, 0)

# Global store instances
_api_key_store: Optional[APIKeyStore] = None
_jwt_blacklist: Optional[JWTBlacklist] = None
_quota_store: Optional[QuotaStore] = None

def get_api_key_store() -> APIKeyStore:
    global _api_key_store
    if _api_key_store is None:
        _api_key_store = InMemoryAPIKeyStore()
    return _api_key_store

def get_jwt_blacklist() -> JWTBlacklist:
    global _jwt_blacklist
    if _jwt_blacklist is None:
        _jwt_blacklist = InMemoryJWTBlacklist()
    return _jwt_blacklist

def get_quota_store() -> QuotaStore:
    global _quota_store
    if _quota_store is None:
        _quota_store = InMemoryQuotaStore()
    return _quota_store