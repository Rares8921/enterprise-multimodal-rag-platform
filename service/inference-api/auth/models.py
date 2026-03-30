#Data models for authentication and authorization.
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Set, Literal
from datetime import datetime
from pydantic import BaseModel
from fastapi import HTTPException, status

class AuthType(str, Enum):
    #Authentication method used
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    EXTERNAL = "external"

class Permission(str, Enum):
    #Fine-grained permissions
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    MODEL_INFERENCE = "model:inference"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    ADMIN_API_KEY_MANAGE = "admin:api_key:manage"
    ADMIN_TENANT_MANAGE = "admin:tenant:manage"
    ADMIN_QUOTA_MANAGE = "admin:quota:manage"

@dataclass(frozen=True)
class AuthContext:
    """
    Unified authentication context returned by all auth methods.
    Immutable to prevent tampering after verification.
    """
    tenant_id: str
    auth_type: AuthType
    scopes: Set[Permission]

    subject: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None

    rate_limit_key: Optional[str] = None
    quota_used: int = 0
    quota_limit: Optional[int] = None

    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.scopes

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        return bool(self.scopes & set(permissions))

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        return set(permissions).issubset(self.scopes)

    def enforce_permission(self, permission: Permission) -> None:
        if not self.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {permission.value}"
            )

    def enforce_quota(self) -> None:
        if self.quota_limit is not None and self.quota_used >= self.quota_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Quota exceeded for tenant {self.tenant_id}"
            )

class APIKeyMetadata(BaseModel):
    #Metadata for API key (stored in database/Redis)
    key_id: str
    key_hash: str
    salt: str
    tenant_id: str
    name: str
    scopes: List[Permission]
    status: Literal["active", "revoked", "expired"] = "active"
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    rate_limit_per_minute: Optional[int] = None
    rate_limit_per_hour: Optional[int] = None
    rotation_reminder_at: Optional[datetime] = None
    rotated_from_key_id: Optional[str] = None

class TenantQuota(BaseModel):
    #Per-tenant quota information
    tenant_id: str
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    total_documents: Optional[int] = None
    storage_mb: Optional[int] = None