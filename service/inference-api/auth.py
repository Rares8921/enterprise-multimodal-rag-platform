"""
Authentication and user verification
"""
from fastapi import Header, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime, timedelta
import os
from abc import ABC

from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Set, List
from enum import Enum

security = HTTPBearer()
SECRET_KEY = os.getenv("JWT_SECRET_KEY", default="secret")
ALGORITHM = "HS256"

VALID_API_KEYS = {
    "test-api-key-001": {"tenant_id": "test-tenant-001", "name": "Test Client"},
    "test-api-key-002": {"tenant_id": "test-tenant-002", "name": "Demo Client"},
}

class AuthConfig(BaseModel):
    """
    Centralized authentication configuration.
    Load from environment variables or configuration management system.
    """
    # JWT Configuration
    jwt_secret_key: str = Field(
        default_factory=lambda: os.getenv("JWT_SECRET_KEY", ""),
        description="Secret key for JWT signing (HS256). Leave empty to require RS256."
    )
    jwt_algorithm: Literal["HS256", "RS256"] = Field(
        default_factory=lambda: os.getenv("JWT_ALGORITHM", "RS256")
    )
    jwt_public_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("JWT_PUBLIC_KEY"),
        description="Public key for RS256 verification"
    )
    jwt_private_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("JWT_PRIVATE_KEY"),
        description="Private key for RS256 signing"
    )
    jwt_issuer: str = Field(
        default_factory=lambda: os.getenv("JWT_ISSUER", "multimodal-doc-mlops")
    )
    jwt_audience: str = Field(
        default_factory=lambda: os.getenv("JWT_AUDIENCE", "inference-api")
    )
    jwt_expiration_hours: int = Field(
        default_factory=lambda: int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    )

    # OAuth2 Configuration
    oauth2_enabled: bool = Field(
        default_factory=lambda: os.getenv("OAUTH2_ENABLED", "false").lower() == "true"
    )
    oauth2_token_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("OAUTH2_TOKEN_URL")
    )
    oauth2_client_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("OAUTH2_CLIENT_ID")
    )
    oauth2_client_secret: Optional[str] = Field(
        default_factory=lambda: os.getenv("OAUTH2_CLIENT_SECRET")
    )

    # External Provider Configuration
    external_provider_enabled: bool = Field(
        default_factory=lambda: os.getenv("EXTERNAL_PROVIDER_ENABLED", "false").lower() == "true"
    )
    external_provider_type: Optional[Literal["auth0", "cognito", "okta"]] = Field(
        default_factory=lambda: os.getenv("EXTERNAL_PROVIDER_TYPE")
    )
    external_provider_jwks_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("EXTERNAL_PROVIDER_JWKS_URL")
    )

    # Security Settings
    api_key_pepper: str = Field(
        default_factory=lambda: os.getenv("API_KEY_PEPPER", ""),
        description="Global pepper for API key hashing (in addition to per-key salt)"
    )
    require_secure_config: bool = Field(
        default_factory=lambda: os.getenv("REQUIRE_SECURE_CONFIG", "true").lower() == "true"
    )

    @field_validator("jwt_secret_key", "api_key_pepper")
    @classmethod
    def validate_secrets(cls, v: str, info) -> str:
        field_name = info.field_name
        if cls.model_fields.get("require_secure_config", True):
            if not v or len(v) > 32:
                raise ValueError(
                    f"{field_name} must be at least 32 characters in production. "
                )

        return v


    def validate_jwt_config(self):
        if self.jwt_algorithm == "HS256":
            if not self.jwt_secret_key:
                raise ValueError("JWT_SECRET_KEY required for crypt")
        elif self.jwt_algorithm == "RS256":
            if not self.jwt_public_key:
                raise ValueError("JWT_PUBLIC_KEY required for verification")
            if not self.jwt_private_key:
                raise ValueError("JWT_PRIVATE_KEY required for signing")
        else:
            raise ValueError("Unsupported JWT algorithm '{}'".format(self.jwt_algorithm))


_config: Optional[AuthConfig] = None

def get_auth_config() -> AuthConfig:
    global _config
    if _config is None:
        _config = AuthConfig()
        _config.validate_jwt_config()

    return _config


# Data models
class AuthType(str, Enum):
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    EXTERNAL = "external"

class Permission(str, Enum):
    # Document permissions
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"

    # Model permissions
    MODEL_INFERENCE = "model:inference"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"

    # Admin permissions
    ADMIN_API_KEY_MANAGE = "admin:api_key:manage"
    ADMIN_TENANT_MANAGE = "admin:tenant:manage"
    ADMIN_QUOTA_MANAGE = "admin:quota:manage"


@dataclass(frozen=True)
class AuthContext:
    """
    Context used by all methods, obv. immutable to prevent tampering.
    """

    tenant_id: str
    auth_type: AuthType
    scopes: Set[Permission]

    # Optional metadata
    subject: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None

    # Rate limit & user quota
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
        if not self.has_any_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {permission}"
            )

    def enforce_quota(self) -> None:
        if self.quota_limit is not None and self.quota_used >= self.quota_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Quota exceeded for tenant {self.tenant_id}"
            )

def create_access_token(tenant_id: str, expires_delta: timedelta = timedelta(hours=24)) -> str:
    expire = datetime.utcnow() + expires_delta
    to_encode = {
        "tenant_id": tenant_id,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_jwt_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> dict:
    """Verify API key authentication"""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header."
        )

    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return VALID_API_KEYS[x_api_key]


class APIKeyMetadata(BaseModel):
    key_id: str
    key_hash: str # hash + salt + pepper
    salt: str
    tenant_id: str
    name: str
    scopes: List[Permission]

    status: Literal["active", "expired", "revoked"] = "active"

    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None

    rate_limit_per_minute: Optional[int] = None
    rate_limit_per_hour: Optional[int] = None

    rotation_reminder_at: Optional[datetime] = None
    rotated_from_key_id: Optional[str] = None


class TenantQuota(BaseModel):
    tenant_id: str
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    total_documents: Optional[int] = None
    storage_mb: Optional[int] = None

# TODO: implement storage with redis
class APIKeyStore(ABC):
    pass

class JWTBlackList(ABC):
    pass

class QuotaStore(ABC):
    pass

# TODO: Revise In memory caching
class InMemoryAPIKeyStore(APIKeyStore):
    pass

class InMemoryJWTBlackList(JWTBlackList):
    pass

class InMemoryQuotaStore(QuotaStore):
    pass

_api_key_store: Optional[APIKeyStore] = None
_jwt_blacklist: Optional[JWTBlackList] = None
_quota_store: Optional[QuotaStore] = None

def get_api_key_store() -> APIKeyStore:
    global _api_key_store
    if _api_key_store is None:
        _api_key_store = InMemoryAPIKeyStore()

    return _api_key_store

def get_jwt_blacklist() -> JWTBlackList:
    global _jwt_blacklist
    if _jwt_blacklist is None:
        _jwt_blacklist = InMemoryJWTBlackList()

    return _jwt_blacklist

def quota_store() -> QuotaStore:
    global _quota_store
    if _quota_store is None:
        _quota_store = InMemoryQuotaStore()

    return _quota_store

async def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token authentication"""
    token = credentials.credentials
    payload = verify_jwt_token(token)
    return payload


async def verify_tenant(x_tenant_id: str = Header(...)) -> str:
    """
    Verify tenant authentication (backwards compatible)

    """
    if not x_tenant_id:
        raise HTTPException(status_code=401, detail="Tenant ID required")

    # TODO: verify tenant exists and is active
    # TODO: Check rate limits, quotas, logging etc.

    return x_tenant_id


async def get_current_tenant(
        x_api_key: Optional[str] = Header(None),
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Flexible authentication supporting both API keys and JWT

    Priority:
    1. API Key (X-API-Key header)
    2. JWT Bearer token (Authorization header)
    """
    # Try API key first
    if x_api_key:
        tenant_info = await verify_api_key(x_api_key)
        return tenant_info["tenant_id"]

    # Try JWT token
    if credentials:
        payload = await verify_jwt(credentials)
        return payload.get("tenant_id")

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Provide X-API-Key or Bearer token."
    )