"""
Authentication and user verification
"""
from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime, timedelta
import os

from pydantic import BaseModel, Field
from typing import Literal

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