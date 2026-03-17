"""
Authentication and user verification
"""
from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime, timedelta
import os

security = HTTPBearer()
SECRET_KEY = os.getenv("JWT_SECRET_KEY", default="secret")
ALGORITHM = "HS256"

VALID_API_KEYS = {
    "test-api-key-001": {"tenant_id": "test-tenant-001", "name": "Test Client"},
    "test-api-key-002": {"tenant_id": "test-tenant-002", "name": "Demo Client"},
}

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