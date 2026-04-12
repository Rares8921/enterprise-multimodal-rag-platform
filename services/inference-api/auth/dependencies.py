from typing import Optional, List
from datetime import datetime, timezone
from fastapi import Header, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .models import AuthContext, AuthType, Permission
from .config import get_auth_config
from .stores import APIKeyStore, JWTBlacklist, QuotaStore
from .apikey import verify_api_key_internal, create_api_key
from .jwt import verify_jwt_token, create_access_token
from .rate_limit import enforce_quota, check_rate_limit, set_rate_limiter
from .audit import audit_log, set_audit_logger, AuditLogCallback
from .rate_limit import RateLimitCallback
from . import stores

class OptionalHTTPBearer(HTTPBearer):
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        try:
            return await super().__call__(request)
        except HTTPException:
            return None

optional_bearer = OptionalHTTPBearer(auto_error=False)

async def authenticate_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[AuthContext]:
    if not x_api_key:
        return None

    metadata = await verify_api_key_internal(x_api_key)
    if metadata.rate_limit_per_minute:
        allowed = await check_rate_limit(metadata.key_id, "per_minute")
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")

    now = datetime.now(timezone.utc)
    return AuthContext(
        tenant_id=metadata.tenant_id,
        auth_type=AuthType.API_KEY,
        scopes=set(metadata.scopes),
        subject=metadata.key_id,
        name=metadata.name,
        rate_limit_key=metadata.key_id,
        issued_at=metadata.created_at,
        expires_at=metadata.expires_at,
    )

async def authenticate_jwt(credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_bearer)) -> Optional[AuthContext]:
    if not credentials:
        return None

    payload = await verify_jwt_token(credentials.credentials)
    scope_strings = payload.get("scopes", [])
    scopes = set()
    for scope_str in scope_strings:
        try:
            scopes.add(Permission(scope_str))
        except ValueError:
            pass

    exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc) if payload.get("exp") else None
    iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc) if payload.get("iat") else None

    return AuthContext(
        tenant_id=payload["tenant_id"],
        auth_type=AuthType.JWT,
        scopes=scopes,
        subject=payload.get("sub"),
        email=payload.get("email"),
        name=payload.get("name"),
        rate_limit_key=payload["tenant_id"],
        issued_at=iat,
        expires_at=exp,
    )

async def get_current_auth_context(
    api_key_context: Optional[AuthContext] = Depends(authenticate_api_key),
    jwt_context: Optional[AuthContext] = Depends(authenticate_jwt),
    request: Request = None,
) -> AuthContext:
    if api_key_context:
        await enforce_quota(api_key_context)
        if request:
            await audit_log(api_key_context, request, "api_key_auth_success")
        return api_key_context

    if jwt_context:
        await enforce_quota(jwt_context)
        if request:
            await audit_log(jwt_context, request, "jwt_auth_success")
        return jwt_context

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide X-API-Key header or Authorization: Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

def require_permissions(*permissions: Permission):
    async def permission_checker(auth: AuthContext = Depends(get_current_auth_context)) -> AuthContext:
        for permission in permissions:
            auth.enforce_permission(permission)
        return auth
    return permission_checker

def require_any_permission(*permissions: Permission):
    async def permission_checker(auth: AuthContext = Depends(get_current_auth_context)) -> AuthContext:
        if not auth.has_any_permission(list(permissions)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions. Need one of: {[p.value for p in permissions]}"
            )
        return auth
    return permission_checker

async def verify_tenant(x_tenant_id: Optional[str] = Header(None)) -> str:
    """⚠DEPRECATED - Use get_current_auth_context() instead."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="verify_tenant() is deprecated. Use get_current_auth_context() instead."
    )

async def init_auth_system(
    api_key_store: Optional[APIKeyStore] = None,
    jwt_blacklist: Optional[JWTBlacklist] = None,
    quota_store: Optional[QuotaStore] = None,
    audit_callback: Optional[AuditLogCallback] = None,
    rate_limit_callback: Optional[RateLimitCallback] = None,
):
    if api_key_store:
        stores._api_key_store = api_key_store
    if jwt_blacklist:
        stores._jwt_blacklist = jwt_blacklist
    if quota_store:
        stores._quota_store = quota_store
    if audit_callback:
        set_audit_logger(audit_callback)
    if rate_limit_callback:
        set_rate_limiter(rate_limit_callback)

    config = get_auth_config()
    config.validate_jwt_config()

# Testing utils
async def create_test_api_key_for_tenant(tenant_id: str = "test-tenant-001", scopes: Optional[List[Permission]] = None):
    if scopes is None:
        scopes = [Permission.DOCUMENT_READ, Permission.DOCUMENT_WRITE, Permission.MODEL_INFERENCE]
    return await create_api_key(
        tenant_id=tenant_id, name=f"Test API Key for {tenant_id}", scopes=scopes, rate_limit_per_minute=100, rate_limit_per_hour=1000
    )

async def create_test_jwt_for_tenant(tenant_id: str = "test-tenant-001", subject: str = "test-user-001", scopes: Optional[List[Permission]] = None):
    if scopes is None:
        scopes = [Permission.DOCUMENT_READ, Permission.DOCUMENT_WRITE, Permission.MODEL_INFERENCE]
    return create_access_token(tenant_id=tenant_id, subject=subject, scopes=scopes)