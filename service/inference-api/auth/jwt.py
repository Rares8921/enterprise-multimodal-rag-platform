import jwt
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from fastapi import HTTPException, status

from .models import Permission
from .config import get_auth_config
from .stores import get_jwt_blacklist

def create_access_token(
    tenant_id: str,
    subject: str,
    scopes: List[Permission],
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict] = None,
) -> str:
    config = get_auth_config()
    now = datetime.now(timezone.utc)
    if expires_delta is None:
        expires_delta = timedelta(hours=config.jwt_expiration_hours)
    expire = now + expires_delta

    claims = {
        "iss": config.jwt_issuer,
        "aud": config.jwt_audience,
        "sub": subject,
        "exp": expire,
        "iat": now,
        "jti": secrets.token_urlsafe(16),
        "tenant_id": tenant_id,
        "scopes": [scope.value for scope in scopes],
    }
    if additional_claims:
        claims.update(additional_claims)

    headers = {}
    if config.jwt_algorithm == "RS256":
        headers["kid"] = "current"

    if config.jwt_algorithm == "HS256":
        encoded_jwt = jwt.encode(
            claims,
            config.jwt_secret_key,
            algorithm=config.jwt_algorithm,
            headers=headers if headers else None
        )
    elif config.jwt_algorithm == "RS256":
        encoded_jwt = jwt.encode(
            claims,
            config.jwt_private_key,
            algorithm=config.jwt_algorithm,
            headers=headers
        )
    else:
        raise ValueError(f"Unsupported algorithm: {config.jwt_algorithm}")
    return encoded_jwt

async def verify_jwt_token(token: str) -> Dict:
    config = get_auth_config()
    blacklist = get_jwt_blacklist()

    try:
        if config.jwt_algorithm == "HS256":
            payload = jwt.decode(
                token,
                config.jwt_secret_key,
                algorithms=[config.jwt_algorithm],
                issuer=config.jwt_issuer,
                audience=config.jwt_audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "require": ["exp", "iat", "iss", "aud", "sub", "tenant_id", "scopes"],
                }
            )
        elif config.jwt_algorithm == "RS256":
            payload = jwt.decode(
                token,
                config.jwt_public_key,
                algorithms=[config.jwt_algorithm],
                issuer=config.jwt_issuer,
                audience=config.jwt_audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "require": ["exp", "iat", "iss", "aud", "sub", "tenant_id", "scopes"],
                }
            )
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unsupported algorithm")

        now = datetime.now(timezone.utc)
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        if iat > now + timedelta(minutes=5):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token issued in the future")

        jti = payload.get("jti")
        if jti and await blacklist.is_revoked(jti):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token issuer")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token audience")
    except jwt.InvalidAlgorithmError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token algorithm")
    except jwt.DecodeError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {str(e)}")

async def revoke_jwt(token: str) -> None:
    blacklist = get_jwt_blacklist()
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        jti = payload.get("jti")
        exp = payload.get("exp")

        if not jti:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token does not have a JTI claim")

        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None
        await blacklist.revoke(jti, expires_at)
    except jwt.DecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token format")