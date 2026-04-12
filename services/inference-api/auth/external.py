import os
from abc import ABC, abstractmethod
from typing import Dict
from fastapi import HTTPException, status
from .config import get_auth_config

class ExternalProviderValidator(ABC):
    @abstractmethod
    async def validate_token(self, token: str) -> Dict:
        pass

# TODO: finish auth0
class Auth0Validator(ExternalProviderValidator):
    def __init__(self, jwks_url: str, audience: str, issuer: str):
        self.jwks_url = jwks_url
        self.audience = audience
        self.issuer = issuer
        self._jwks_client = None

    async def validate_token(self, token: str) -> Dict:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Auth0 validation requires PyJWT[crypto] and jwks fetching"
        )

# TODO: finish cognito
class CognitoValidator(ExternalProviderValidator):
    def __init__(self, jwks_url: str, user_pool_id: str, client_id: str):
        self.jwks_url = jwks_url
        self.user_pool_id = user_pool_id
        self.client_id = client_id

    async def validate_token(self, token: str) -> Dict:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Cognito validation requires implementation"
        )

async def validate_external_provider_token(token: str) -> Dict:
    config = get_auth_config()
    if not config.external_provider_enabled:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="External provider authentication not enabled")

    if config.external_provider_type == "auth0":
        validator = Auth0Validator(jwks_url=config.external_provider_jwks_url, audience=config.jwt_audience, issuer=config.jwt_issuer)
    elif config.external_provider_type == "cognito":
        validator = CognitoValidator(jwks_url=config.external_provider_jwks_url, user_pool_id=os.getenv("COGNITO_USER_POOL_ID", ""), client_id=os.getenv("COGNITO_CLIENT_ID", ""))
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unsupported external provider")

    return await validator.validate_token(token)