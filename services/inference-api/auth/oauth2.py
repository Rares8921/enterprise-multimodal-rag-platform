from typing import List, Literal
from datetime import datetime
from pydantic import BaseModel
from fastapi import HTTPException, status
from .models import Permission

class OAuth2ClientCredentials(BaseModel):
    client_id: str
    client_secret: str
    tenant_id: str
    scopes: List[Permission]
    client_name: str
    created_at: datetime
    status: Literal["active", "revoked"] = "active"

class OAuth2TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    scope: str

# TODO: finish oauth2 login
async def oauth2_client_credentials_flow(client_id: str, client_secret: str, scope: str) -> OAuth2TokenResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="OAuth2 client credentials flow not yet implemented with storage backend"
    )