import os
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


def _require_secure_config() -> bool:
    return os.getenv("REQUIRE_SECURE_CONFIG", "true").lower() == "true"


class AuthConfig(BaseModel):
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
        # Ensure secrets are not default/weak values in production
        if _require_secure_config():
            field_name = info.field_name
            if not v or len(v) < 32:
                raise ValueError(
                    f"{field_name} must be at least 32 characters in production. "
                    f"Set REQUIRE_SECURE_CONFIG=false to disable this check in dev."
                )
        return v

    def validate_jwt_config(self):
        if self.jwt_algorithm == "HS256":
            if not self.jwt_secret_key:
                raise ValueError("JWT_SECRET_KEY required for HS256")
        elif self.jwt_algorithm == "RS256":
            if not self.jwt_public_key:
                raise ValueError("JWT_PUBLIC_KEY required for RS256 verification")
            if not self.jwt_private_key:
                raise ValueError("JWT_PRIVATE_KEY required for RS256 signing")
        else:
            raise ValueError(f"Unsupported JWT algorithm: {self.jwt_algorithm}")

# Global configuration instance
_config: Optional[AuthConfig] = None

def get_auth_config() -> AuthConfig:
    #Get or create authentication configuration singleton
    global _config
    if _config is None:
        _config = AuthConfig()
        _config.validate_jwt_config()
    return _config