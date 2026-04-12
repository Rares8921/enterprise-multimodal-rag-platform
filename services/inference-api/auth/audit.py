from typing import Callable, Awaitable, Optional
from fastapi import Request
from .models import AuthContext

AuditLogCallback = Callable[[AuthContext, Request, str], Awaitable[None]]
_audit_logger: Optional[AuditLogCallback] = None

def set_audit_logger(callback: AuditLogCallback):
    global _audit_logger
    _audit_logger = callback

async def audit_log(auth_context: AuthContext, request: Request, event: str):
    if _audit_logger:
        await _audit_logger(auth_context, request, event)