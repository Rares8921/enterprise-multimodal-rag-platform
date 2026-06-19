from typing import Dict, List, Any
from pydantic import BaseModel

try:
    from .ModelChoice import ModelChoice
except ImportError:  # pragma: no cover - supports direct script-style imports
    from ModelChoice import ModelChoice

class QueryRequest(BaseModel):
    query: str
    context: List[Dict[str, Any]]
    doc_type: str
    tenant_id: str
    agent: str = "default"
    model_choice: ModelChoice = ModelChoice.AUTO
    temperature: float = 0.1
    max_tokens: int = 1024
