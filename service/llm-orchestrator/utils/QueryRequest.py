from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from ModelChoice import ModelChoice

class QueryRequest(BaseModel):
    query: str
    context: List[Dict[str, Any]]
    doc_type: str
    tenant_id: str
    model_choice: ModelChoice = ModelChoice.AUTO
    temperature: float = 0.1
    max_tokens: int = 1024
