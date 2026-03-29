from typing import Dict, List, Any
from pydantic import BaseModel

class QueryResponse(BaseModel):
    answer: str
    model_used: str
    citations: List[Dict[str, Any]]
    confidence_score: float
    tokens_used: int
    latency_ms: float
