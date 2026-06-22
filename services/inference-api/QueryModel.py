from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    query: str
    tenant_id: str
    doc_type: Optional[str] = None
    doc_id: Optional[str] = None
    top_k: int = 10
    retrieval_candidate_pool: Optional[int] = None
    sec_aware_rerank: bool = False
    sec_metadata_weight: float = 0.5
    model_choice: str = "auto"
    agent: Optional[str] = None
    include_citations: bool = True

    class Config:
        max_anystr_length = 2000

    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 2000:
            raise ValueError('Query too long (max 2000 characters)')
        return v.strip()

    @validator('top_k')
    def validate_top_k(cls, v):
        if v < 1:
            raise ValueError('top_k must be at least 1')
        if v > 50:
            return 50
        return v

    @validator('retrieval_candidate_pool')
    def validate_retrieval_candidate_pool(cls, v):
        if v is None:
            return v
        if v < 1:
            raise ValueError('retrieval_candidate_pool must be at least 1')
        if v > 200:
            return 200
        return v

    @validator('sec_metadata_weight')
    def validate_sec_metadata_weight(cls, v):
        if v < 0:
            raise ValueError('sec_metadata_weight must be non-negative')
        return v

    @validator('doc_type', 'doc_id')
    def validate_strings(cls, v):
        if v and len(v) > 100:
            raise ValueError('Field too long (max 100 characters)')
        return v

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    model_used: str
    confidence_score: float
    latency_ms: float
    metadata: Dict[str, Any]
