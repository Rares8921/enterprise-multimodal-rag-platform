"""
Configuration for RAG evaluation framework
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class LLMJudgeConfig:
    enabled: bool = False
    model: str = "gpt-4"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass
class RetrievalConfig:
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class GroundingConfig:
    enabled: bool = True
    similarity_threshold: float = 0.5
    chunk_size: int = 512


@dataclass
class StrictGroundingConfig:
    enabled: bool = False
    use_claim_decomposition: bool = True
    use_nli_verification: bool = True
    nli_model: str = "microsoft/deberta-v2-xlarge-mnli"
    nli_device: str = "cpu"
    entailment_threshold: float = 0.7
    use_llm_claim_extraction: bool = False


@dataclass
class EvidenceMatchingConfig:
    enabled: bool = False
    fuzzy_threshold: int = 80
    token_overlap_threshold: float = 0.6
    min_span_length: int = 20
    verify_with_nli: bool = True


@dataclass
class FailureAnalysisConfig:
    enabled: bool = True
    retrieval_recall_threshold: float = 0.5
    grounding_threshold: float = 0.7
    semantic_similarity_threshold: float = 0.6
    citation_f1_threshold: float = 0.5


@dataclass
class ConfidenceCalibrationConfig:
    enabled: bool = False
    n_bins: int = 10
    save_calibration_plot: bool = False


@dataclass
class CitationConfig:
    enabled: bool = True
    min_similarity: float = 0.6


@dataclass
class PerformanceConfig:
    max_concurrent_requests: int = 10
    request_timeout: float = 60.0
    batch_size: int = 16


@dataclass
class OutputConfig:
    save_json: bool = True
    save_csv: bool = True
    output_dir: str = "evaluation_results"
    save_per_sample: bool = True
    save_worst_cases: int = 10


@dataclass
class EvaluationConfig:
    api_url: str = "http://localhost:8000"
    tenant_id: str = "default"

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm_judge: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    strict_grounding: StrictGroundingConfig = field(default_factory=StrictGroundingConfig)
    evidence_matching: EvidenceMatchingConfig = field(default_factory=EvidenceMatchingConfig)
    citation: CitationConfig = field(default_factory=CitationConfig)
    failure_analysis: FailureAnalysisConfig = field(default_factory=FailureAnalysisConfig)
    confidence_calibration: ConfidenceCalibrationConfig = field(default_factory=ConfidenceCalibrationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
