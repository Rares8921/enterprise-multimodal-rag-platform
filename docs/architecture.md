# Architecture

## Service Overview

This repository is organized as a multi-service document intelligence platform:

| Area | Evidence | Responsibility |
|---|---|---|
| Ingestion | `services/ingestion` | Validate uploads, store document objects, persist metadata, enqueue processing jobs. |
| OCR | `services/ingestion/ocr_engine.py` and `services/ingestion/worker.py` | Run EasyOCR over PDFs/images and store extracted text. |
| Layout parsing | `services/layout-parser` | Use LayoutLMv3 token classification to extract document structure. |
| Embeddings | `services/embedding` | Chunk OCR/layout output, generate sentence-transformer embeddings, write vectors to Pinecone. |
| Query API | `services/inference-api` | Authenticate, rate-limit, retrieve context, call LLM orchestrator, cache responses. |
| LLM orchestration | `services/llm-orchestrator` | Select prompt, route model, call provider wrapper, fallback, cache, estimate cost. |
| Monitoring | `monitoring`, `services/monitoring`, Prometheus metrics in services | Export service metrics and provide Prometheus/Grafana configuration. |
| Benchmarks | `benchmarks`, `tests/benchmark` | Reproducible benchmark utilities, labeled synthetic retrieval fixtures, real-service corpus harness, and smoke tests. |

## Document Ingestion Flow

```mermaid
flowchart LR
  U["Client upload"] --> API["ingestion /documents/upload"]
  API --> V["Validate MIME type, doc_type, size, metadata"]
  V --> O["MinIO object storage"]
  V --> P["PostgreSQL document row"]
  V --> R["Redis document metadata"]
  R --> Q["Redis queue: ocr_processing"]
  Q --> W["worker.py OCR worker"]
  W --> E["EasyOCR"]
  E --> RO["Redis doc:{id}:ocr"]
  W --> Q2["Redis queue: layout_parsing"]
```

The ingestion service accepts PDFs and images for `legal_contract` and `financial_report` document types. It stores raw content in MinIO, metadata in PostgreSQL and Redis, and queues OCR work.

## Async Pipeline Flow

```mermaid
flowchart LR
  Q1["ocr_processing"] --> OCR["OCR worker"]
  OCR --> Q2["layout_parsing"]
  Q2 --> L["Layout parser worker"]
  L --> LM["LayoutLMv3"]
  LM --> LR["Redis layout JSON"]
  L --> Q3["embedding"]
  Q3 --> EG["Embedding worker"]
  EG --> C["Chunk text with layout"]
  C --> ST["SentenceTransformer embeddings"]
  ST --> PC["Pinecone namespace per tenant"]
```

Reliability mechanisms in this flow include task acknowledgement, retry/fail handling in `TaskQueue`, worker timeouts, status updates, and error fields on document metadata.

## Query Flow

```mermaid
flowchart LR
  Client["Client query"] --> IA["inference-api /query"]
  IA --> Auth["Auth context and tenant check"]
  Auth --> RL["Rate limit and concurrency controls"]
  RL --> Cache["Redis query cache"]
  Cache -->|hit| Resp["Return cached response"]
  Cache -->|miss| Emb["Encode query"]
  Emb --> Pinecone["Pinecone vector query"]
  Pinecone --> Hybrid["Hybrid rerank: vector score + BM25"]
  Hybrid --> Context["Sanitize and limit context"]
  Context --> LLM["llm-orchestrator /generate"]
  LLM --> Resp
```

The retrieval path uses Pinecone for first-stage vector candidates, then reranks those candidates with BM25 lexical scores in `services/inference-api/utils/hybrid_retrieval.py`. This supports a bounded hybrid retrieval claim: vector retrieval plus BM25 reranking over retrieved candidates. It is not a separate first-stage BM25 index.

## Retrieval Benchmark Flow

```mermaid
flowchart LR
  Docs["Synthetic document chunks"] --> Load["Load fixtures"]
  Queries["Labeled synthetic queries"] --> Load
  Load --> Vec["Deterministic semantic_terms cosine"]
  Vec --> Pool["Top-N simulated vector candidate pool"]
  Pool --> VOnly["vector_only ranking"]
  Pool --> BM25["BM25-only reranking"]
  Pool --> Hybrid["Hybrid vector + BM25 ablations"]
  VOnly --> Metrics["Recall@k, MRR, nDCG@5"]
  BM25 --> Metrics
  Hybrid --> Metrics
  Metrics --> Reports["JSON and Markdown reports"]
```

`benchmarks/retrieval_benchmark.py` runs fully offline over:

- `benchmarks/data_samples/retrieval_documents.json`
- `benchmarks/data_samples/retrieval_queries.json`

The benchmark compares vector-only, BM25-only reranking over the same candidate pool, and hybrid score weights. It reports overall metrics, category metrics, per-query top results, candidate-pool misses, and limitations.

The benchmark uses a deterministic semantic proxy rather than Pinecone embeddings. It supports reproducible comparison of retrieval mechanics on labeled synthetic fixtures, not production retrieval quality.

## Document RAG Evaluation Harness Flow

```mermaid
flowchart LR
  M["Committed corpus manifest"] --> V["validate-only mode"]
  P["Ignored local PDFs"] --> V
  V --> I["ingest mode via ingestion API"]
  I --> Pipe["OCR, layout, embedding pipeline"]
  Pipe --> PC["Pinecone namespace"]
  PC --> R["retrieve mode: Pinecone candidates + BM25 rerank"]
  R --> A["optional answer mode via query API"]
  R --> Reports["JSON/Markdown reports"]
  A --> Reports
```

`benchmarks/e2e_document_rag_eval.py` provides the real-service evaluation harness. Raw PDFs live under ignored local paths such as `benchmarks/corpora/local_pdfs/`, while manifests are small committed JSON files that describe document metadata and query labels. Generated reports are written under `benchmarks/corpora/results/` and are ignored by default.

The harness supports:

- `validate-only`: schema and local-file validation without service calls
- `ingest`: upload PDFs to the ingestion API and record per-document status
- `retrieve`: query a configured Pinecone index/namespace, apply the existing BM25 hybrid reranker, and compute Recall@k, MRR, and nDCG when labels are available
- `answer`: optionally call the query API and record lightweight answer proxy metrics

This harness enables local real-service case-study runs over curated PDF corpora. It does not create a production retrieval-quality claim by itself; a claim about a specific corpus requires a generated report, documented environment, and clear limitations.

## LLM Routing Flow

```mermaid
flowchart TD
  Req["QueryRequest"] --> Hash["Stable cache key"]
  Hash --> Cache{"Redis LLM cache hit?"}
  Cache -->|yes| Cached["Return QueryResponse"]
  Cache -->|no| Prompt["PromptManager selects doc_type/agent template"]
  Prompt --> Analyze["QueryComplexityAnalyzer returns ComplexityResult"]
  Analyze --> Router["ModelRouter returns RoutingDecision"]
  Router --> Choice{"Selected model"}
  Choice --> Gemini["Gemini wrapper"]
  Choice --> Mistral["Mistral wrapper"]
  Gemini -->|error| Fallback["Fallback to Mistral"]
  Mistral --> Normalize["Normalize provider response"]
  Fallback --> Normalize
  Normalize --> Cite["Extract [N] citations"]
  Cite --> Conf["Confidence heuristic"]
  Conf --> Metrics["Prometheus token/cost/latency metrics"]
  Metrics --> Store["Cache response with TTL"]
```

The router uses:

- typed complexity output from `ComplexityResult`
- context-length guard for Mistral's configured context limit
- complexity threshold from settings
- explicit force-model support for tests and baselines
- static token cost estimates by model

## Storage And Queue Components

| Component | Use |
|---|---|
| PostgreSQL | Document metadata and status persistence. |
| Redis | Task queues, status/cache data, circuit breaker state, rate-limit windows. |
| MinIO | Raw document object storage and MLflow artifact storage in local compose. |
| Pinecone | Vector index for tenant-scoped document chunks. |
| MLflow | Benchmark/training tracking hooks and LayoutLM pipeline metadata. |

## Monitoring Components

The repository includes:

- Prometheus metrics in ingestion, embedding, layout parsing, inference API, and LLM orchestrator services.
- `monitoring/prometheus/prometheus.yml` and `monitoring/prometheus/alerts.yml`.
- Grafana datasource/dashboard provisioning under `monitoring/grafana`.
- A monitoring service under `services/monitoring` with control-plane and metrics modules.

## Reliability Mechanisms

Implemented mechanisms include:

- upload validation for MIME type, document type, JSON metadata, empty files, and file size
- Redis queue acknowledgement and fail paths
- worker timeouts for OCR/layout tasks
- document status and error persistence
- inference API rate limiting
- per-tenant and global semaphores
- query cache and confidence-based cache TTLs
- context sanitization for prompt-injection patterns
- context-size truncation before LLM calls
- LLM retry loop and Redis-backed circuit breaker in inference API
- LLM fallback from Gemini to Mistral in the orchestrator
- provider response normalization and 502 errors for malformed model payloads

## Current Gaps

- Hybrid retrieval is implemented as BM25 reranking over vector candidates; the included benchmark is synthetic/offline and does not measure Pinecone production behavior.
- A real-service document RAG harness exists for local PDF corpora, but no checked-in real PDF/Pinecone report is included yet.
- The LLM routing benchmark is mock/synthetic and does not measure real providers.
- LayoutLMv3 code is present, but this documentation does not claim a validated production model accuracy number.
- The compose stack uses some `latest` images; pinning all runtime images would improve reproducibility.
- Security and compliance readiness have not been audited.
