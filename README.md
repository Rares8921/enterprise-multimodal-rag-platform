# Enterprise Multimodal RAG Platform

Multiservice document intelligence prototype for OCR, LayoutLMv3 layout parsing, vector retrieval, LLM orchestration, monitoring, and reproducible mock benchmarking.

## What This Project Demonstrates

- Async document ingestion with Redis queues, MinIO object storage, PostgreSQL metadata, and worker status updates.
- OCR processing for PDFs/images through EasyOCR.
- Layout-aware document processing with LayoutLMv3 token classification code.
- Tenant-scoped vector indexing and retrieval through Pinecone.
- Hybrid retrieval reranking that combines vector candidate scores with BM25 lexical scores.
- Labeled synthetic retrieval benchmark comparing vector-only, BM25-only, and hybrid reranking strategies with Recall@k, MRR, and nDCG.
- Real-service document RAG evaluation harness for curated local PDF corpora, with manifest validation, ingestion runs, Pinecone retrieval evaluation, optional answer proxy evaluation, and JSON/Markdown reports.
- Public corpus acquisition tooling for CUAD legal contracts and SEC EDGAR financial filings, generating manifest-compatible corpora without committing raw files.
- Public-safe synthetic PDF corpus generator for smoke-testing PDF ingestion workflow readiness.
- Preflight and report-promotion tooling for safer local document RAG evaluation runs.
- LLM prompt selection for legal contracts and financial reports.
- Cost-aware LLM routing between Gemini and Mistral using typed complexity scoring.
- LLM fallback, response caching, citation extraction, confidence scoring, and token/cost accounting.
- Reliability mechanisms including rate limiting, semaphores, retries, circuit breaker state, context sanitization, and task fail paths.
- Prometheus/Grafana monitoring configuration and service-level Prometheus metrics.
- Reproducible mock LLM routing benchmark with fixed workload, baselines, JSON/Markdown evidence, and tests.

## Architecture Summary

The platform is split into ingestion, OCR/layout workers, embedding generation, inference API, LLM orchestration, and monitoring services. Docker Compose wires these services to Redis, PostgreSQL, MinIO, Pinecone, MLflow, Prometheus, Grafana, and a local Mistral-compatible inference server.

See:

- `docs/architecture.md`
- `docs/case-study.md`
- `docs/llm-routing-benchmark.md`
- `benchmarks/corpora/README.md`
- `benchmarks/results/retrieval_benchmark_latest.md`

## Quickstart

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
```

Run deterministic tests that do not require external API keys:

```powershell
pytest tests\unit\test_llm_routing.py -q
pytest tests\benchmark\test_llm_routing_benchmark.py -q
pytest tests\benchmark\test_retrieval_benchmark.py -q
```

Run the mock LLM routing benchmark:

```powershell
python benchmarks\llm_routing_benchmark.py --output-dir benchmarks\results --run-id mock_latest
```

Run the synthetic offline retrieval benchmark:

```powershell
python benchmarks\retrieval_benchmark.py --output-dir benchmarks\results --run-id latest
```

Review the checked-in benchmark evidence:

- `benchmarks/results/llm_routing_benchmark_mock_latest.json`
- `benchmarks/results/llm_routing_benchmark_mock_latest.md`
- `benchmarks/results/retrieval_benchmark_latest.json`
- `benchmarks/results/retrieval_benchmark_latest.md`

## Required Services

For the full Docker Compose stack:

- PostgreSQL for document metadata and MLflow backend storage.
- Redis for queues, cache, rate-limit windows, and circuit breaker state.
- MinIO for object storage and local MLflow artifacts.
- Pinecone for vector retrieval.
- MLflow for training/benchmark tracking hooks.
- Prometheus and Grafana for metrics and dashboards.
- Mistral-compatible local inference endpoint through vLLM.
- Gemini API key for Gemini provider calls.

The focused tests, mock LLM benchmark, and synthetic retrieval benchmark listed above do not start these services and do not require real provider credentials.

## Environment Variables

Common variables:

```text
POSTGRES_DB
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_URL
REDIS_URL
MINIO_ENDPOINT
MINIO_ROOT_USER
MINIO_ROOT_PASSWORD
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
PINECONE_API_KEY
PINECONE_ENVIRONMENT
PINECONE_INDEX
GEMINI_API_KEY
MISTRAL_API_URL
LLM_ORCHESTRATOR_URL
INGESTION_SERVICE_URL
DOCUMENT_RAG_EVAL_TENANT_ID
DOCUMENT_RAG_EVAL_API_KEY
DOCUMENT_RAG_EVAL_BEARER_TOKEN
DOCUMENT_RAG_EVAL_EMBEDDING_MODEL
PINECONE_NAMESPACE
SEC_USER_AGENT
MLFLOW_TRACKING_URI
PROMETHEUS_URL
GRAFANA_ADMIN_PASSWORD
JWT_SECRET_KEY
API_KEY_PEPPER
```

Use `.env.example` for local defaults. Do not commit `.env` files or real secrets.

## Running Tests

Focused routing tests:

```powershell
pytest tests\unit\test_llm_routing.py -q
```

Benchmark runner tests:

```powershell
pytest tests\benchmark\test_llm_routing_benchmark.py -q
pytest tests\benchmark\test_retrieval_benchmark.py -q
pytest tests\benchmark\test_public_corpus_workflow.py -q
```

Full test suite:

```powershell
pytest
```

Make targets are also available:

```powershell
make test
make test-llm-routing
make test-hybrid-retrieval
make test-benchmark
```

Some integration tests may require external services or heavier ML dependencies. Prefer focused tests when validating LLM routing changes.

## Running Benchmarks

Mock LLM routing benchmark:

```powershell
python benchmarks\llm_routing_benchmark.py --output-dir benchmarks\results --run-id mock_latest
```

Synthetic offline retrieval benchmark:

```powershell
python benchmarks\retrieval_benchmark.py --output-dir benchmarks\results --run-id latest
```

Curated PDF document RAG harness:

```powershell
python benchmarks\e2e_document_rag_eval.py validate-only --manifest benchmarks\corpora\example_manifest.json --skip-file-check
```

Make target:

```powershell
make benchmark-llm-routing
```

The LLM benchmark compares:

- `always_expensive`: Gemini for every query.
- `always_cheap`: Mistral for every query.
- `heuristic`: repository router logic.

The retrieval benchmark compares:

- `vector_only`: deterministic semantic proxy ranking.
- `bm25_only`: BM25 reranking over the same simulated vector candidate pool.
- `hybrid_70_30`, `hybrid_50_50`, and `hybrid_30_70`: score-weight ablations.

Benchmarks write JSON and Markdown reports. CSV is optional for the retrieval benchmark and ignored by `.gitignore`.

## Curated PDF Corpus Evaluation

Raw PDFs should be placed under the ignored local directory:

```text
benchmarks\corpora\local_pdfs\
```

Use `benchmarks\corpora\example_manifest.json` as the manifest template. The manifest is committed, but private PDFs and local generated reports are ignored by default.

Validate a corpus manifest and local PDF references:

```powershell
python benchmarks\e2e_document_rag_eval.py validate-only --manifest benchmarks\corpora\my_manifest.json
```

Run ingestion against a local ingestion service:

```powershell
python benchmarks\e2e_document_rag_eval.py ingest --manifest benchmarks\corpora\my_manifest.json --tenant-id tenant_eval_local --ingestion-url http://localhost:8001 --poll-status --run-id local_ingest
```

Run Pinecone-backed retrieval evaluation after ingestion:

```powershell
python benchmarks\e2e_document_rag_eval.py retrieve --manifest benchmarks\corpora\my_manifest.json --tenant-id tenant_eval_local --pinecone-index doc-intelligence --ingestion-run benchmarks\corpora\results\document_rag_eval_ingest_local_ingest.json --run-id local_retrieve
```

Run optional answer proxy evaluation against the query service:

```powershell
python benchmarks\e2e_document_rag_eval.py answer --manifest benchmarks\corpora\my_manifest.json --tenant-id tenant_eval_local --query-api-url http://localhost:8000 --ingestion-run benchmarks\corpora\results\document_rag_eval_ingest_local_ingest.json --run-id local_answer
```

These commands produce local JSON/Markdown reports under `benchmarks\corpora\results\`. The reports are ignored by default unless a specific, safe report is intentionally selected for review.

## Public Corpus Workflow

Generate a public-safe synthetic PDF smoke corpus:

```powershell
python benchmarks\generate_synthetic_pdf_corpus.py --output-pdf-dir benchmarks\corpora\local_pdfs\synthetic_smoke --manifest-out benchmarks\corpora\synthetic_smoke_manifest.json --overwrite --seed 7 --num-docs 6
```

Prepare a CUAD manifest from local CUAD-style metadata without downloading PDFs:

```powershell
python benchmarks\acquire_public_corpus.py cuad --metadata-json benchmarks\corpora\local_pdfs\cuad_metadata.json --output-pdf-dir benchmarks\corpora\local_pdfs\cuad --manifest-out benchmarks\corpora\cuad_manifest.generated.json --sample-size 10
```

Prepare a SEC EDGAR manifest from local filing metadata without network access:

```powershell
python benchmarks\acquire_public_corpus.py sec-edgar --filings-json benchmarks\corpora\local_pdfs\sec_filings.json --output-file-dir benchmarks\corpora\local_pdfs\sec_edgar --manifest-out benchmarks\corpora\sec_edgar_manifest.generated.json --sample-size 6
```

Fetch SEC filing metadata only when `SEC_USER_AGENT` is set to a real contact string:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
python benchmarks\acquire_public_corpus.py sec-edgar --fetch-metadata --ticker AAPL --form-type 10-K --sample-size 1 --manifest-out benchmarks\corpora\sec_edgar_manifest.generated.json
```

Run preflight before service calls:

```powershell
python benchmarks\e2e_document_rag_eval.py preflight --preflight-target retrieve --manifest benchmarks\corpora\synthetic_smoke_manifest.json --pdf-root benchmarks\corpora\local_pdfs
```

Promote a local report into a sanitized public summary:

```powershell
python benchmarks\promote_document_rag_report.py benchmarks\corpora\results\document_rag_eval_retrieve_local.json --output-md benchmarks\corpora\results\sanitized_document_rag_summary.md
```

Make targets mirror these commands: `corpus-generate-synthetic`, `corpus-acquire-cuad`, `corpus-acquire-sec`, `corpus-preflight`, `corpus-validate`, `corpus-ingest`, `corpus-retrieve`, `corpus-answer`, and `corpus-promote-report`.

## Reading Benchmark Reports

LLM routing reports include:

- command used
- timestamp
- git commit
- environment summary
- query categories
- selected model per query
- estimated input/output tokens
- estimated cost
- estimated latency p50/p95/p99
- cache hit rate
- fallback count
- quality proxy fields
- limitations

Retrieval reports include:

- command used
- timestamp
- git commit
- environment summary
- dataset paths and query categories
- number of chunks and queries
- selected strategy per query
- Recall@1, Recall@3, Recall@5, MRR, and nDCG@5
- category-level metrics
- top-5 misses and candidate-pool misses
- limitations

Document RAG corpus reports include:

- command used
- timestamp and git commit
- environment summary
- manifest path and PDF root
- corpus document/query counts
- services used, without secrets
- ingestion success/failure counts when ingestion is run
- Recall@1, Recall@3, Recall@5, MRR, and nDCG@5 when retrieval labels are available
- optional answer proxy metrics, including non-empty answer rate, citation presence, and expected-hint overlap
- per-query rows, misses, limitations, and unsupported claims

The current checked-in LLM routing benchmark is mock/synthetic. The current checked-in retrieval benchmark is synthetic/offline and uses simulated vector scores. These reports support reproducibility of the methods, not real production performance.

The curated PDF document RAG harness now includes a checked-in sanitized local real-service SEC EDGAR retrieval report. It is public-corpus, Pinecone-backed, and section-level, but it is still environment-specific evidence and must not be described as production retrieval quality.

Current retrieval benchmark evidence from `benchmarks/results/retrieval_benchmark_latest.json`:

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| vector_only | 0.9000 | 1.0000 | 1.0000 | 0.9667 | 0.9754 |
| bm25_only | 0.8667 | 0.9667 | 1.0000 | 0.9222 | 0.9468 |
| hybrid_70_30 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_30_70 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 0.9946 |

Current public SEC section-level retrieval evidence from `benchmarks/corpora/results/sanitized_sec_section_retrieval_summary.md`:

| Corpus | Queries | Label granularity | Candidate pool | Candidate misses | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 public SEC 10-K filings | 29 | section | 25 | 13 | 0.1034 | 0.2759 | 0.3448 | 0.1879 | 0.2269 |

Interpretation: this run is useful because it exposes retrieval weaknesses at section granularity. It does not prove production retrieval quality, legal correctness, financial correctness, or chunk-level retrieval quality.

## Supported Claims

- The repository contains code for OCR-based document ingestion and LayoutLMv3 layout parsing.
- The repository contains vector retrieval over Pinecone-indexed document chunks plus BM25 reranking over retrieved candidates.
- The repository contains a labeled synthetic retrieval benchmark comparing vector-only, BM25-only, and hybrid reranking strategies.
- The repository contains a real-service evaluation harness for curated PDF corpora, supporting manifest validation, ingestion runs, Pinecone-backed retrieval evaluation, optional answer proxy evaluation, and report generation.
- The repository contains public corpus acquisition tooling for CUAD and SEC EDGAR that generates manifest-compatible corpora while keeping raw files ignored by default.
- The repository contains a checked-in sanitized SEC EDGAR section-level retrieval report from a local Pinecone-backed run, with low metrics and explicit limitations.
- The repository contains a public-safe synthetic PDF corpus generator for smoke-testing the PDF ingestion workflow.
- The repository contains preflight and report-sanitization tooling for safer local document RAG evaluation runs.
- The repository contains typed, tested LLM routing with cost-aware model selection.
- The repository contains deterministic tests for routing, prompts, fallback, caching, citations, confidence, cost estimation, and malformed provider responses.
- The repository contains a reproducible mock benchmark comparing LLM routing strategies.
- The repository contains monitoring hooks and Prometheus/Grafana configuration.

## What This Project Does Not Claim

- No production usage is claimed.
- No real users are claimed.
- No uptime, QPS, or SLA is claimed.
- No real cost savings are claimed without real billing/provider evidence.
- No compliance readiness is claimed.
- No production security guarantee is claimed.
- No real LLM answer accuracy is claimed from the mock benchmark.
- No production retrieval quality or real Pinecone performance is claimed from the synthetic retrieval benchmark.
- No real PDF/Pinecone result beyond the checked-in sanitized SEC section-level local report is claimed.
- No customer/private document evaluation is claimed.
- No CUAD evaluation result or SEC answer-quality result is claimed.
- No legal or financial correctness is claimed from the synthetic retrieval benchmark, synthetic PDF smoke corpus, or SEC section-level retrieval report.
- No claim is made that BM25 is a separate first-stage index; current hybrid retrieval reranks vector candidates with BM25.
- No LayoutLMv3 production accuracy number is claimed.

## Known Limitations

- The LLM benchmark is mock/synthetic and uses estimated latency/cost.
- The synthetic retrieval benchmark is offline and uses simulated vector scores; the SEC report is a separate local Pinecone-backed run.
- The committed SEC report is section-level only and has low top-k metrics, including 13 candidate-pool misses out of 29 queries.
- Public acquisition tooling has not produced a committed CUAD evaluation result in this repository state.
- SEC filings are often HTML and may need local rendering/conversion before PDF ingestion.
- Hybrid retrieval is currently a reranking layer over vector candidates, not a separate first-stage BM25 index.
- Some Docker Compose images use `latest`, which weakens environment reproducibility.
- Full local stack execution requires external services and credentials.
- Real provider benchmark mode is not implemented yet.

## CI

`.github/workflows/test-and-benchmark-smoke.yml` runs deterministic routing tests, hybrid retrieval unit tests, LLM benchmark tests, and the mock LLM benchmark without real API keys or external services. Retrieval benchmark tests can be run locally with `pytest tests\benchmark\test_retrieval_benchmark.py -q`.
