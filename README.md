# Enterprise Multimodal RAG Platform

Multiservice document intelligence prototype for OCR, LayoutLMv3 layout parsing, vector retrieval, LLM orchestration, monitoring, and reproducible mock benchmarking.

## What This Project Demonstrates

- Async document ingestion with Redis queues, MinIO object storage, PostgreSQL metadata, and worker status updates.
- OCR processing for PDFs/images through EasyOCR.
- Layout-aware document processing with LayoutLMv3 token classification code.
- Tenant-scoped vector indexing and retrieval through Pinecone.
- Hybrid retrieval reranking that combines vector candidate scores with BM25 lexical scores.
- Labeled synthetic retrieval benchmark comparing vector-only, BM25-only, and hybrid reranking strategies with Recall@k, MRR, and nDCG.
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

The current checked-in LLM routing benchmark is mock/synthetic. The current checked-in retrieval benchmark is synthetic/offline and uses simulated vector scores. These reports support reproducibility of the methods, not real production performance.

Current retrieval benchmark evidence from `benchmarks/results/retrieval_benchmark_latest.json`:

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| vector_only | 0.9000 | 1.0000 | 1.0000 | 0.9667 | 0.9754 |
| bm25_only | 0.8667 | 0.9667 | 1.0000 | 0.9222 | 0.9468 |
| hybrid_70_30 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_30_70 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 0.9946 |

## Supported Claims

- The repository contains code for OCR-based document ingestion and LayoutLMv3 layout parsing.
- The repository contains vector retrieval over Pinecone-indexed document chunks plus BM25 reranking over retrieved candidates.
- The repository contains a labeled synthetic retrieval benchmark comparing vector-only, BM25-only, and hybrid reranking strategies.
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
- No legal or financial correctness is claimed from the synthetic retrieval benchmark.
- No claim is made that BM25 is a separate first-stage index; current hybrid retrieval reranks vector candidates with BM25.
- No LayoutLMv3 production accuracy number is claimed.

## Known Limitations

- The LLM benchmark is mock/synthetic and uses estimated latency/cost.
- The retrieval benchmark is synthetic/offline and uses simulated vector scores, not Pinecone measurements.
- Hybrid retrieval is currently a reranking layer over vector candidates, not a separate first-stage BM25 index.
- Some Docker Compose images use `latest`, which weakens environment reproducibility.
- Full local stack execution requires external services and credentials.
- Real provider benchmark mode is not implemented yet.

## CI

`.github/workflows/test-and-benchmark-smoke.yml` runs deterministic routing tests, hybrid retrieval unit tests, LLM benchmark tests, and the mock LLM benchmark without real API keys or external services. Retrieval benchmark tests can be run locally with `pytest tests\benchmark\test_retrieval_benchmark.py -q`.
