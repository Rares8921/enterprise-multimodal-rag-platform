# Enterprise Multimodal RAG Platform

Multiservice document intelligence prototype for OCR, LayoutLMv3 layout parsing, vector retrieval, LLM orchestration, monitoring, and reproducible mock benchmarking.

## What This Project Demonstrates

- Async document ingestion with Redis queues, MinIO object storage, PostgreSQL metadata, and worker status updates.
- OCR processing for PDFs/images through EasyOCR.
- Layout-aware document processing with LayoutLMv3 token classification code.
- Tenant-scoped vector indexing and retrieval through Pinecone.
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
```

Run the mock LLM routing benchmark:

```powershell
python benchmarks\llm_routing_benchmark.py --output-dir benchmarks\results --run-id mock_latest
```

Review the checked-in benchmark evidence:

- `benchmarks/results/llm_routing_benchmark_mock_latest.json`
- `benchmarks/results/llm_routing_benchmark_mock_latest.md`

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

The unit tests and mock benchmark listed above do not start these services and do not require real provider credentials.

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
```

Full test suite:

```powershell
pytest
```

Make targets are also available:

```powershell
make test
make test-llm-routing
make test-benchmark
```

Some integration tests may require external services or heavier ML dependencies. Prefer focused tests when validating LLM routing changes.

## Running Benchmarks

Mock LLM routing benchmark:

```powershell
python benchmarks\llm_routing_benchmark.py --output-dir benchmarks\results --run-id mock_latest
```

Make target:

```powershell
make benchmark-llm-routing
```

The benchmark compares:

- `always_expensive`: Gemini for every query.
- `always_cheap`: Mistral for every query.
- `heuristic`: repository router logic.

The benchmark writes JSON, Markdown, and CSV reports. CSV is currently ignored by `.gitignore`.

## Reading Benchmark Reports

Reports include:

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

The current checked-in benchmark is mock/synthetic. It supports reproducibility of the method, not real production performance.

## Supported Claims

- The repository contains code for OCR-based document ingestion and LayoutLMv3 layout parsing.
- The repository contains vector retrieval over Pinecone-indexed document chunks.
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
- No hybrid vector-plus-BM25 retrieval claim is supported yet; BM25 is not implemented in the current repository.
- No LayoutLMv3 production accuracy number is claimed.

## Known Limitations

- The LLM benchmark is mock/synthetic and uses estimated latency/cost.
- Hybrid retrieval with BM25 is not implemented yet.
- Some Docker Compose images use `latest`, which weakens environment reproducibility.
- Full local stack execution requires external services and credentials.
- Real provider benchmark mode is not implemented yet.

## CI

`.github/workflows/test-and-benchmark-smoke.yml` runs the deterministic routing tests, benchmark tests, and mock benchmark without real API keys or external services.
