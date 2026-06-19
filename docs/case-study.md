# Document Intelligence RAG Case Study

## 1. Problem

The project explores a document intelligence system for legal contracts and financial reports. The system needs to ingest documents, extract text and layout structure, retrieve relevant context, and route questions to an appropriate LLM path.

The engineering challenge is not only model invocation. The harder parts are traceability, async processing, failure handling, cost-aware routing, and making benchmark claims reproducible.

## 2. Constraints

- Document inputs can be PDFs or images.
- OCR and layout extraction are slower than request/response API work, so processing is queued.
- LLM providers have different cost, context, latency, and reliability characteristics.
- Tests and benchmarks must run without real API keys.
- Public claims must be backed by code, tests, benchmark output, or clearly marked limitations.

## 3. Architecture

The repository uses separate services for ingestion, OCR/layout processing, embedding generation, inference, LLM orchestration, and monitoring. Docker Compose wires these services to Redis, PostgreSQL, MinIO, Pinecone, MLflow, Prometheus, and Grafana.

The query path is:

1. `inference-api` authenticates the tenant and applies rate/concurrency controls.
2. It checks a Redis query cache.
3. It embeds the query and retrieves candidate chunks from Pinecone.
4. It sanitizes and bounds context.
5. It calls `llm-orchestrator`.
6. The orchestrator selects a prompt and model, calls the provider wrapper, extracts citations, computes confidence, records metrics, and caches the response.

## 4. Key Technical Decisions

- Use Redis queues for OCR, layout parsing, and embedding to keep document processing asynchronous.
- Store raw documents in MinIO and metadata/status in PostgreSQL plus Redis for fast status/cache access.
- Use LayoutLMv3 for document structure extraction rather than plain text-only OCR output.
- Use tenant-scoped vector namespaces in Pinecone.
- Keep LLM routing deterministic enough to unit test with mocked providers.
- Use a mock benchmark first so routing mechanics can be reproduced without credentials.

## 5. LLM Routing Design

The router now has an explicit typed contract:

- `QueryComplexityAnalyzer.analyze()` returns `ComplexityResult`.
- `ModelRouter.route()` returns `RoutingDecision`.
- `ModelRouter.select_model()` remains as a compatibility method for the existing endpoint.

Routing considers:

- forced model for baselines/tests
- context length
- complexity score and level
- analytical query markers
- configured cost table for Gemini and Mistral

The orchestrator also validates provider responses before using them. A malformed provider payload now returns a clear 502 response instead of relying on incidental `KeyError` behavior.

## 6. Reliability Mechanisms

Implemented reliability mechanisms include:

- task retries/fail handling through Redis queue helpers
- OCR/layout worker timeouts
- document status and error persistence
- rate limiting and concurrency controls in `inference-api`
- Redis query caching and adaptive TTLs
- context sanitization and truncation before LLM calls
- LLM retry and circuit breaker logic in `inference-api`
- Gemini-to-Mistral fallback in `llm-orchestrator`
- malformed LLM response normalization
- Prometheus counters and histograms across services

These mechanisms are present in code, but they do not prove production uptime, production incident recovery, or operational maturity without deployment evidence.

## 7. Benchmark Methodology

The LLM routing benchmark is in `benchmarks/llm_routing_benchmark.py`.

It compares:

- always-expensive baseline: Gemini for every query
- always-cheap baseline: Mistral for every query
- heuristic router: `ModelRouter`

The workload is fixed and checked in at `benchmarks/data_samples/llm_routing_workload.json`. It includes simple factual, medium document QA, complex legal/financial, citation-heavy, long-context, adversarial/ambiguous, and cache-probe queries.

The benchmark writes JSON, Markdown, and CSV reports. The checked-in evidence is:

- `benchmarks/results/llm_routing_benchmark_mock_latest.json`
- `benchmarks/results/llm_routing_benchmark_mock_latest.md`

This is a mock/synthetic benchmark. It estimates cost and latency from fixed formulas and checks only lightweight quality proxies.

## 8. Results

Current checked-in report commit: `f0d1fd0`

| Strategy | Estimated Total Cost | p50 ms | p95 ms | p99 ms | Cache Hit Rate | Keyword Overlap | Citation Presence |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_expensive | 0.40670000 | 1306.12 | 20326.14 | 26680.18 | 0.0769 | 1.0000 | 1.0000 |
| always_cheap | 0.08134000 | 783.96 | 13057.39 | 17168.82 | 0.0769 | 0.7756 | 0.4545 |
| heuristic | 0.35658280 | 783.96 | 20326.14 | 26680.18 | 0.0769 | 0.9141 | 0.8182 |

Interpretation: the heuristic selects Mistral for simple/medium requests and Gemini for complex or long-context requests in the fixed workload. The estimated cost is lower than always-expensive and higher than always-cheap. Because this is synthetic, it supports only reproducibility of the benchmark method, not real cost savings.

## 9. Limitations

- Hybrid vector plus BM25 retrieval is not implemented yet.
- The LLM benchmark does not call real model providers.
- Latency numbers are deterministic estimates, not measured provider latency.
- Quality proxy is not semantic evaluation.
- No production traffic, users, uptime, QPS, or billing records are present.
- No compliance readiness claim is supported.
- LayoutLMv3 model code exists, but no validated accuracy result is claimed here.
- Some compose images use `latest`, which weakens environment reproducibility.

## 10. What I Would Improve Next

- Add BM25 lexical retrieval and an explicit hybrid vector/BM25 ranking test.
- Add a real-provider benchmark mode with opt-in credentials and strict report labeling.
- Add retrieval evaluation datasets with known relevant chunk IDs.
- Pin container image versions used by Docker Compose.
- Move FastAPI startup/shutdown hooks to lifespan handlers.
- Add CI smoke checks for unit tests and the mock benchmark.
- Add an evidence file mapping each CV claim to code, tests, and benchmark output.
