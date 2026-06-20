# Project Evidence

This file is the source of truth for CV or portfolio claims. Do not make claims that are not mapped to repository evidence here.

## Supported Claims

Claim:
The project implements an OCR-based document ingestion pipeline for PDFs and images.

Evidence:
The ingestion API validates uploads, stores raw files, persists metadata, and queues OCR work. The worker processes PDFs/images with EasyOCR and stores OCR output/status.

Files:
- `services/ingestion/main.py`
- `services/ingestion/worker.py`
- `services/ingestion/ocr_engine.py`
- `services/ingestion/queue.py`
- `docker-compose.yml`

Validation:
- Architecture documented in `docs/architecture.md`.

Limitations:
- No OCR accuracy benchmark is currently included.
- No production throughput or uptime evidence is included.

Claim:
The project contains LayoutLMv3 document layout processing code.

Evidence:
The layout parser service loads `LayoutLMv3Processor` and `LayoutLMv3ForTokenClassification`, parses OCR words/boxes, and writes structured document layout output into Redis before enqueueing embedding work.

Files:
- `services/layout-parser/LayoutParser.py`
- `services/layout-parser/main.py`
- `ml_pipeline/kubeflow/layoutlm_logic.py`
- `tests/unit/test_load_dataset.py`
- `tests/unit/test_finetune.py`
- `tests/unit/test_evaluate_model.py`
- `tests/unit/test_register_model.py`

Validation:
- Existing unit tests cover dataset/schema/training-registration logic.

Limitations:
- No validated LayoutLMv3 model accuracy number is claimed.
- No production model registry evidence is claimed beyond code/test scaffolding.

Claim:
The project implements vector-based RAG retrieval over tenant-scoped document chunks with BM25 hybrid reranking.

Evidence:
The embedding worker chunks OCR/layout output, generates sentence-transformer embeddings, and writes vectors with tenant/document metadata into Pinecone namespaces. The inference API embeds queries, retrieves matching metadata from Pinecone, then reranks vector candidates with BM25 lexical scores before building LLM context.

Files:
- `services/embedding/main.py`
- `services/embedding/EmbeddingGenerator.py`
- `services/embedding/VectorStore.py`
- `services/inference-api/main.py`
- `services/inference-api/utils/hybrid_retrieval.py`
- `tests/unit/test_hybrid_retrieval.py`
- `docker-compose.yml`

Validation:
- Query flow documented in `docs/architecture.md`.
- `python -m pytest tests\unit\test_hybrid_retrieval.py -q` passed with 5 tests.

Limitations:
- BM25 is used as a reranker over vector candidates, not as a separate first-stage index.
- No retrieval quality benchmark over labeled relevant documents is currently included.

Claim:
The project implements typed, cost-aware LLM routing and orchestration.

Evidence:
`QueryComplexityAnalyzer` returns a typed `ComplexityResult`. `ModelRouter.route()` returns a typed `RoutingDecision` and selects between Gemini and Mistral based on forced baselines, context length, complexity, analytical markers, and static cost estimates.

Files:
- `services/llm-orchestrator/complexity_analyzer.py`
- `services/llm-orchestrator/utils/ModelRouter.py`
- `services/llm-orchestrator/main.py`
- `services/llm-orchestrator/prompt_manager.py`
- `services/llm-orchestrator/specialized_prompts/`
- `tests/unit/test_llm_routing.py`

Validation:
- `python -m pytest tests\unit\test_llm_routing.py -q` passed with 11 tests.
- Combined deterministic smoke run passed with 19 tests: `python -m pytest tests\unit\test_llm_routing.py tests\unit\test_hybrid_retrieval.py tests\benchmark\test_llm_routing_benchmark.py -q`.

Limitations:
- Unit tests use mocked providers and do not call real Gemini or Mistral endpoints.
- Static cost estimates are not billing records.

Claim:
The project includes LLM reliability mechanisms for fallback, caching, citations, confidence scoring, and malformed provider responses.

Evidence:
The LLM orchestrator uses Redis response caching, stable cache keys, Gemini-to-Mistral fallback, citation extraction from `[N]` markers, confidence scoring, token/cost metrics, and provider response normalization.

Files:
- `services/llm-orchestrator/main.py`
- `tests/unit/test_llm_routing.py`

Validation:
- Unit tests cover fallback when preferred provider fails, cached response path, citation extraction, confidence behavior, token/cost accounting, and malformed provider response handling.

Limitations:
- Fallback tests use fake provider objects.
- No production incident or availability evidence is included.

Claim:
The inference API contains reliability controls around query execution.

Evidence:
The inference API includes tenant checks, rate limiting, per-tenant/global semaphores, query cache, context sanitization, context truncation, LLM retries, circuit breaker state, response shape validation, and adaptive cache TTLs.

Files:
- `services/inference-api/main.py`
- `services/inference-api/auth/`

Validation:
- Reliability mechanisms are documented in `docs/architecture.md`.

Limitations:
- No load test, QPS result, uptime report, or production failure-rate evidence is included.

Claim:
The project includes monitoring instrumentation and monitoring configuration.

Evidence:
Services expose Prometheus counters/histograms, and the repository includes Prometheus/Grafana configuration plus a monitoring service.

Files:
- `services/ingestion/main.py`
- `services/embedding/main.py`
- `services/layout-parser/main.py`
- `services/inference-api/main.py`
- `services/llm-orchestrator/main.py`
- `monitoring/prometheus/prometheus.yml`
- `monitoring/prometheus/alerts.yml`
- `monitoring/grafana/`
- `services/monitoring/`

Validation:
- Monitoring components are documented in `docs/architecture.md`.

Limitations:
- No production dashboards, alert history, SLOs, or uptime evidence is included.

Claim:
The project includes a reproducible mock LLM routing benchmark with baselines.

Evidence:
The benchmark compares `always_expensive`, `always_cheap`, and `heuristic` routing over a fixed workload. It writes JSON, Markdown, and CSV reports with command, timestamp, git commit, environment, query categories, selected model, token estimates, cost estimates, latency estimates, failure rate, cache hit rate, fallback count, and quality proxy fields.

Files:
- `benchmarks/llm_routing_benchmark.py`
- `benchmarks/data_samples/llm_routing_workload.json`
- `benchmarks/results/llm_routing_benchmark_mock_latest.json`
- `benchmarks/results/llm_routing_benchmark_mock_latest.md`
- `tests/benchmark/test_llm_routing_benchmark.py`
- `docs/llm-routing-benchmark.md`

Validation:
- `python benchmarks\llm_routing_benchmark.py --output-dir benchmarks\results --run-id mock_latest`
- `python -m pytest tests\benchmark\test_llm_routing_benchmark.py -q` passed with 3 tests.

Limitations:
- Benchmark mode is `mock_synthetic`.
- Latency is deterministic estimated latency, not measured provider latency.
- Quality proxy is not semantic evaluation.
- Costs are estimates from static token prices, not billing records.

Claim:
The current mock routing benchmark produced reproducible evidence for cost/latency tradeoff methodology.

Evidence:
Checked-in report `benchmarks/results/llm_routing_benchmark_mock_latest.json` was generated from commit `f0d1fd0`.

Files:
- `benchmarks/results/llm_routing_benchmark_mock_latest.json`
- `benchmarks/results/llm_routing_benchmark_mock_latest.md`

Validation:
- always_expensive: estimated total cost `0.40670000`, p50 `1306.12 ms`, p95 `20326.14 ms`, p99 `26680.18 ms`, cache hit rate `0.0769`, fallback count `0`.
- always_cheap: estimated total cost `0.08134000`, p50 `783.96 ms`, p95 `13057.39 ms`, p99 `17168.82 ms`, cache hit rate `0.0769`, fallback count `0`.
- heuristic: estimated total cost `0.35658280`, p50 `783.96 ms`, p95 `20326.14 ms`, p99 `26680.18 ms`, cache hit rate `0.0769`, fallback count `0`.

Limitations:
- These numbers are mock estimates and cannot be described as real cost savings, real latency, or real model accuracy.

Claim:
The project has reproducible developer hygiene for focused routing tests and benchmarks.

Evidence:
Makefile targets, `.env.example`, and a GitHub Actions smoke workflow provide repeatable commands that do not require real API keys.

Files:
- `Makefile`
- `.env.example`
- `.github/workflows/test-and-benchmark-smoke.yml`
- `README.md`

Validation:
- `python -m pytest tests\unit\test_llm_routing.py tests\unit\test_hybrid_retrieval.py tests\benchmark\test_llm_routing_benchmark.py -q` passed with 19 tests.
- `python benchmarks\llm_routing_benchmark.py --output-dir $env:TEMP\llm-routing-benchmark --run-id hygiene` wrote JSON, Markdown, and CSV reports to a temporary directory.

Limitations:
- CI smoke workflow does not run the full Docker Compose stack.
- CI smoke workflow does not validate external services or real provider credentials.

## Claims That Must Not Be Made Yet

- Do not claim production usage.
- Do not claim real users.
- Do not claim real uptime, QPS, SLA, or incident recovery.
- Do not claim production cost savings.
- Do not claim real provider latency from the mock benchmark.
- Do not claim real LLM answer accuracy from the mock benchmark.
- Do not claim legal or financial correctness.
- Do not claim compliance readiness.
- Do not claim production security guarantees.
- Do not claim BM25 is a separate first-stage retrieval index; current BM25 support is candidate reranking.
- Do not claim LayoutLMv3 production accuracy; no validated accuracy report is included.
- Do not claim retrieval quality over a labeled dataset; no such benchmark is included yet.
