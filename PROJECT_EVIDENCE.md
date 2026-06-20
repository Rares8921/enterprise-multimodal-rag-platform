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
- Retrieval-quality evidence is synthetic/offline and uses a simulated vector scoring proxy, not live Pinecone retrieval.

Claim:
Added a labeled synthetic retrieval benchmark comparing vector-only, BM25-only, and hybrid reranking strategies using Recall@k, MRR, and nDCG.

Evidence:
The benchmark runs offline over synthetic legal/financial-style document chunks and labeled queries. It compares vector-only ranking, BM25-only reranking over the same simulated vector candidate pool, and hybrid vector/BM25 score-weight ablations. It reports overall metrics, category-level metrics, per-query top results, top-5 misses, candidate-pool misses, exact command, timestamp, git commit, environment summary, dataset paths, and limitations.

Files:
- `benchmarks/retrieval_benchmark.py`
- `benchmarks/data_samples/retrieval_documents.json`
- `benchmarks/data_samples/retrieval_queries.json`
- `benchmarks/results/retrieval_benchmark_latest.json`
- `benchmarks/results/retrieval_benchmark_latest.md`
- `tests/benchmark/test_retrieval_benchmark.py`

Validation:
- `python benchmarks\retrieval_benchmark.py --output-dir benchmarks\results --run-id latest`
- `python -m pytest tests\benchmark\test_retrieval_benchmark.py -q` passed with 7 tests.
- Checked-in report was generated from commit `61fedd9` with 40 chunks, 15 labeled synthetic queries, and candidate pool size 25.
- vector_only: Recall@1 `0.9000`, Recall@3 `1.0000`, Recall@5 `1.0000`, MRR `0.9667`, nDCG@5 `0.9754`.
- bm25_only: Recall@1 `0.8667`, Recall@3 `0.9667`, Recall@5 `1.0000`, MRR `0.9222`, nDCG@5 `0.9468`.
- hybrid_70_30: Recall@1 `0.9667`, Recall@3 `1.0000`, Recall@5 `1.0000`, MRR `1.0000`, nDCG@5 `1.0000`.
- hybrid_50_50: Recall@1 `0.9667`, Recall@3 `1.0000`, Recall@5 `1.0000`, MRR `1.0000`, nDCG@5 `1.0000`.
- hybrid_30_70: Recall@1 `0.9667`, Recall@3 `1.0000`, Recall@5 `1.0000`, MRR `1.0000`, nDCG@5 `0.9946`.

Limitations:
- Dataset is synthetic and should not be described as customer, private, or production data.
- Vector scores come from a deterministic `semantic_terms` cosine simulator, not Pinecone embeddings.
- Results compare retrieval mechanics on controlled fixtures only.
- This does not prove production retrieval quality, legal correctness, financial correctness, real latency, QPS, or customer data behavior.

Claim:
Built a real-service evaluation harness for curated PDF corpora, supporting ingestion validation, Pinecone-backed retrieval evaluation, optional answer proxy evaluation, and report generation with Recall@k, MRR, and nDCG.

Evidence:
The harness validates committed corpus manifests and ignored local PDF references, can upload PDFs through the existing ingestion API, can query a configured Pinecone index/namespace and apply the repository BM25 hybrid reranker, and can optionally call the query API for lightweight answer proxy checks. Reports include command, timestamp, git commit, environment summary, corpus counts, services used without secrets, mode-specific metrics, per-query rows, limitations, and unsupported claims.

Files:
- `benchmarks/e2e_document_rag_eval.py`
- `benchmarks/corpus_manifest.py`
- `benchmarks/corpora/README.md`
- `benchmarks/corpora/example_manifest.json`
- `benchmarks/corpora/.gitignore`
- `tests/benchmark/test_corpus_manifest.py`

Validation:
- `python -m py_compile benchmarks\e2e_document_rag_eval.py`
- `python -m pytest tests\benchmark\test_corpus_manifest.py -q` passed with 7 tests.
- `python benchmarks\e2e_document_rag_eval.py validate-only --manifest benchmarks\corpora\example_manifest.json --skip-file-check --output-dir $env:TEMP\document-rag-eval --run-id report_validate --write-csv`
- `python benchmarks\e2e_document_rag_eval.py answer --manifest benchmarks\corpora\example_manifest.json --skip-file-check --query-api-url http://127.0.0.1:9 --request-timeout-seconds 1 --output-dir $env:TEMP\document-rag-eval --run-id report_answer`

Limitations:
- No real local PDF corpus run is committed.
- Ingest, retrieve, and answer modes require local services, credentials, and PDFs.
- No Pinecone-backed retrieval metrics are claimed yet from a real PDF corpus.
- Answer mode is a lightweight proxy and does not prove semantic correctness, legal correctness, financial correctness, or provider accuracy.

Claim:
Added public corpus acquisition tooling for CUAD legal contracts and SEC EDGAR financial filings, generating manifest-compatible corpora for the document RAG evaluation harness.

Evidence:
The acquisition CLI has CUAD and SEC EDGAR subcommands. CUAD support parses local CUAD-style metadata, prepares manifests, and can optionally copy local PDFs or download explicit PDF URLs. SEC support uses a curated ticker/CIK list, supports no-network filing metadata, requires a real `SEC_USER_AGENT` for SEC network fetches, rate-limits requests, records source format, and generates public manifests.

Files:
- `benchmarks/acquire_public_corpus.py`
- `benchmarks/corpus_sources/cuad.py`
- `benchmarks/corpus_sources/sec_edgar.py`
- `benchmarks/corpora/public_sources.json`
- `benchmarks/corpora/sec_edgar_sample_companies.json`
- `tests/benchmark/test_public_corpus_workflow.py`

Validation:
- `python -m pytest tests\benchmark\test_public_corpus_workflow.py -q` passed with 12 tests.
- No-network CUAD smoke with mocked metadata generated a manifest and validate-only smoke passed.
- No-network SEC smoke with mocked filing metadata generated an HTML-source manifest and validate-only smoke passed.
- SEC fetch mode fails clearly when `SEC_USER_AGENT` is missing.

Limitations:
- No real CUAD or SEC documents were downloaded or committed.
- No public-corpus ingestion, Pinecone retrieval, or answer evaluation result is claimed.
- CUAD source/license terms must be reviewed before redistributing raw files or selected reports.
- SEC HTML filings may require local rendering/conversion before PDF ingestion.

Claim:
Added a public-safe synthetic PDF corpus generator for smoke-testing the real document ingestion workflow.

Evidence:
The generator writes deterministic small legal-style and financial-style PDFs to ignored local corpus storage and writes a safe committed manifest with document/page-level labels, expected answer hints, and citation requirements.

Files:
- `benchmarks/generate_synthetic_pdf_corpus.py`
- `benchmarks/corpora/synthetic_smoke_manifest.json`
- `tests/benchmark/test_public_corpus_workflow.py`

Validation:
- `python -m py_compile benchmarks\generate_synthetic_pdf_corpus.py`
- Temp synthetic generation plus validate-only with file checks passed.
- Default synthetic generation plus validate-only with file checks passed.
- `python -m pytest tests\benchmark\test_public_corpus_workflow.py -q` passed with 12 tests.

Limitations:
- The PDFs are synthetic smoke fixtures, not real legal or financial documents.
- Results from this corpus cannot be described as public legal/financial retrieval quality.

Claim:
Added preflight and report-sanitization tooling for safe local document RAG evaluation runs.

Evidence:
Preflight mode checks manifest validity, local files, corpus warnings, output writability, tracked artifact safety, service reachability, Pinecone configuration, embedding model configuration, and SEC User-Agent readiness. Report promotion turns local JSON reports into sanitized public Markdown/JSON summaries while redacting local paths and content-bearing fields and refusing private-local reports by default.

Files:
- `benchmarks/e2e_document_rag_eval.py`
- `benchmarks/promote_document_rag_report.py`
- `tests/benchmark/test_public_corpus_workflow.py`

Validation:
- `python -m py_compile benchmarks\e2e_document_rag_eval.py benchmarks\promote_document_rag_report.py`
- Validate-only preflight smoke passed.
- Retrieve preflight with missing Pinecone key/index produced a failure report.
- Synthetic report promotion to temp Markdown/JSON passed.
- Private-local report promotion was refused without `--allow-private-summary`.
- `python -m pytest tests\benchmark\test_public_corpus_workflow.py -q` passed with 12 tests.

Limitations:
- Preflight checks readiness only; they do not prove service correctness, retrieval quality, or production readiness.
- Sanitized report promotion still requires human review before a report is selected as public evidence.

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
- `python -m pytest tests\benchmark\test_retrieval_benchmark.py -q` passed with 7 tests.

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
- Do not claim production retrieval quality or real Pinecone performance from the synthetic offline retrieval benchmark.
- Do not claim the document RAG harness has produced real PDF/Pinecone results until a real-service report exists.
- Do not claim customer/private document evaluation.
- Do not claim the synthetic retrieval benchmark proves legal or financial correctness.
