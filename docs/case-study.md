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
4. It reranks vector candidates with BM25 lexical scores and builds bounded context.
5. It calls `llm-orchestrator`.
6. The orchestrator selects a prompt and model, calls the provider wrapper, extracts citations, computes confidence, records metrics, and caches the response.

## 4. Key Technical Decisions

- Use Redis queues for OCR, layout parsing, and embedding to keep document processing asynchronous.
- Store raw documents in MinIO and metadata/status in PostgreSQL plus Redis for fast status/cache access.
- Use LayoutLMv3 for document structure extraction rather than plain text-only OCR output.
- Use tenant-scoped vector namespaces in Pinecone, then rerank vector candidates with BM25 lexical scores.
- Separate corpus manifests from raw PDF data so local/private/public evaluation corpora can be validated without committing large or sensitive files.
- Add acquisition and preflight tooling before claiming public-corpus evaluation results.
- Keep LLM routing deterministic enough to unit test with mocked providers.
- Use a mock benchmark first so routing mechanics can be reproduced without credentials.
- Add a synthetic labeled retrieval benchmark before making any retrieval-quality claim.

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

The retrieval benchmark is in `benchmarks/retrieval_benchmark.py`.

It compares:

- vector-only ranking with a deterministic `semantic_terms` cosine simulator
- BM25-only reranking over the same simulated vector candidate pool
- hybrid vector/BM25 score combinations: 70/30, 50/50, and 30/70

The fixed retrieval fixtures are:

- `benchmarks/data_samples/retrieval_documents.json`
- `benchmarks/data_samples/retrieval_queries.json`

They contain 10 synthetic legal/financial-style documents, 40 chunks, and 15 labeled queries. Query labels include exact lexical, paraphrase/semantic, numeric financial, legal clause, citation-oriented, ambiguous, distractor-heavy, BM25-helpful, and vector-helpful cases.

The retrieval benchmark writes JSON, Markdown, and optional CSV reports. The checked-in evidence is:

- `benchmarks/results/retrieval_benchmark_latest.json`
- `benchmarks/results/retrieval_benchmark_latest.md`

This benchmark is synthetic and offline. It does not call Pinecone or external embedding services, and it does not measure real legal/financial answer correctness.

The real-service document RAG evaluation harness is in `benchmarks/e2e_document_rag_eval.py`.

It supports:

- `validate-only`: validate the committed corpus manifest and local PDF references
- `ingest`: upload local PDFs through the ingestion API and record per-document statuses
- `retrieve`: query a configured Pinecone index/namespace, rerank candidates with the existing BM25 hybrid reranker, and compute Recall@k, MRR, and nDCG when labels exist
- `answer`: optionally call the query API and record lightweight answer proxy metrics

Corpus manifests live under `benchmarks/corpora/`; raw PDFs and local reports are ignored by default. This harness is intended for curated public or private-local case-study runs. No checked-in report currently proves real PDF retrieval quality, real legal/financial correctness, or production behavior.

Public corpus readiness tooling now includes:

- CUAD / Atticus Project manifest preparation for legal contracts
- SEC EDGAR filing metadata preparation for financial reports, with explicit `SEC_USER_AGENT` handling
- deterministic synthetic PDF generation for public-safe smoke tests
- preflight checks before service calls
- sanitized report promotion for reviewed local reports

These tools support a workflow toward real public-corpus evaluation. They do not replace a real run report.

## 8. Results

Current checked-in LLM routing report generated from commit: `f0d1fd0`

| Strategy | Estimated Total Cost | p50 ms | p95 ms | p99 ms | Cache Hit Rate | Keyword Overlap | Citation Presence |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_expensive | 0.40670000 | 1306.12 | 20326.14 | 26680.18 | 0.0769 | 1.0000 | 1.0000 |
| always_cheap | 0.08134000 | 783.96 | 13057.39 | 17168.82 | 0.0769 | 0.7756 | 0.4545 |
| heuristic | 0.35658280 | 783.96 | 20326.14 | 26680.18 | 0.0769 | 0.9141 | 0.8182 |

Interpretation: the heuristic selects Mistral for simple/medium requests and Gemini for complex or long-context requests in the fixed workload. The estimated cost is lower than always-expensive and higher than always-cheap. Because this is synthetic, it supports only reproducibility of the benchmark method, not real cost savings.

Current checked-in retrieval report generated from commit: `61fedd9`

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| vector_only | 0.9000 | 1.0000 | 1.0000 | 0.9667 | 0.9754 |
| bm25_only | 0.8667 | 0.9667 | 1.0000 | 0.9222 | 0.9468 |
| hybrid_70_30 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_30_70 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 0.9946 |

Interpretation: on this controlled synthetic fixture, the hybrid variants rank all labeled relevant chunks within top 5 and improve Recall@1 over either single-strategy baseline. This supports only a bounded claim that retrieval behavior is measurable on labeled synthetic fixtures.

Sanitized real-service public SEC section-level retrieval reports are checked in at `benchmarks/corpora/results/sanitized_sec_section_retrieval_summary.md` and `benchmarks/corpora/results/sanitized_sec_section_retrieval_v2_summary.md`. They evaluate 8 public SEC 10-K filings and 29 section-level queries. The v2 run enriches Pinecone chunk metadata in namespace `tenant_eval_sec_sections_v2` and uses opt-in SEC-aware reranking over vector candidates.

| Run | Namespace | Candidate Pool | Candidate Misses | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Section baseline | `tenant_eval_local` | 25 | 13 | 0.1034 | 0.2759 | 0.3448 | 0.1879 | 0.2269 |
| Section metadata + SEC-aware rerank | `tenant_eval_sec_sections_v2` | 100 | 6 | 0.7586 | 0.7931 | 0.7931 | 0.7759 | 0.7804 |

Interpretation: document-level retrieval previously showed that the correct filing could usually be found, but section-level evaluation exposed table-of-contents, adjacent-year, and wrong-section failures. The v2 result shows that section-aware indexed metadata plus SEC-aware reranking materially improves this controlled local benchmark, with no Recall@5 regressions in the 29-query comparison. It is still local section-level evidence, not production retrieval quality or legal/financial correctness.

A sanitized SEC answer proxy report is also checked in at `benchmarks/corpora/results/sanitized_sec_section_answer_summary.md`. It uses the v2 retrieval setup, Gemini, candidate pool 100, SEC-aware reranking, and a 15 second answer delay for provider rate limits.

| Queries | Failures | Non-empty answer rate | Required citation presence | Expected-hint overlap | Estimated tokens |
|---:|---:|---:|---:|---:|---:|
| 29 | 18 | 0.37931 | 0.344828 | 0.344828 | 28098 |

Interpretation: this is a lightweight answer/citation proxy, not an answer correctness benchmark. It shows that the live query path can produce grounded-looking responses for part of the SEC workload, but failures remain high and the metrics do not establish factual, legal, financial, or semantic correctness.

## 9. Limitations

- Hybrid retrieval is implemented as BM25 reranking over vector candidates, not as a separate first-stage BM25 index.
- The synthetic retrieval benchmark is offline and uses simulated vector scores; the SEC section report is separate local Pinecone-backed evidence.
- The SEC retrieval reports are section-level only, use approximate labels generated from rendered public filings, and do not include chunk-level labels.
- The SEC answer report is a lightweight proxy only; 18 of 29 queries failed and citation/hint overlap metrics are not semantic correctness.
- CUAD acquisition code exists, but no CUAD evaluation report is committed.
- The LLM benchmark does not call real model providers.
- Latency numbers are deterministic estimates, not measured provider latency.
- Quality proxy is not semantic evaluation.
- No production traffic, users, uptime, QPS, or billing records are present.
- No compliance readiness claim is supported.
- LayoutLMv3 model code exists, but no validated accuracy result is claimed here.
- Some compose images use `latest`, which weakens environment reproducibility.

## 10. What I Would Improve Next

- Reduce the remaining SEC candidate-pool misses, add independently reviewed page/chunk labels, and test whether section metadata generalizes beyond the small 8-filing corpus.
- Add a real-provider benchmark mode with opt-in credentials and strict report labeling.
- Expand retrieval evaluation datasets with more labeled queries and independent label review.
- Pin container image versions used by Docker Compose.
- Move FastAPI startup/shutdown hooks to lifespan handlers.
- Add CI smoke checks for unit tests and the mock benchmark.
- Add an evidence file mapping each CV claim to code, tests, and benchmark output.
