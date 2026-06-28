# Repository Guide

This guide is for a reviewer who wants to know where the important pieces live before reading code in detail.

## Start Here

| File | Why it matters |
|---|---|
| `README.md` | Main entry point, supported claims, commands, and checked-in evidence. |
| `PROJECT_EVIDENCE.md` | Source of truth for CV-safe claims, validation commands, files, and limitations. |
| `docs/architecture.md` | Component diagrams and service/data-flow explanation. |
| `docs/case-study.md` | Narrative explanation of the problem, design choices, evaluation, and limitations. |
| `benchmarks/corpora/results/sanitized_sec_section_retrieval_v2_summary.md` | Strongest checked-in real-service evidence: local SEC section-level retrieval v2. |
| `benchmarks/corpora/results/sanitized_sec_section_answer_summary.md` | Bounded answer/citation proxy evidence with failures and limitations. |

## Core Services

| Path | What it contains | Where it fits |
|---|---|---|
| `services/ingestion/main.py` | FastAPI upload/status API, validation, object storage, metadata writes, OCR queue enqueueing. | Entry point for document ingestion. |
| `services/ingestion/worker.py` | OCR worker loop. | Consumes Redis OCR jobs and queues layout work. |
| `services/ingestion/ocr_engine.py` | EasyOCR/PDF text extraction helpers. | Converts uploaded PDFs/images into OCR text. |
| `services/ingestion/task_queue.py` | Redis queue helper with acknowledgement/failure paths. | Shared queue reliability layer for processing jobs. |
| `services/layout-parser/main.py` | Layout parser worker process. | Consumes layout jobs after OCR. |
| `services/layout-parser/LayoutParser.py` | LayoutLMv3 processor/model wrapper. | Converts OCR words/boxes into structured layout output. |
| `services/embedding/main.py` | Embedding worker and chunking flow. | Converts processed documents into vector records. |
| `services/embedding/EmbeddingGenerator.py` | Sentence-transformer embedding wrapper. | Generates chunk embeddings. |
| `services/embedding/VectorStore.py` | Pinecone vector-store integration. | Writes/searches tenant-scoped vector chunks. |
| `services/inference-api/main.py` | Query API, cache/rate limits, retrieval, context construction, LLM call handling. | Main question-answering service. |
| `services/inference-api/utils/hybrid_retrieval.py` | BM25 scoring, hybrid reranking, SEC-aware reranking helpers. | Retrieval quality logic used by query/evaluation flows. |
| `services/llm-orchestrator/main.py` | Generation endpoint, provider call flow, caching, metrics, validation. | LLM orchestration service. |
| `services/llm-orchestrator/complexity_analyzer.py` | Typed query complexity scoring. | Input to cost-aware model routing. |
| `services/llm-orchestrator/utils/ModelRouter.py` | Model routing decisions and static cost estimates. | Chooses Gemini/Mistral paths for requests and benchmark baselines. |
| `services/llm-orchestrator/prompt_manager.py` | Prompt template loading and selection. | Chooses document-type/task prompt templates. |

## Evaluation And Evidence

| Path | What it does | Evidence boundary |
|---|---|---|
| `benchmarks/llm_routing_benchmark.py` | Runs mock/synthetic LLM routing baselines. | Reproducible routing methodology, not real provider performance. |
| `benchmarks/retrieval_benchmark.py` | Runs synthetic offline retrieval benchmark over labeled fixtures. | Retrieval mechanics on synthetic data, not Pinecone production quality. |
| `benchmarks/e2e_document_rag_eval.py` | Validates corpora, uploads PDFs, runs Pinecone-backed retrieval evaluation, and optional answer proxy evaluation. | Local real-service harness; reports must state corpus/mode/limitations. |
| `benchmarks/sec_section_labeler.py` | Generates conservative SEC section labels from rendered public filings. | Section labels are approximate and not legal/financial correctness labels. |
| `benchmarks/sec_metadata_enrichment.py` | Copies/enriches Pinecone metadata into a section-aware namespace. | Used for the checked-in SEC retrieval v2 evidence. |
| `benchmarks/promote_document_rag_report.py` | Sanitizes local JSON reports into public Markdown summaries. | Removes local paths/content-bearing fields before selected reports are committed. |
| `benchmarks/corpora/sec_edgar_section_manifest.generated.json` | Committed SEC section-level manifest used by the public SEC evaluation. | Contains labels/queries, not raw SEC filings. |
| `benchmarks/results/retrieval_benchmark_latest.md` | Checked-in synthetic retrieval report. | Synthetic/offline only. |
| `benchmarks/results/llm_routing_benchmark_mock_latest.md` | Checked-in mock LLM routing report. | Mock/synthetic only. |

## Public Corpus Tooling

| Path | Purpose |
|---|---|
| `benchmarks/corpora/public_sources.json` | Source registry for CUAD and SEC EDGAR with usage notes. |
| `benchmarks/corpus_sources/cuad.py` | CUAD metadata/manifest adapter. Does not imply CUAD was evaluated. |
| `benchmarks/corpus_sources/sec_edgar.py` | SEC filing metadata/document acquisition adapter with User-Agent handling. |
| `benchmarks/render_sec_html_to_pdf.py` | Local SEC HTML-to-PDF rendering helper. |
| `benchmarks/generate_synthetic_pdf_corpus.py` | Public-safe synthetic PDF smoke corpus generator. |
| `benchmarks/corpora/README.md` | Runbook for local corpora, manifests, acquisition, preflight, and report promotion. |

## Tests

| Path | Coverage focus |
|---|---|
| `tests/unit/test_llm_routing.py` | Routing, fallback, caching, citations, confidence, accounting, malformed provider payloads. |
| `tests/unit/test_hybrid_retrieval.py` | BM25/hybrid scoring and SEC-aware reranking helpers. |
| `tests/benchmark/test_retrieval_benchmark.py` | Synthetic retrieval fixtures, metrics, report generation, strategy behavior. |
| `tests/benchmark/test_public_corpus_workflow.py` | Public source adapters, preflight, synthetic PDF generation, report sanitization. |
| `tests/benchmark/test_document_rag_eval_metrics.py` | Retrieval metric calculation and report fields for the document RAG harness. |
| `tests/benchmark/test_sec_section_labeler.py` | SEC section label extraction behavior. |
| `tests/benchmark/test_sec_metadata_enrichment.py` | SEC metadata mapping/enrichment behavior. |

## Infrastructure And Monitoring

| Path | Purpose |
|---|---|
| `docker-compose.yml` | Local multi-service stack wiring for Redis, PostgreSQL, MinIO, services, monitoring, MLflow, and local inference dependencies. |
| `requirements*.txt` | Split dependency sets for core/dev/service-specific installs. |
| `Makefile` | Convenience commands for focused tests, benchmarks, corpus validation, acquisition, ingestion, retrieval, answer, and report promotion. |
| `.env.example` | Placeholder-only local configuration template. |
| `.github/workflows/test-and-benchmark-smoke.yml` | Deterministic CI smoke workflow for selected tests and the mock benchmark. |
| `monitoring/prometheus/` | Prometheus scrape and alert configuration. |
| `monitoring/grafana/` | Grafana datasource/dashboard provisioning. |
| `services/monitoring/` | Monitoring service modules, metrics exporter, drift/control-plane scaffolding. |
| `infrastructure/kubernetes/` and `infrastructure/terraform/` | Deployment scaffolding. No live production deployment claim is made from these files. |

## Suggested Reading Order

1. Read `README.md` for the project shape and exact supported claims.
2. Read `docs/architecture.md` for service/data-flow diagrams.
3. Read `PROJECT_EVIDENCE.md` before turning any result into a CV bullet.
4. Inspect `services/inference-api/utils/hybrid_retrieval.py`, `benchmarks/e2e_document_rag_eval.py`, and `benchmarks/sec_metadata_enrichment.py` for the strongest SEC retrieval evidence path.
5. Inspect the sanitized SEC v2 retrieval report to see the exact bounded metric claim and its limitations.
