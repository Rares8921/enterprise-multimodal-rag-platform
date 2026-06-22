# Worklog

## Plan

1. Audit and stabilize the LLM routing subsystem, with emphasis on typed complexity scoring, routing decisions, fallback behavior, token/cost accounting, caching, confidence scoring, prompt selection, error handling, and importability.
2. Add deterministic unit tests for routing, prompts, fallback, caching, confidence, citations, accounting, and malformed responses.
3. Add a reproducible mock/synthetic LLM routing benchmark comparing always-expensive, always-cheap, and heuristic routing strategies.
4. Document the benchmark, architecture, and case-study evidence without unsupported production claims.
5. Strengthen the README so reviewers can run tests and benchmarks and understand supported versus unsupported claims.
6. Improve test and benchmark hygiene with Makefile targets, ignored local artifacts, sample environment configuration, and CI smoke checks if safe.
7. Create `PROJECT_EVIDENCE.md` as the source of truth for CV claims and limitations.

## Audit Notes

- `services/llm-orchestrator/complexity_analyzer.py` returned a dictionary, while `services/llm-orchestrator/utils/ModelRouter.py` compared the result as a float and formatted it with `:.2f`.
- The orchestrator utility package used bare imports in `utils/__init__.py`, which made package imports fragile from the service root and difficult to test.
- The inference API already contains reliability mechanisms such as retries, circuit breaker state, cache TTL selection, context sanitization, context truncation, and response validation.
- Existing local repository state before this work included a modified `.gitignore` and an untracked `PROJECT_CONTEXT.md`; these are treated as pre-existing user changes unless explicitly updated for this task.

## Completed Work

- Created this worklog with the phase plan and initial audit findings.
- Stabilized the LLM routing contract by replacing dictionary complexity results with a typed `ComplexityResult`.
- Added an inspectable `RoutingDecision` while keeping `select_model()` compatible with the existing generation endpoint.
- Fixed orchestrator utility imports so the service package can be imported from the service root.
- Made orchestrator settings compatible with Pydantic v2 and shared multi-service `.env` files.
- Added deterministic unit tests for LLM routing, prompt template selection, fallback, response caching, token/cost accounting, citation extraction, confidence scoring, empty/long queries, and malformed provider responses.
- Fixed model wrapper package exports so the orchestrator can import wrappers from the service root.
- Added model response normalization so malformed provider payloads return a clear 502 instead of an incidental key error.
- Made MLflow setup lazy during orchestrator startup to keep unit imports lightweight.
- Added a reproducible mock/synthetic LLM routing benchmark comparing always-expensive, always-cheap, and heuristic routing strategies.
- Added a fixed mixed workload covering simple factual, medium document QA, complex legal/financial, citation-heavy, long-context, adversarial/ambiguous, and cache-hit cases.
- Added benchmark tests for workload coverage, strategy summaries, output schemas, and JSON/Markdown/CSV report writers.
- Reduced router import coupling so benchmark code can use the router without loading service environment settings.
- Generated checked-in mock benchmark evidence under `benchmarks/results` for commit `f0d1fd0`.
- Added LLM routing benchmark documentation, system architecture documentation, and a portfolio-style case study with explicit limitations.
- Initially documented BM25/hybrid retrieval as unsupported, then added a bounded BM25 reranking implementation and updated the claim ledger.
- Added a reviewer-oriented README with quickstart, architecture summary, required services, environment variables, test commands, benchmark commands, supported claims, unsupported claims, and known limitations.
- Added Makefile targets for focused routing tests, benchmark tests, and the mock LLM routing benchmark.
- Added `.env.example` with placeholder-only local development variables.
- Added a lightweight GitHub Actions smoke workflow that runs deterministic tests and the mock benchmark without real API keys.
- Updated `.gitignore` for local LLM routing benchmark artifacts while preserving the existing `TODO.txt` ignore.
- Added the missing `google-generativeai` runtime dependency used by the Gemini wrapper.
- Added `PROJECT_EVIDENCE.md` mapping supported CV claims to code, tests, benchmark evidence, validation commands, and limitations.
- Added BM25 hybrid reranking over Pinecone vector candidates in the inference API query path.
- Added deterministic unit tests for tokenization, BM25 scoring, hybrid reranking, empty-query behavior, and weight validation.
- Updated README, architecture docs, case study, CI, Makefile, and evidence file to support a bounded hybrid retrieval claim.

## Files Changed

- `WORKLOG.md`
- `README.md`
- `.env.example`
- `.github/workflows/test-and-benchmark-smoke.yml`
- `.gitignore`
- `Makefile`
- `PROJECT_EVIDENCE.md`
- `requirements.txt`
- `services/inference-api/config.py`
- `services/inference-api/main.py`
- `services/inference-api/utils/__init__.py`
- `services/inference-api/utils/context_utility.py`
- `services/inference-api/utils/hybrid_retrieval.py`
- `services/llm-orchestrator/complexity_analyzer.py`
- `services/llm-orchestrator/config.py`
- `services/llm-orchestrator/main.py`
- `services/llm-orchestrator/model_wrapper/__init__.py`
- `services/llm-orchestrator/utils/ModelRouter.py`
- `services/llm-orchestrator/utils/QueryRequest.py`
- `services/llm-orchestrator/utils/__init__.py`
- `benchmarks/llm_routing_benchmark.py`
- `benchmarks/data_samples/llm_routing_workload.json`
- `benchmarks/results/llm_routing_benchmark_mock_latest.json`
- `benchmarks/results/llm_routing_benchmark_mock_latest.md`
- `docs/llm-routing-benchmark.md`
- `docs/architecture.md`
- `docs/case-study.md`
- `tests/benchmark/test_llm_routing_benchmark.py`
- `tests/unit/test_hybrid_retrieval.py`
- `tests/unit/test_llm_routing.py`

## Tests and Checks Run

- `git status --short`
- `rg --files`
- Read relevant LLM orchestrator and inference API files.
- `python -c "import sys; sys.path.insert(0, 'services/llm-orchestrator'); from complexity_analyzer import QueryComplexityAnalyzer, ComplexityResult; from utils import ModelRouter; from config import Settings; s=Settings(gemini_api_key='test'); r=ModelRouter(s, QueryComplexityAnalyzer()); d=r.route('What is the contract date?', 100, 'legal_contract'); print(type(r.complexity_analyzer.analyze('x','generic')).__name__, d.model, d.reason, d.complexity.score)"`
- `pytest tests\unit\test_llm_routing.py -q` - 11 passed.
- `pytest tests\benchmark\test_llm_routing_benchmark.py -q` - 3 passed.
- `python benchmarks\llm_routing_benchmark.py --output-dir $env:TEMP\llm-routing-benchmark --run-id smoke` - wrote JSON, Markdown, and CSV reports to a temporary directory.
- `python benchmarks\llm_routing_benchmark.py --output-dir benchmarks\results --run-id mock_latest` - wrote checked-in JSON/Markdown benchmark evidence and an ignored CSV.
- `python -c "import json; data=json.load(open('benchmarks/results/llm_routing_benchmark_mock_latest.json')); print(data['git_commit']); [print(k, v['summary']['estimated_total_cost_usd'], v['summary']['latency_ms'], v['summary']['cache_hit_rate'], v['summary']['fallback_count'], v['summary']['quality_proxy']) for k,v in data['strategies'].items()]"`
- `python -m pytest tests\unit\test_llm_routing.py tests\benchmark\test_llm_routing_benchmark.py -q` - 14 passed.
- `python benchmarks\llm_routing_benchmark.py --output-dir $env:TEMP\llm-routing-benchmark --run-id hygiene` - wrote JSON, Markdown, and CSV reports to a temporary directory.
- `python -m pytest tests\unit\test_hybrid_retrieval.py -q` - 5 passed.
- `python -m pytest tests\unit\test_llm_routing.py tests\unit\test_hybrid_retrieval.py tests\benchmark\test_llm_routing_benchmark.py -q` - 19 passed.

## Remaining Risks and Limitations

- The unit tests use mocked providers and cache; they do not prove external provider availability or answer quality.
- The benchmark is explicitly mock/synthetic; it estimates latency and cost and must not be described as production performance or real model quality.
- Documentation is not a substitute for real provider benchmarks, real retrieval evaluation, or production deployment evidence.
- CI is a smoke workflow only; it does not run the full Docker Compose stack or integration tests.
- Hybrid retrieval is implemented as BM25 reranking over vector candidates; there is still no labeled retrieval-quality benchmark.
- `PROJECT_EVIDENCE.md` marks real provider performance, production usage, and compliance/security claims as unsupported.

## Retrieval Benchmark Plan

### Current Retrieval Flow

- `services/embedding/main.py` reads OCR and layout data from Redis, chunks document text with layout metadata, generates sentence-transformer embeddings, and upserts chunk vectors into Pinecone with tenant, document, page, type, filename, and text metadata.
- `services/inference-api/main.py` encodes the query, requests top-k vector candidates from Pinecone, reranks those vector candidates with `hybrid_rerank()`, filters by the original vector score threshold, and builds LLM context with sanitized text plus `score`, `vector_score`, and `bm25_score`.
- `services/inference-api/utils/hybrid_retrieval.py` provides deterministic tokenization, BM25 scoring, score normalization, and vector/BM25 weighted reranking.

### Supported Retrieval Claim

- The project supports vector retrieval over Pinecone-indexed chunks plus BM25 reranking over that candidate pool.
- This is a bounded hybrid retrieval claim, not a claim about a separate first-stage BM25 index.

### Benchmark Gap

- There is no labeled retrieval-quality benchmark yet.
- Current tests prove ranking mechanics for small hand-built examples, but they do not measure Recall@k, MRR, or nDCG over a labeled query/chunk fixture set.
- No production Pinecone retrieval quality, legal correctness, financial correctness, customer data behavior, or production accuracy should be claimed.

### Plan

1. Add synthetic labeled legal/financial retrieval fixtures with document/chunk metadata and relevant chunk IDs per query.
2. Implement an offline retrieval benchmark comparing vector-only, BM25-only, hybrid, and simple score-weight ablations.
3. Add deterministic tests for fixture schema, metrics, ranking output, weight validation, report generation, and strategy-specific behavior.
4. Generate checked-in JSON/Markdown benchmark evidence labeled as synthetic and offline.
5. Update docs, README, and `PROJECT_EVIDENCE.md` with bounded retrieval benchmark claims and limitations.

### Fixture Progress

- Added synthetic offline retrieval document fixtures with 10 documents and 40 chunks spanning legal contracts, financial reports, and distractor policy/memo content.
- Added 15 labeled synthetic retrieval queries across exact lexical, paraphrase/semantic, numeric financial, legal clause, citation-oriented, ambiguous, distractor-heavy, BM25-helpful, and vector-helpful categories.
- Validated that every labeled relevant chunk ID exists in the chunk fixture.

### Benchmark Implementation Progress

- Added `benchmarks/retrieval_benchmark.py`, an offline synthetic retrieval-quality benchmark that compares vector-only, BM25-only reranking over the same candidate pool, and hybrid vector/BM25 score-weight ablations.
- The vector path uses a deterministic `semantic_terms` bag-of-words cosine simulator; the benchmark does not call Pinecone, external embeddings, or production services.
- Metrics include Recall@1, Recall@3, Recall@5, MRR, nDCG@5, category summaries, per-query top results, top-5 misses, and candidate-pool miss counts.
- Added a controlled labeled query where exact numeric terms allow BM25 to improve over the semantic proxy, while existing ambiguous/citation queries show semantic proxy behavior where BM25 alone ranks worse.
- Validation run: `python benchmarks\retrieval_benchmark.py --output-dir $env:TEMP\retrieval-benchmark --run-id implementation_check`.

### Retrieval Benchmark Test Progress

- Added deterministic tests for retrieval fixture schema validation, metric calculation, ranking output shape, strategy/candidate-pool validation, report generation, BM25-improves behavior, and semantic-proxy-improves behavior.

### Retrieval Benchmark Evidence Progress

- Generated checked-in synthetic offline retrieval benchmark evidence under `benchmarks/results/retrieval_benchmark_latest.json` and `benchmarks/results/retrieval_benchmark_latest.md`.
- Current run used 40 chunks, 15 labeled synthetic queries, a candidate pool of 25, and git commit `61fedd9`.
- Overall results from the generated report: vector-only Recall@1 0.9000 / MRR 0.9667 / nDCG@5 0.9754; BM25-only Recall@1 0.8667 / MRR 0.9222 / nDCG@5 0.9468; hybrid 70/30 and 50/50 Recall@1 0.9667 / MRR 1.0000 / nDCG@5 1.0000; hybrid 30/70 Recall@1 0.9667 / MRR 1.0000 / nDCG@5 0.9946.
- Limitations remain: the report is synthetic/offline, uses simulated vector scores, and does not measure production Pinecone behavior, real legal/financial correctness, real latency, QPS, or customer data behavior.

### Retrieval Benchmark Documentation Progress

- Updated README, architecture docs, case study, and `PROJECT_EVIDENCE.md` to document the retrieval benchmark methodology, exact checked-in results, supported CV claim, and unsupported production-quality claims.

## Document RAG Corpus Evaluation Plan

### Corpus Structure And Manifest Progress

- Added a curated corpus workspace under `benchmarks/corpora/` with ignored `local_pdfs/` and `results/` directories for private PDFs and generated local reports.
- Added `benchmarks/corpora/example_manifest.json` to document the manifest shape without committing raw PDFs.
- Added `benchmarks/corpus_manifest.py` with schema-level validation for corpus identity, corpus mode, document metadata, relative filenames, query labels, relevant pages/chunks, expected answer hints, and citation requirements.
- Current supported claim remains harness-only: the project can define and validate curated PDF corpus manifests. No PDF run, Pinecone retrieval result, answer quality, production usage, or legal/financial correctness is claimed yet.

### Corpus Loader Validation Progress

- Extended `benchmarks/corpus_manifest.py` with a local corpus loader that resolves PDFs under a configured local PDF root, checks missing files, rejects path traversal, warns for private/local manifests, and rejects `allowed_to_commit=true` in `private_local` corpora.
- Added deterministic tests for valid manifests, missing PDFs, duplicate document IDs, missing query targets, private/local warnings, commit-safety validation, and bad filenames.
- Validation run: `python -m pytest tests\benchmark\test_corpus_manifest.py -q` - 7 passed.

### Document RAG Ingestion Runner Progress

- Added `benchmarks/e2e_document_rag_eval.py` with `validate-only` and `ingest` modes.
- `validate-only` validates a manifest and local file references without service calls.
- `ingest` uploads supported PDF document types to the configured ingestion API, records per-document service IDs/statuses, can poll document status, and writes a JSON run artifact.
- Service URLs, tenant ID, API key, and bearer token are provided by CLI arguments or environment variables; no secrets are hardcoded.
- Validation run: `python benchmarks\e2e_document_rag_eval.py validate-only --manifest benchmarks\corpora\example_manifest.json --skip-file-check --output-dir $env:TEMP\document-rag-eval --run-id smoke`.
- Ingestion was not executed because no local PDFs or live ingestion service are committed with the repository.

### Pinecone Retrieval Evaluation Mode Progress

- Extended `benchmarks/e2e_document_rag_eval.py` with `retrieve` mode for local real-service retrieval evaluation against a configured Pinecone index and namespace.
- Retrieval mode loads the same sentence-transformer embedding model family, queries Pinecone directly, applies the existing BM25 hybrid reranker over vector candidates, and computes Recall@1, Recall@3, Recall@5, MRR, and nDCG@5.
- Query labels can be evaluated at chunk, page, or document granularity. Ingestion run artifacts can map manifest document IDs to service document IDs.
- Reports include per-query top results, category metrics, label granularity counts, top-5 misses, and candidate-pool miss counts.
- Validation run: `python -m py_compile benchmarks\e2e_document_rag_eval.py`; `validate-only` smoke; expected missing-Pinecone-key retrieve failure.
- Retrieval was not executed against Pinecone because no live Pinecone credentials/index are committed with the repository.

### Optional Answer Evaluation Mode Progress

- Extended `benchmarks/e2e_document_rag_eval.py` with optional `answer` mode that calls the configured query endpoint only when explicitly requested.
- Answer mode records non-empty answer rate, citation presence for citation-required queries, expected-hint overlap, selected model, confidence, latency, token fields when returned, failures, and per-query proxy details.
- The mode supports ingestion-run mappings so manifest document IDs can be translated to service document IDs for filtered query calls.
- Validation run: `python -m py_compile benchmarks\e2e_document_rag_eval.py`; answer smoke against `http://127.0.0.1:9` wrote a report with 2 expected service-unavailable failures and no provider calls.
- Answer proxy metrics remain lightweight and must not be described as semantic correctness, legal correctness, financial correctness, or provider accuracy.

### Document RAG Report Generation Progress

- Extended `benchmarks/e2e_document_rag_eval.py` so each run writes JSON and Markdown reports by default, with optional CSV output via `--write-csv`.
- Reports include command, timestamp, git commit, environment summary, manifest path, corpus counts, services used without secrets, mode-specific metrics, per-query rows when available, limitations, and unsupported claims.
- Validation run: `python -m py_compile benchmarks\e2e_document_rag_eval.py`; validate-only JSON/Markdown/CSV smoke; answer JSON/Markdown smoke against an unreachable local query endpoint.
- At that earlier phase, the document RAG report-generation work was validated only with smoke runs because no curated PDFs, live services, Pinecone credentials, or provider credentials were committed with the repository.

### Document RAG Documentation Progress

- Updated `README.md`, `docs/architecture.md`, `docs/case-study.md`, and `PROJECT_EVIDENCE.md` to document the real-service document RAG evaluation harness, local corpus workflow, exact commands, supported harness claim, and unsupported production-quality claims.
- Files changed in this step: `README.md`, `docs/architecture.md`, `docs/case-study.md`, `PROJECT_EVIDENCE.md`, and `WORKLOG.md`.
- Validation run: `python -m pytest tests\benchmark\test_corpus_manifest.py -q` - 7 passed; `python -m py_compile benchmarks\e2e_document_rag_eval.py`; `python benchmarks\e2e_document_rag_eval.py validate-only --manifest benchmarks\corpora\example_manifest.json --skip-file-check --output-dir $env:TEMP\document-rag-eval --run-id docs_check`.
- Remaining limitation: no real PDF corpus, live ingestion service, Pinecone index, or provider-backed answer report is committed with the repository.

## Public Corpus Readiness Plan

### Audit Progress

- Inspected the document RAG harness, corpus manifest schema, corpus README, `.gitignore`, Makefile, README, architecture docs, case study, evidence ledger, and existing worklog entries.
- Found and restored an accidental local corpus README rename from `benchmarks/corpora/README_corpora.md` back to the tracked `benchmarks/corpora/README.md`; no content change was needed.
- Current evidence language remains bounded: synthetic/offline retrieval evidence exists, the real-service harness exists, and no real public PDF/Pinecone result is claimed.
- Existing gap: Makefile has no corpus workflow targets yet, and the project has no public source registry, acquisition adapters, preflight checks, synthetic PDF generator, or report promotion tooling.
- Remaining risk: raw downloaded corpora and local reports must stay ignored unless a specific sanitized artifact is intentionally promoted.

### Public Source Registry Progress

- Added `benchmarks/corpora/public_sources.json` with public metadata for CUAD / Atticus Project legal contracts and SEC EDGAR financial filings.
- Added `benchmarks/public_corpus_sources.py` to load and validate source metadata, expose summaries, and provide manifest-ready source attribution fields.
- Updated `benchmarks/corpora/README.md` to clarify that the registry is not a downloaded corpus and raw public files still stay in ignored local storage by default.
- Validation run: `python -m py_compile benchmarks\public_corpus_sources.py`; `python -c "from benchmarks.public_corpus_sources import load_public_source_registry; r=load_public_source_registry(); print(r.summary())"`.
- Remaining limitation: this step adds source metadata only; no public documents were downloaded, ingested, indexed, or evaluated.

### CUAD Acquisition Adapter Progress

- Added `benchmarks/corpus_sources/cuad.py` for CUAD-style metadata parsing, deterministic document selection, manifest generation, optional local PDF copying, and optional explicit-URL PDF downloads into ignored local corpus storage.
- Added `benchmarks/acquire_public_corpus.py` with a `cuad` subcommand for generating `benchmarks/corpora/cuad_manifest.generated.json` and preparing files under `benchmarks/corpora/local_pdfs/cuad/`.
- The adapter preserves CUAD attribution/source metadata in generated manifests and keeps `allowed_to_commit=false` for raw contract PDFs by default.
- Validation run: `python -m py_compile benchmarks\corpus_sources\cuad.py benchmarks\acquire_public_corpus.py`; no-network CUAD smoke with mocked metadata; validate-only smoke against the generated temp manifest.
- Remaining limitation: no CUAD files were downloaded or committed; real CUAD usage still requires local dataset acquisition and license/terms review.

### SEC EDGAR Acquisition Adapter Progress

- Added `benchmarks/corpus_sources/sec_edgar.py` for SEC company metadata, recent filing metadata parsing/fetching, manifest generation, optional primary document download, source format labeling, and conservative request delays.
- Added `benchmarks/corpora/sec_edgar_sample_companies.json` with a small curated ticker/CIK list for AAPL, MSFT, AMZN, NVDA, JPM, and XOM.
- Extended `benchmarks/acquire_public_corpus.py` with a `sec-edgar` subcommand that supports no-network `--filings-json` mode and explicit SEC network mode through `--fetch-metadata`.
- Validation run: `python -m py_compile benchmarks\corpus_sources\sec_edgar.py benchmarks\acquire_public_corpus.py`; no-network SEC smoke with mocked filing metadata; expected missing-`SEC_USER_AGENT` failure for network fetch mode.
- Remaining limitation: no SEC filings were downloaded or committed; HTML filing manifests require the public manifest schema integration step before `validate-only` can accept non-PDF source files.

### Public Manifest Integration Progress

- Extended `benchmarks/corpus_manifest.py` to preserve root/document `source_metadata`, validate `source_format`, and allow public HTML filing source files while keeping path traversal checks and ignored-local-file expectations.
- Updated `benchmarks/e2e_document_rag_eval.py` so ingest mode skips non-PDF source files with a clear conversion/rendering message instead of trying to upload SEC HTML filings.
- Updated `benchmarks/corpora/example_manifest.json` and corpus README to document source metadata and source format fields.
- Updated `benchmarks/corpus_sources/sec_edgar.py` so generated SEC manifests validate against the expanded schema.
- Validation run: `python -m pytest tests\benchmark\test_corpus_manifest.py -q` - 7 passed; `python -m py_compile benchmarks\corpus_manifest.py benchmarks\e2e_document_rag_eval.py benchmarks\corpus_sources\sec_edgar.py`; SEC HTML no-network manifest generation plus validate-only smoke.
- Remaining limitation: HTML filings still require rendering or conversion before ingestion because the existing ingestion API accepts PDFs/images, not SEC HTML.

### Document RAG Preflight Progress

- Extended `benchmarks/e2e_document_rag_eval.py` with `preflight` mode and target-specific readiness checks for validate-only, ingest, retrieve, answer, and acquisition workflows.
- Preflight checks manifest schema, local file references, corpus warnings, output directory writability, tracked corpus artifact safety, ingestion/query endpoint reachability when required, Pinecone configuration for retrieval, embedding model configuration, and SEC User-Agent readiness for SEC acquisition/evaluation contexts.
- Preflight writes JSON and Markdown reports using the existing report writer and exits non-zero when required checks fail.
- Validation run: `python -m py_compile benchmarks\e2e_document_rag_eval.py`; validate-only preflight smoke against the example manifest; expected failing retrieve preflight with missing Pinecone key/index and generated failure report.
- Remaining limitation: preflight does not prove service correctness, Pinecone index contents, provider quality, or production readiness.

### Synthetic PDF Corpus Generator Progress

- Added `benchmarks/generate_synthetic_pdf_corpus.py`, a dependency-free deterministic PDF generator for public-safe smoke testing of the real PDF ingestion workflow.
- Generated and committed `benchmarks/corpora/synthetic_smoke_manifest.json`; the corresponding PDFs are generated under ignored local storage at `benchmarks/corpora/local_pdfs/synthetic_smoke/`.
- The synthetic corpus includes legal-style and financial-style PDFs, document/page-level labels, expected answer hints, and citation-required queries.
- Validation run: `python -m py_compile benchmarks\generate_synthetic_pdf_corpus.py`; temp synthetic generation plus validate-only with file checks; default synthetic generation plus validate-only with file checks; ingest preflight smoke generated an expected service-unavailable failure report when no ingestion service was running.
- Remaining limitation: synthetic PDFs are not real legal/financial documents and cannot support public-document retrieval-quality claims.

### Report Promotion Tooling Progress

- Added `benchmarks/promote_document_rag_report.py` to convert local JSON reports into sanitized public Markdown summaries and optional sanitized JSON summaries.
- The promotion tool redacts local paths, removes content-bearing fields such as raw responses, requests, answers, contexts, document/chunk text previews, and removes query text unless explicitly allowed.
- The tool refuses `private_local` reports unless `--allow-private-summary` is passed, while preserving metrics, corpus mode, service names without secrets, limitations, unsupported claims, and sanitization metadata.
- Validation run: `python -m py_compile benchmarks\promote_document_rag_report.py`; promoted a synthetic validate-only report to temp Markdown/JSON; verified private-local report promotion is refused without explicit allow flag.
- Remaining limitation: report promotion is a review aid, not evidence that a run is public-safe or methodologically strong without human review.

### Public Corpus Workflow Test Progress

- Added `tests/benchmark/test_public_corpus_workflow.py` covering source registry schema, CUAD mocked manifest generation, CUAD manual-required behavior, SEC mocked manifest generation, SEC User-Agent failure, SEC metadata/fetch mode validation, preflight reports, missing file/Pinecone/private warnings, synthetic PDF generation, and report promotion sanitization/refusal behavior.
- Fixed `benchmarks/promote_document_rag_report.py` so sanitization preserves the top-level answer proxy metrics section while still removing raw answer text fields.
- Validation run: `python -m pytest tests\benchmark\test_public_corpus_workflow.py -q` - 12 passed; `python -m py_compile benchmarks\promote_document_rag_report.py tests\benchmark\test_public_corpus_workflow.py`.
- Remaining limitation: tests use mocked metadata and temp PDFs only; they do not perform real CUAD/SEC downloads, Pinecone calls, OCR, or provider calls.

### Public Corpus Documentation Progress

- Added Makefile targets for corpus validation, preflight, synthetic generation, CUAD acquisition, SEC acquisition, ingestion, retrieval, answer proxy evaluation, and report promotion.
- Updated README, corpus runbook, architecture docs, case study, and `PROJECT_EVIDENCE.md` with public corpus acquisition commands, synthetic PDF smoke workflow, preflight commands, report promotion commands, supported claims, and unsupported claims.
- Documentation now distinguishes acquisition/readiness tooling from actual public-corpus evaluation results; no CUAD/SEC retrieval metrics are claimed.
- Validation run: `python -m pytest tests\benchmark\test_public_corpus_workflow.py -q` - 12 passed; `python -m py_compile benchmarks\acquire_public_corpus.py benchmarks\generate_synthetic_pdf_corpus.py benchmarks\promote_document_rag_report.py benchmarks\e2e_document_rag_eval.py benchmarks\corpus_manifest.py benchmarks\public_corpus_sources.py`; synthetic manifest preflight smoke passed. `make -n corpus-generate-synthetic` could not run because `make` is not installed in this Windows environment.
- Remaining limitation: public corpus acquisition code is documented, but no real CUAD/SEC documents or sanitized public-corpus evaluation reports are committed.
## SEC Section-Level Retrieval Evaluation Progress

### Completed Work

- Inspected the prior SEC document-level retrieval report and confirmed the main weakness: document-level labels were too coarse, while rank-1 misses often confused adjacent annual filings or returned broad contents/index pages.
- Added section-aware SEC 10-K labeling support that extracts conservative section/page ranges from rendered public SEC PDFs without storing raw filing text.
- Added `relevant_sections` manifest support and section-level relevance/deduplication in the real-service retrieval evaluator.
- Generated and committed `benchmarks/corpora/sec_edgar_section_manifest.generated.json` with 8 public SEC 10-K filings, 29 section-level queries, page ranges, and confidence markers.
- Ran Pinecone-backed retrieval against namespace `tenant_eval_local` using the existing indexed SEC corpus and promoted a sanitized section-level report.
- Ran a diagnostic candidate-pool-100 ablation. It reduced candidate-pool misses but worsened top-k metrics, so it is recorded as diagnostic rather than an improvement.
- Updated the evidence ledger, README, architecture docs, and case study to reflect the bounded SEC section-level result and its limitations.

### Files Changed

- `benchmarks/sec_section_labeler.py`
- `benchmarks/corpus_manifest.py`
- `benchmarks/e2e_document_rag_eval.py`
- `benchmarks/promote_document_rag_report.py`
- `benchmarks/corpora/sec_edgar_section_manifest.generated.json`
- `benchmarks/corpora/results/sanitized_sec_section_retrieval_summary.md`
- `tests/benchmark/test_sec_section_labeler.py`
- `tests/benchmark/test_document_rag_eval_metrics.py`
- `tests/benchmark/test_corpus_manifest.py`
- `tests/benchmark/test_public_corpus_workflow.py`
- `README.md`
- `docs/architecture.md`
- `docs/case-study.md`
- `PROJECT_EVIDENCE.md`
- `WORKLOG.md`

### Tests and Checks Run

- `python -m py_compile benchmarks\sec_section_labeler.py benchmarks\corpus_manifest.py benchmarks\e2e_document_rag_eval.py`
- `python -m pytest tests\benchmark\test_corpus_manifest.py tests\benchmark\test_document_rag_eval_metrics.py tests\benchmark\test_sec_section_labeler.py -q` - 11 passed.
- `python -m pytest tests\benchmark\test_sec_section_labeler.py -q` - 3 passed after conservative TOC/range refinements.
- `python -m py_compile benchmarks\promote_document_rag_report.py`
- `python -m pytest tests\benchmark\test_public_corpus_workflow.py::test_report_promotion_sanitizes_paths_and_refuses_private tests\benchmark\test_public_corpus_workflow.py::test_report_promotion_preserves_retrieval_granularity_metadata -q` - 2 passed.
- `python benchmarks\sec_section_labeler.py --manifest benchmarks\corpora\sec_edgar_rendered_manifest.generated.json --pdf-root benchmarks\corpora\local_pdfs --labels-out benchmarks\corpora\sec_edgar_section_labels.generated.json --manifest-out benchmarks\corpora\sec_edgar_section_manifest.generated.json --overwrite` - generated 29 section labels/queries.
- `python benchmarks\e2e_document_rag_eval.py validate-only --manifest benchmarks\corpora\sec_edgar_section_manifest.generated.json --pdf-root benchmarks\corpora\local_pdfs --run-id sec_section_validate` - passed.
- `python benchmarks\e2e_document_rag_eval.py retrieve --manifest benchmarks\corpora\sec_edgar_section_manifest.generated.json --pdf-root benchmarks\corpora\local_pdfs --tenant-id tenant_eval_local --pinecone-index $env:PINECONE_INDEX --embedding-model sentence-transformers/all-MiniLM-L6-v2 --ingestion-run benchmarks\corpora\results\document_rag_eval_ingest_sec_ingest.json --run-id sec_section_retrieve` - produced section-level retrieval report.
- `python benchmarks\e2e_document_rag_eval.py retrieve --manifest benchmarks\corpora\sec_edgar_section_manifest.generated.json --pdf-root benchmarks\corpora\local_pdfs --tenant-id tenant_eval_local --pinecone-index $env:PINECONE_INDEX --embedding-model sentence-transformers/all-MiniLM-L6-v2 --ingestion-run benchmarks\corpora\results\document_rag_eval_ingest_sec_ingest.json --retrieval-candidate-pool 100 --run-id sec_section_retrieve_pool100` - diagnostic ablation.
- `python benchmarks\promote_document_rag_report.py benchmarks\corpora\results\document_rag_eval_retrieve_sec_section_retrieve.json --output-md benchmarks\corpora\results\sanitized_sec_section_retrieval_summary.md` - promoted sanitized evidence.
- Manual sanitized-report scan for local paths, secrets, and raw text markers.

### Results

- Previous document-level SEC report: Recall@1 `0.5000`, Recall@3 `1.0000`, Recall@5 `1.0000`, MRR `0.7500`, nDCG@5 `0.8155` over 16 document-level queries.
- New section-level SEC report: Recall@1 `0.1034`, Recall@3 `0.2759`, Recall@5 `0.3448`, MRR `0.1879`, nDCG@5 `0.2269` over 29 section-level queries.
- Candidate-pool misses: `13` of 29 for the pool-25 run.
- Pool-100 diagnostic: Recall@1 `0.0345`, Recall@3 `0.2414`, Recall@5 `0.3103`, MRR `0.1477`, nDCG@5 `0.1887`, candidate-pool misses `6`.

### Remaining Risks and Limitations

- Section labels are approximate and generated from rendered public SEC PDF text with conservative heading extraction; uncertain ranges are single-page medium-confidence labels.
- Metrics are section-level only. No chunk-level labels, answer correctness labels, legal correctness, or financial correctness are claimed.
- The run is local real-service evidence over a small public SEC corpus, not production retrieval quality.
- Top-k section retrieval is weak; contents/index pages and adjacent-year filings remain major failure modes.
- Raw SEC HTML/PDF files, raw JSON reports, model cache, and local generated labels remain local/ignored unless intentionally sanitized and promoted.
## SEC Section Retrieval Improvement Diagnosis

### Failure Analysis

- Current section-level baseline remains: Recall@1 `0.1034`, Recall@3 `0.2759`, Recall@5 `0.3448`, MRR `0.1879`, nDCG@5 `0.2269`, candidate-pool misses `13` of 29.
- Rank-1 misses: `26` of 29.
- Candidate-pool misses: `13` of 29.
- Same target document but wrong/unknown section at rank 1: `13` queries.
- Adjacent-year or same-ticker wrong document at rank 1: `11` queries.
- Wrong-company rank-1 top result: `2` queries.
- Top result looked contents-like or mapped to no section for `19` rank-1 misses.
- Weakest categories by Recall@5: Item 7 MD&A (`0.0000`), Item 7A market risk (`0.0000`), Item 8 financial statements (`0.2000`), Item 1 business (`0.2500`).

### Metadata Gap

- Live Pinecone metadata in namespace `tenant_eval_local` contains only `chunk_id`, `doc_id`, `doc_type`, `filename`, `page`, `tenant_id`, `text`, and `type`.
- Current chunks do not contain `section_id`, `section_name`, `section_confidence`, `ticker`, `company_name`, `filing_date`, `form_type`, `accession_number`, `page_number`, or manifest-level `document_id`.
- This makes section-aware reranking impossible from indexed metadata alone and explains why contents/index chunks and adjacent-year filings survive reranking.

### Planned Improvement

- Preserve the old namespace and create `tenant_eval_sec_sections_v2` for enriched metadata.
- Enrich existing indexed vectors with public SEC manifest metadata and conservative section labels instead of re-acquiring documents or overwriting the baseline.
- Add SEC-aware reranking that uses only query-visible facts and indexed metadata: section synonyms, explicit ticker/company, explicit filing date/year/accession, and table-of-contents downranking.
- Run ablations against the same section manifest and report all results, including regressions.


## SEC-Aware Retrieval Reranking

### Completed Work

- Added opt-in SEC-aware reranking that infers only query-visible facts: SEC section, ticker, accession number, and filing year.
- Added metadata scoring for matching section/ticker/accession/year, plus conservative penalties for table-of-contents chunks, unknown sections, and explicit wrong ticker/year/section metadata.
- Wired the real-service retrieval evaluator with `--sec-aware-rerank` and `--sec-metadata-weight` flags while preserving the original BM25 hybrid reranker as the baseline.
- Added retrieval diagnostics to top-result rows for `sec_aware_score`, `sec_metadata_score`, indexed section/ticker/accession/year, and table-of-contents markers.
- Added deterministic unit coverage for SEC query inference, metadata scoring, and reranking behavior.

### Files Changed

- `services/inference-api/utils/hybrid_retrieval.py`
- `benchmarks/e2e_document_rag_eval.py`
- `tests/unit/test_hybrid_retrieval.py`
- `WORKLOG.md`

### Tests and Checks Run

- `python -m py_compile services\inference-api\utils\hybrid_retrieval.py benchmarks\e2e_document_rag_eval.py`
- `python -m pytest tests\unit\test_hybrid_retrieval.py tests\benchmark\test_document_rag_eval_metrics.py -q` - 12 passed.

### Remaining Risks and Limitations

- The reranker uses SEC 10-K metadata and should be treated as a benchmark/reranking improvement, not a general legal or financial correctness signal.
- It can only help when relevant chunks are present in the vector candidate pool.
- The next step is to run the planned Pinecone ablations before claiming any metric improvement.


## SEC Section Retrieval V2 Ablations

### Completed Work

- Verified enriched Pinecone namespace `tenant_eval_sec_sections_v2` contains `1,343` vectors.
- Confirmed sampled vector metadata includes `section_id`, `section_name`, `ticker`, `company_name`, `filing_date`, `filing_year`, `form_type`, `accession_number`, `page_number`, `document_id`, and table-of-contents markers.
- Ran five Pinecone-backed section-level retrieval ablations over the same 8 public SEC 10-K filings and 29 section-level queries.
- Promoted the best honest run to `benchmarks/corpora/results/sanitized_sec_section_retrieval_v2_summary.md` and added a sanitized baseline comparison table.
- Updated README, architecture docs, case study, and project evidence to reflect the improved but still bounded result.

### Files Changed

- `benchmarks/corpora/results/sanitized_sec_section_retrieval_v2_summary.md`
- `PROJECT_EVIDENCE.md`
- `README.md`
- `docs/architecture.md`
- `docs/case-study.md`
- `WORKLOG.md`

### Tests and Checks Run

- Pinecone metadata verification for namespace `tenant_eval_sec_sections_v2`.
- `python benchmarks\e2e_document_rag_eval.py retrieve ... --run-id sec_section_retrieve_v2_baseline`
- `python benchmarks\e2e_document_rag_eval.py retrieve ... --pinecone-namespace tenant_eval_sec_sections_v2 --run-id sec_section_retrieve_v2_metadata`
- `python benchmarks\e2e_document_rag_eval.py retrieve ... --pinecone-namespace tenant_eval_sec_sections_v2 --sec-aware-rerank --run-id sec_section_retrieve_v2_rerank`
- `python benchmarks\e2e_document_rag_eval.py retrieve ... --pinecone-namespace tenant_eval_sec_sections_v2 --sec-aware-rerank --retrieval-candidate-pool 50 --run-id sec_section_retrieve_v2_pool50`
- `python benchmarks\e2e_document_rag_eval.py retrieve ... --pinecone-namespace tenant_eval_sec_sections_v2 --sec-aware-rerank --retrieval-candidate-pool 100 --run-id sec_section_retrieve_v2_pool100`
- `python benchmarks\promote_document_rag_report.py benchmarks\corpora\results\document_rag_eval_retrieve_sec_section_retrieve_v2_pool100.json --output-md benchmarks\corpora\results\sanitized_sec_section_retrieval_v2_summary.md`

### Results

| Run | Namespace | Candidate Pool | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | Candidate Misses |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Previous section baseline | `tenant_eval_local` | 25 | 0.1034 | 0.2759 | 0.3448 | 0.1879 | 0.2269 | 13 |
| V2 old namespace baseline | `tenant_eval_local` | 25 | 0.1034 | 0.2759 | 0.3448 | 0.1879 | 0.2269 | 13 |
| V2 enriched metadata only | `tenant_eval_sec_sections_v2` | 25 | 0.1034 | 0.2759 | 0.3448 | 0.1879 | 0.2269 | 13 |
| V2 SEC-aware rerank | `tenant_eval_sec_sections_v2` | 25 | 0.5172 | 0.5517 | 0.5517 | 0.5287 | 0.5345 | 13 |
| V2 SEC-aware rerank | `tenant_eval_sec_sections_v2` | 50 | 0.6897 | 0.7241 | 0.7241 | 0.7069 | 0.7114 | 8 |
| V2 SEC-aware rerank | `tenant_eval_sec_sections_v2` | 100 | 0.7586 | 0.7931 | 0.7931 | 0.7759 | 0.7804 | 6 |

- Best honest run: SEC-aware reranking over `tenant_eval_sec_sections_v2` with candidate pool `100`.
- Per-query comparison against the previous section baseline: 13 queries improved at Recall@5, 0 regressed, and 6 remaining Recall@5 misses were candidate-pool misses.

### Remaining Risks and Limitations

- Metrics are section-level only; there are no chunk-level labels or answer correctness labels.
- The SEC-aware reranker uses query-visible metadata and indexed metadata for this public SEC corpus, not production-only signals.
- Remaining misses are candidate-pool misses, so reranking alone cannot solve them.
- This is local Pinecone-backed public-corpus evidence, not production retrieval quality.


## SEC Answer Evaluation Readiness

### Completed Work

- Inspected the answer-mode harness, query API, LLM orchestrator, SEC section manifest, v2 retrieval evidence, and environment variable names without printing secret values.
- Confirmed the SEC section manifest has 29 citation-required queries and 29 expected answer-hint lists.
- Validated answer-mode plumbing for passing retrieval candidate pool and SEC-aware reranking settings into the query API.
- Validated inference API support for returning retrieval strategy, candidate pool, and SEC reranking metadata in query responses.

### Files Changed

- `benchmarks/e2e_document_rag_eval.py`
- `services/inference-api/QueryModel.py`
- `services/inference-api/main.py`
- `WORKLOG.md`

### Tests and Checks Run

- `python -m py_compile benchmarks\e2e_document_rag_eval.py services\inference-api\QueryModel.py services\inference-api\main.py`
- `python -m pytest tests\benchmark\test_document_rag_eval_metrics.py tests\unit\test_hybrid_retrieval.py -q` - 12 passed.
- `docker compose ps` - Redis, PostgreSQL, and MinIO were running; inference API, LLM orchestrator, and local Mistral endpoint were not listening yet.

### Remaining Risks and Limitations

- Answer evaluation still requires live query and LLM orchestration services.
- The local Mistral-compatible endpoint was unavailable; answer evaluation should force Gemini if Gemini is configured and reachable.
- The query API uses the authenticated/request tenant ID as the Pinecone namespace, so the SEC v2 answer run should use tenant `tenant_eval_sec_sections_v2`.
