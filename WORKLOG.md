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
- No checked-in real-service report was generated because no curated PDFs, live services, Pinecone credentials, or provider credentials are committed with the repository.

### Document RAG Documentation Progress

- Updated `README.md`, `docs/architecture.md`, `docs/case-study.md`, and `PROJECT_EVIDENCE.md` to document the real-service document RAG evaluation harness, local corpus workflow, exact commands, supported harness claim, and unsupported production-quality claims.
- Files changed in this step: `README.md`, `docs/architecture.md`, `docs/case-study.md`, `PROJECT_EVIDENCE.md`, and `WORKLOG.md`.
- Validation run: `python -m pytest tests\benchmark\test_corpus_manifest.py -q` - 7 passed; `python -m py_compile benchmarks\e2e_document_rag_eval.py`; `python benchmarks\e2e_document_rag_eval.py validate-only --manifest benchmarks\corpora\example_manifest.json --skip-file-check --output-dir $env:TEMP\document-rag-eval --run-id docs_check`.
- Remaining limitation: no real PDF corpus, live ingestion service, Pinecone index, or provider-backed answer report is committed with the repository.
