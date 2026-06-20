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
