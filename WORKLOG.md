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

## Files Changed

- `WORKLOG.md`
- `services/llm-orchestrator/complexity_analyzer.py`
- `services/llm-orchestrator/config.py`
- `services/llm-orchestrator/utils/ModelRouter.py`
- `services/llm-orchestrator/utils/QueryRequest.py`
- `services/llm-orchestrator/utils/__init__.py`

## Tests and Checks Run

- `git status --short`
- `rg --files`
- Read relevant LLM orchestrator and inference API files.
- `python -c "import sys; sys.path.insert(0, 'services/llm-orchestrator'); from complexity_analyzer import QueryComplexityAnalyzer, ComplexityResult; from utils import ModelRouter; from config import Settings; s=Settings(gemini_api_key='test'); r=ModelRouter(s, QueryComplexityAnalyzer()); d=r.route('What is the contract date?', 100, 'legal_contract'); print(type(r.complexity_analyzer.analyze('x','generic')).__name__, d.model, d.reason, d.complexity.score)"`

## Remaining Risks and Limitations

- Routing behavior still needs deterministic tests before public claims can be made.
- Generation fallback, caching, provider response validation, and confidence behavior still need direct unit coverage.
