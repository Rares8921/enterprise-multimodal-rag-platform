import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = REPO_ROOT / "services" / "llm-orchestrator"

if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from complexity_analyzer import ComplexityResult, QueryComplexityAnalyzer
from config import Settings
from prompt_manager import PromptManager
from utils import ModelChoice, ModelRouter, QueryRequest


@pytest.fixture()
def router():
    settings = Settings(gemini_api_key="test-key", _env_file=None)
    return ModelRouter(settings, QueryComplexityAnalyzer())


class FakeRedis:
    def __init__(self, cached_value=None):
        self.cached_value = cached_value
        self.get_keys = []
        self.setex_calls = []

    async def get(self, key):
        self.get_keys.append(key)
        return self.cached_value

    async def setex(self, key, ttl, value):
        self.setex_calls.append((key, ttl, value))
        self.cached_value = value


class FakeModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def generate(self, prompt, temperature=0.1, max_tokens=1024):
        self.calls.append({
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if self.error:
            raise self.error
        return self.result


@pytest.fixture()
def orchestrator_module(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []
    genai_mod = types.ModuleType("google.generativeai")
    genai_mod.configure = lambda api_key: None
    genai_mod.GenerationConfig = lambda **kwargs: kwargs

    class DummyGenerativeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt, generation_config=None):
            raise AssertionError("Unit tests should replace model globals")

    genai_mod.GenerativeModel = DummyGenerativeModel
    google_pkg.generativeai = genai_mod

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", genai_mod)

    module_name = "llm_orchestrator_main_for_tests"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, SERVICE_DIR / "main.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    module.prompt_manager = PromptManager(str(SERVICE_DIR / "specialized_prompts"))
    return module


@pytest.mark.unit
def test_simple_query_uses_cost_efficient_model(router):
    decision = router.route("What is the invoice date?", context_length=120, doc_type="generic")

    assert decision.model == "mistral"
    assert decision.reason == "cost_efficient"
    assert isinstance(decision.complexity, ComplexityResult)
    assert decision.complexity.level == "low"


@pytest.mark.unit
def test_complex_legal_query_uses_high_performance_model(router):
    query = (
        "Analyze the implications of the indemnification, liability, and breach "
        "clauses if termination depends on arbitration outcomes, and compare the risks."
    )

    decision = router.route(query, context_length=2_000, doc_type="legal_contract")

    assert decision.model == "gemini"
    assert decision.reason == "high_complexity"
    assert decision.complexity.score >= router.settings.complexity_threshold
    assert any(signal.startswith("domain_terms_legal:") for signal in decision.complexity.signals)


@pytest.mark.unit
def test_very_long_query_routes_to_high_performance_model(router):
    query = "Analyze " + " ".join(["liability"] * 40)

    decision = router.route(query, context_length=500, doc_type="legal_contract")

    assert decision.model == "gemini"
    assert decision.complexity.level == "high"
    assert "length:long" in decision.complexity.signals


@pytest.mark.unit
def test_empty_query_complexity_is_explicit(router):
    decision = router.route("", context_length=50, doc_type="generic")

    assert decision.model == "mistral"
    assert decision.complexity.score == 0.0
    assert decision.complexity.signals == ["empty_query"]


@pytest.mark.unit
def test_prompt_manager_selects_legal_and_financial_prompts():
    manager = PromptManager(str(SERVICE_DIR / "specialized_prompts"))

    legal_prompt = manager.get_prompt_template("legal_contract")
    financial_prompt = manager.get_prompt_template("financial_report", agent="extraction")

    assert "Legal Document Analyst" in legal_prompt
    assert "NO LEGAL ADVICE" in legal_prompt
    assert "Financial Report Extractor" in financial_prompt
    assert "Return EXACTLY this JSON schema" in financial_prompt


@pytest.mark.unit
def test_cost_estimation_uses_configured_model_prices(router):
    cost = router.estimate_cost("mistral", input_tokens=1_000, output_tokens=500)

    assert cost == pytest.approx(0.00175)
    assert router.estimate_cost("gemini", input_tokens=-10, output_tokens=-5) == 0.0


@pytest.mark.unit
def test_extract_citations_ignores_out_of_range_markers(orchestrator_module):
    context = [
        {"text": "first chunk", "page": 1, "doc_id": "a"},
        {"text": "second chunk", "page": 2, "doc_id": "b"},
        {"text": "third chunk", "page": 3, "doc_id": "c"},
    ]

    citations = orchestrator_module.extract_citations("Use [1], [3], and [99].", context)

    assert [citation["index"] for citation in citations] == [1, 3]
    assert citations[0]["text"] == "first chunk"
    assert citations[1]["page"] == 3


@pytest.mark.unit
def test_confidence_increases_with_citations_and_drops_with_hedging(orchestrator_module):
    answer = "The agreement states a supported finding. " * 12

    cited = orchestrator_module.calculate_confidence(answer, citations=[{"index": 1}])
    uncited = orchestrator_module.calculate_confidence(answer, citations=[])
    hedged = orchestrator_module.calculate_confidence(
        answer + " Maybe this could be incomplete.",
        citations=[{"index": 1}],
    )

    assert cited > uncited
    assert hedged < cited


@pytest.mark.unit
def test_fallback_when_preferred_provider_fails(orchestrator_module, router):
    orchestrator_module.redis_client = FakeRedis()
    orchestrator_module.model_router = router
    orchestrator_module.gemini_model = FakeModel(error=RuntimeError("provider unavailable"))
    orchestrator_module.mistral_model = FakeModel(result={
        "text": "Fallback answer with evidence [1].",
        "usage": {"input_tokens": 100, "output_tokens": 25},
    })

    request = QueryRequest(
        query="Analyze liability and indemnification implications if termination occurs.",
        context=[{"text": "The contract includes an indemnification clause.", "page": 4, "doc_id": "doc-1"}],
        doc_type="legal_contract",
        tenant_id="tenant-a",
        model_choice=ModelChoice.AUTO,
    )

    response = asyncio.run(orchestrator_module.generate_response(request))

    assert response.model_used == "mistral"
    assert response.answer.startswith("Fallback answer")
    assert response.tokens_used == 125
    assert response.citations[0]["index"] == 1
    assert len(orchestrator_module.gemini_model.calls) == 1
    assert len(orchestrator_module.mistral_model.calls) == 1
    assert orchestrator_module.redis_client.setex_calls


@pytest.mark.unit
def test_cached_response_skips_model_calls(orchestrator_module, router):
    cached_payload = {
        "answer": "Cached answer [1].",
        "model_used": "mistral",
        "citations": [{"index": 1}],
        "confidence_score": 0.8,
        "tokens_used": 12,
        "latency_ms": 1.0,
    }
    orchestrator_module.redis_client = FakeRedis(json.dumps(cached_payload))
    orchestrator_module.model_router = router
    orchestrator_module.gemini_model = FakeModel(result={"text": "should not run", "usage": {}})
    orchestrator_module.mistral_model = FakeModel(result={"text": "should not run", "usage": {}})

    request = QueryRequest(
        query="What is the invoice date?",
        context=[{"text": "Invoice date is Jan 1.", "page": 1}],
        doc_type="generic",
        tenant_id="tenant-a",
        model_choice=ModelChoice.AUTO,
    )

    response = asyncio.run(orchestrator_module.generate_response(request))

    assert response.answer == "Cached answer [1]."
    assert response.model_used == "mistral"
    assert orchestrator_module.gemini_model.calls == []
    assert orchestrator_module.mistral_model.calls == []


@pytest.mark.unit
def test_malformed_provider_response_returns_bad_gateway(orchestrator_module, router):
    orchestrator_module.redis_client = FakeRedis()
    orchestrator_module.model_router = router
    orchestrator_module.gemini_model = FakeModel(result={"text": "unused", "usage": {}})
    orchestrator_module.mistral_model = FakeModel(result={"usage": {"input_tokens": 10}})

    request = QueryRequest(
        query="What is the invoice date?",
        context=[{"text": "Invoice date is Jan 1.", "page": 1}],
        doc_type="generic",
        tenant_id="tenant-a",
        model_choice=ModelChoice.AUTO,
    )

    with pytest.raises(orchestrator_module.HTTPException) as exc:
        asyncio.run(orchestrator_module.generate_response(request))

    assert exc.value.status_code == 502
    assert "Malformed model response" in exc.value.detail
