import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_PATH = REPO_ROOT / "services" / "llm-orchestrator" / "model_wrapper" / "GeminiLLM.py"

MODULE_NAME = "gemini_model_wrapper_for_tests"
spec = importlib.util.spec_from_file_location(MODULE_NAME, WRAPPER_PATH)
gemini_module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = gemini_module
spec.loader.exec_module(gemini_module)


@pytest.mark.unit
def test_gemini_wrapper_uses_configured_model_name(monkeypatch):
    configured = {}

    def fake_configure(*, api_key):
        configured["api_key"] = api_key

    class FakeGenerativeModel:
        def __init__(self, model_name):
            configured["model_name"] = model_name

    monkeypatch.setattr(gemini_module.genai, "configure", fake_configure)
    monkeypatch.setattr(gemini_module.genai, "GenerativeModel", FakeGenerativeModel)

    wrapper = gemini_module.GeminiLLM("test-key", model_name="gemini-2.5-flash")

    assert configured == {"api_key": "test-key", "model_name": "gemini-2.5-flash"}
    assert wrapper.model_name == "gemini-2.5-flash"
