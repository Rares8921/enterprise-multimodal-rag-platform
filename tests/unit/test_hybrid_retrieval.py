import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = REPO_ROOT / "services" / "inference-api"

MODULE_NAME = "inference_api_hybrid_retrieval_for_tests"
spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    INFERENCE_DIR / "utils" / "hybrid_retrieval.py",
)
hybrid_module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = hybrid_module
spec.loader.exec_module(hybrid_module)

bm25_scores = hybrid_module.bm25_scores
hybrid_rerank = hybrid_module.hybrid_rerank
tokenize = hybrid_module.tokenize


@pytest.mark.unit
def test_tokenize_normalizes_text_for_bm25():
    assert tokenize("Revenue, revenue-growth 2024!") == ["revenue", "revenue", "growth", "2024"]


@pytest.mark.unit
def test_bm25_scores_reward_lexical_overlap():
    scores = bm25_scores("termination notice", [
        "termination requires thirty days notice",
        "invoice total and payment date",
    ])

    assert scores[0] > scores[1]


@pytest.mark.unit
def test_hybrid_rerank_can_promote_lexically_relevant_candidate():
    matches = [
        {
            "id": "vector-top",
            "score": 0.95,
            "metadata": {"text": "invoice amount and payment reference"},
        },
        {
            "id": "lexical-hit",
            "score": 0.70,
            "metadata": {"text": "termination notice period and contract termination clause"},
        },
    ]

    ranked = hybrid_rerank("termination notice", matches, vector_weight=0.2, bm25_weight=0.8)

    assert ranked[0]["id"] == "lexical-hit"
    assert ranked[0]["bm25_score"] > ranked[1]["bm25_score"]
    assert "hybrid_score" in ranked[0]
    assert ranked[0]["vector_score"] == 0.70


@pytest.mark.unit
def test_hybrid_rerank_preserves_vector_order_when_query_has_no_terms():
    matches = [
        {"id": "a", "score": 0.4, "metadata": {"text": "alpha"}},
        {"id": "b", "score": 0.9, "metadata": {"text": "beta"}},
    ]

    ranked = hybrid_rerank("", matches)

    assert [match["id"] for match in ranked] == ["b", "a"]
    assert all(match["bm25_score"] == 0.0 for match in ranked)


@pytest.mark.unit
def test_hybrid_rerank_validates_weights():
    with pytest.raises(ValueError):
        hybrid_rerank("query", [{"score": 1.0, "metadata": {"text": "query"}}], vector_weight=0, bm25_weight=0)

    with pytest.raises(ValueError):
        hybrid_rerank("query", [{"score": 1.0, "metadata": {"text": "query"}}], vector_weight=-1)
