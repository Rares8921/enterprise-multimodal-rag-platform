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
infer_sec_query_metadata = hybrid_module.infer_sec_query_metadata
sec_aware_rerank = hybrid_module.sec_aware_rerank
sec_metadata_score = hybrid_module.sec_metadata_score
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


@pytest.mark.unit
def test_infer_sec_query_metadata_uses_explicit_query_facts():
    inferred = infer_sec_query_metadata(
        "In AAPL 10-K 2025-10-31 0000320193-25-000079, where is Item 7A market risk discussed?"
    )

    assert inferred == {
        "section_id": "item_7a_market_risk",
        "section_name": "Item 7A Quantitative and Qualitative Disclosures About Market Risk",
        "ticker": "AAPL",
        "accession_number": "0000320193-25-000079",
        "filing_year": 2025,
    }


@pytest.mark.unit
def test_sec_metadata_score_boosts_matching_section_and_filing_metadata():
    inferred = {
        "section_id": "item_1a_risk_factors",
        "ticker": "MSFT",
        "accession_number": "0000950170-25-100235",
        "filing_year": 2025,
    }

    score, reasons = sec_metadata_score(inferred, {
        "section_id": "item_1a_risk_factors",
        "ticker": "MSFT",
        "accession_number": "0000950170-25-100235",
        "filing_year": 2025,
        "is_table_of_contents": False,
    })

    assert score > 1.5
    assert {"section_match", "ticker_match", "accession_match", "filing_year_match"}.issubset(reasons)


@pytest.mark.unit
def test_sec_metadata_score_downranks_table_of_contents_and_wrong_ticker():
    inferred = {"section_id": "item_8_financial_statements", "ticker": "NVDA", "filing_year": 2026}

    score, reasons = sec_metadata_score(inferred, {
        "section_id": "unknown",
        "ticker": "AAPL",
        "filing_year": 2025,
        "is_table_of_contents": "true",
    })

    assert score < 0
    assert "table_of_contents_penalty" in reasons
    assert "ticker_mismatch" in reasons
    assert "filing_year_mismatch" in reasons


@pytest.mark.unit
def test_sec_aware_rerank_promotes_matching_section_without_ground_truth_labels():
    matches = [
        {
            "id": "vector-top-wrong-section",
            "score": 0.98,
            "metadata": {
                "text": "Table of contents Item 1A Risk Factors Item 7 Management discussion",
                "section_id": "unknown",
                "ticker": "AAPL",
                "accession_number": "0000320193-25-000079",
                "filing_year": 2025,
                "is_table_of_contents": True,
            },
        },
        {
            "id": "matching-section",
            "score": 0.70,
            "metadata": {
                "text": "Management's Discussion and Analysis results of operations",
                "section_id": "item_7_mda",
                "ticker": "AAPL",
                "accession_number": "0000320193-25-000079",
                "filing_year": 2025,
                "is_table_of_contents": False,
            },
        },
    ]

    ranked = sec_aware_rerank(
        "In AAPL 10-K 2025-10-31 0000320193-25-000079, where is Item 7 Management's Discussion and Analysis discussed?",
        matches,
        metadata_weight=0.5,
    )

    assert ranked[0]["id"] == "matching-section"
    assert ranked[0]["sec_metadata_score"] > ranked[1]["sec_metadata_score"]
    assert "section_match" in ranked[0]["sec_metadata_reasons"]


@pytest.mark.unit
def test_sec_aware_rerank_does_not_penalize_missing_ticker_when_query_has_none():
    matches = [
        {"id": "a", "score": 0.8, "metadata": {"text": "risk factors", "section_id": "item_1a_risk_factors", "ticker": "AAPL"}},
        {"id": "b", "score": 0.7, "metadata": {"text": "risk factors", "section_id": "item_1a_risk_factors", "ticker": "MSFT"}},
    ]

    ranked = sec_aware_rerank("Where are Item 1A Risk Factors discussed?", matches)

    assert all("ticker_mismatch" not in match["sec_metadata_reasons"] for match in ranked)
