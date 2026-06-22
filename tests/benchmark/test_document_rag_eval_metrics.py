from types import SimpleNamespace

from benchmarks import e2e_document_rag_eval as eval_harness


def test_document_level_metrics_deduplicate_repeated_chunks():
    query = SimpleNamespace(
        query_id="q_doc",
        query="What is the filing?",
        category="document_label",
        target_document_ids=["manifest_doc_a"],
        relevant_pages=[],
        relevant_chunk_ids=[],
    )
    matches = [
        {
            "id": f"service_doc_a_{idx}",
            "score": 1.0 - idx * 0.01,
            "metadata": {"doc_id": "service_doc_a", "chunk_id": idx, "text": "same document"},
        }
        for idx in range(5)
    ]
    matches.append({
        "id": "service_doc_b_0",
        "score": 0.1,
        "metadata": {"doc_id": "service_doc_b", "chunk_id": 0, "text": "other document"},
    })

    result = eval_harness._evaluate_retrieval_query(
        query,
        matches,
        {"manifest_doc_a": "service_doc_a"},
        top_k=5,
    )

    assert result["metrics_deduplicated_by"] == "document"
    assert result["metric_result_count"] == 2
    assert result["metrics"]["recall@5"] == 1.0
    assert result["metrics"]["mrr"] == 1.0
    assert result["metrics"]["ndcg@5"] == 1.0
    assert all(0.0 <= value <= 1.0 for value in result["metrics"].values())

def test_section_level_metrics_deduplicate_repeated_section_chunks():
    query = SimpleNamespace(
        query_id="q_section",
        query="Where is Item 1A discussed?",
        category="section_label",
        target_document_ids=["manifest_doc_a"],
        relevant_pages=[10, 11, 12],
        relevant_sections=["item_1a_risk_factors"],
        relevant_chunk_ids=[],
    )
    matches = [
        {
            "id": "service_doc_a_1",
            "score": 0.99,
            "metadata": {"doc_id": "service_doc_a", "filename": "doc_a.pdf", "page": 10, "chunk_id": 1, "text": "risk factors"},
        },
        {
            "id": "service_doc_a_2",
            "score": 0.98,
            "metadata": {"doc_id": "service_doc_a", "filename": "doc_a.pdf", "page": 11, "chunk_id": 2, "text": "more risk factors"},
        },
        {
            "id": "service_doc_a_3",
            "score": 0.5,
            "metadata": {"doc_id": "service_doc_a", "filename": "doc_a.pdf", "page": 30, "chunk_id": 3, "text": "management discussion"},
        },
    ]

    result = eval_harness._evaluate_retrieval_query(
        query,
        matches,
        {"manifest_doc_a": "service_doc_a"},
        top_k=5,
        section_labels_by_document={
            "manifest_doc_a": [
                {"section_id": "item_1a_risk_factors", "start_page": 10, "end_page": 12},
                {"section_id": "item_7_mda", "start_page": 30, "end_page": 35},
            ]
        },
        document_identity_lookup={"manifest_doc_a": "manifest_doc_a", "service_doc_a": "manifest_doc_a", "doc_a.pdf": "manifest_doc_a"},
    )

    assert result["label_granularity"] == "section"
    assert result["metrics_deduplicated_by"] == "section"
    assert result["metric_result_count"] == 2
    assert result["metrics"]["recall@1"] == 1.0
    assert result["metrics"]["mrr"] == 1.0
    assert result["top_results"][0]["section_id"] == "item_1a_risk_factors"
