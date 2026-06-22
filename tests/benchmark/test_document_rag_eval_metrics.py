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
