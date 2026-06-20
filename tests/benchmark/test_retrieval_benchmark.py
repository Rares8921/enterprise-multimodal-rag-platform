import json
import math

import pytest

from benchmarks import retrieval_benchmark as rb


@pytest.fixture()
def retrieval_chunks():
    documents = rb.read_json(rb.DOCUMENTS_PATH)
    return rb.flatten_chunks(documents)


@pytest.fixture()
def retrieval_queries(retrieval_chunks):
    queries = rb.read_json(rb.QUERIES_PATH)
    chunk_ids = {chunk["chunk_id"] for chunk in retrieval_chunks}
    return rb.validate_queries(queries, chunk_ids)


def _query_by_id(queries, query_id):
    return next(query for query in queries if query["query_id"] == query_id)


def _rank_with_metrics(query, chunks, strategy):
    result = rb.rank_query(query, chunks, strategy, candidate_pool_size=25)
    result["metrics"] = rb.metrics_for_result(result)
    return result


@pytest.mark.benchmark
def test_retrieval_fixtures_validate_schema(retrieval_chunks, retrieval_queries):
    categories = {query["category"] for query in retrieval_queries}

    assert len(retrieval_chunks) == 40
    assert len(retrieval_queries) == 15
    assert {
        "exact_lexical_match",
        "paraphrase_semantic_match",
        "numeric_financial_query",
        "legal_clause_query",
        "citation_oriented_query",
        "ambiguous_query",
        "distractor_heavy_query",
        "lexical_bm25_should_help",
        "vector_semantic_should_help",
    }.issubset(categories)
    assert all(query["relevant_chunk_ids"] for query in retrieval_queries)


@pytest.mark.benchmark
def test_retrieval_metric_calculations_are_deterministic():
    ranked = ["miss_1", "hit_1", "miss_2", "hit_2", "miss_3"]
    relevant = {"hit_1", "hit_2"}

    assert rb.recall_at_k(ranked, relevant, 1) == 0.0
    assert rb.recall_at_k(ranked, relevant, 3) == 0.5
    assert rb.recall_at_k(ranked, relevant, 5) == 1.0
    assert rb.reciprocal_rank(ranked, relevant) == 0.5

    expected_ndcg = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert rb.ndcg_at_k(ranked, relevant, 5) == pytest.approx(expected_ndcg)


@pytest.mark.benchmark
def test_retrieval_ranking_output_shape(retrieval_chunks, retrieval_queries):
    query = _query_by_id(retrieval_queries, "rq015")
    result = _rank_with_metrics(query, retrieval_chunks, rb.Strategy("hybrid_test", 0.5, 0.5))

    assert result["query_id"] == "rq015"
    assert result["candidate_pool_size"] == 25
    assert result["candidate_pool_contains_relevant"] is True
    assert len(result["top_results"]) == 5
    assert {"recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"} == set(result["metrics"])

    top_result = result["top_results"][0]
    assert {
        "chunk_id",
        "doc_id",
        "doc_type",
        "section",
        "vector_score",
        "bm25_score",
        "score",
        "text",
        "is_relevant",
    }.issubset(top_result)


@pytest.mark.benchmark
def test_retrieval_strategy_and_candidate_pool_validation(retrieval_chunks, retrieval_queries):
    query = retrieval_queries[0]

    with pytest.raises(ValueError, match="non-negative"):
        rb.validate_strategy(rb.Strategy("bad_negative", -0.1, 1.0))

    with pytest.raises(ValueError, match="positive"):
        rb.validate_strategy(rb.Strategy("bad_zero", 0.0, 0.0))

    with pytest.raises(ValueError, match="candidate_pool_size"):
        rb.rank_query(query, retrieval_chunks, rb.Strategy("valid", 0.5, 0.5), candidate_pool_size=0)


@pytest.mark.benchmark
def test_retrieval_benchmark_writes_reports(temp_dir):
    base_report = rb.run_benchmark(rb.DOCUMENTS_PATH, rb.QUERIES_PATH, candidate_pool_size=25)
    report = rb.build_report(base_report, "python benchmarks/retrieval_benchmark.py --run-id test")

    json_path = temp_dir / "retrieval.json"
    markdown_path = temp_dir / "retrieval.md"
    csv_path = temp_dir / "retrieval.csv"
    rb.write_json(json_path, report)
    rb.write_markdown(markdown_path, report)
    rb.write_csv(csv_path, report)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")

    assert loaded["benchmark_name"] == "synthetic_retrieval_quality_benchmark"
    assert loaded["mode"] == "synthetic_offline"
    assert "does not call Pinecone" in " ".join(loaded["limitations"])
    assert "Synthetic Retrieval Quality Benchmark" in markdown
    assert "vector_only" in markdown
    assert csv_text.startswith("strategy,query_id,category")


@pytest.mark.benchmark
def test_bm25_improves_controlled_lexical_query(retrieval_chunks, retrieval_queries):
    query = _query_by_id(retrieval_queries, "rq015")
    vector = _rank_with_metrics(query, retrieval_chunks, rb.Strategy("vector_only", 1.0, 0.0))
    bm25 = _rank_with_metrics(query, retrieval_chunks, rb.Strategy("bm25_only", 0.0, 1.0))

    assert vector["top_results"][0]["chunk_id"] == "finance_nova_c03"
    assert bm25["top_results"][0]["chunk_id"] == "finance_orion_c03"
    assert bm25["metrics"]["mrr"] > vector["metrics"]["mrr"]
    assert bm25["metrics"]["recall@1"] > vector["metrics"]["recall@1"]


@pytest.mark.benchmark
def test_semantic_proxy_improves_controlled_ambiguous_query(retrieval_chunks, retrieval_queries):
    query = _query_by_id(retrieval_queries, "rq006")
    vector = _rank_with_metrics(query, retrieval_chunks, rb.Strategy("vector_only", 1.0, 0.0))
    bm25 = _rank_with_metrics(query, retrieval_chunks, rb.Strategy("bm25_only", 0.0, 1.0))

    assert vector["top_results"][0]["is_relevant"] is True
    assert bm25["top_results"][0]["is_relevant"] is False
    assert vector["metrics"]["mrr"] > bm25["metrics"]["mrr"]
