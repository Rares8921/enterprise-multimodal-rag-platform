import json
import pytest
from pathlib import Path
from typing import Dict, Any

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.tasks.retrieval import RetrievalEvaluator
from benchmarks.utils.metrics import compute_mrr, compute_recall_at_k


@pytest.mark.benchmark
@pytest.mark.smoke
class TestRetrievalEvaluatorSmoke:
    @pytest.fixture
    def retrieval_config(self) -> Dict[str, Any]:
        return {
            "smoke_test": True,
            "device": "cpu",
            "batch_size": 4,
            "tenant_id": "benchmark_tenant"
        }

    @pytest.fixture
    def retrieval_evaluator(self, retrieval_config: Dict[str, Any]) -> RetrievalEvaluator:
        return RetrievalEvaluator(retrieval_config)

    def test_load_model_initializes_api_caller(self, retrieval_evaluator: RetrievalEvaluator):
        retrieval_evaluator.load_model("dummy")

        assert retrieval_evaluator.model is not None
        assert callable(retrieval_evaluator.model)

    def test_load_dataset_from_file(self, retrieval_evaluator: RetrievalEvaluator, smoke_retrieval_data: list,
                                    temp_dir: Path):
        dataset_path = temp_dir / "retrieval_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_retrieval_data, f)

        retrieval_evaluator.load_dataset(str(dataset_path))

        assert retrieval_evaluator.dataset is not None
        assert len(retrieval_evaluator.dataset) > 0

    def test_smoke_test_limits_dataset_size(self, retrieval_evaluator: RetrievalEvaluator, temp_dir: Path):
        large_dataset = [
            {
                "id": str(i),
                "query": f"Query {i}",
                "relevant_ids": [f"doc_{i}"],
                "mock_retrieved_ids": [f"doc_{i}", "doc_random"]
            }
            for i in range(20)
        ]

        dataset_path = temp_dir / "large_retrieval.json"
        with open(dataset_path, "w") as f:
            json.dump(large_dataset, f)

        retrieval_evaluator.load_dataset(str(dataset_path))

        assert len(retrieval_evaluator.dataset) == 5

    def test_compute_metrics_returns_expected_keys(self, retrieval_evaluator: RetrievalEvaluator,
                                                   smoke_retrieval_data: list, temp_dir: Path):
        dataset_path = temp_dir / "retrieval_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_retrieval_data, f)

        retrieval_evaluator.load_model("dummy")
        retrieval_evaluator.load_dataset(str(dataset_path))

        metrics = retrieval_evaluator.compute_metrics()

        assert "retrieval_mrr" in metrics
        assert "retrieval_recall@5" in metrics

    def test_metrics_values_in_valid_range(self, retrieval_evaluator: RetrievalEvaluator, smoke_retrieval_data: list,
                                           temp_dir: Path):
        """Test that MRR and recall@5 are between 0 and 1."""
        dataset_path = temp_dir / "retrieval_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_retrieval_data, f)

        retrieval_evaluator.load_model("dummy")
        retrieval_evaluator.load_dataset(str(dataset_path))

        metrics = retrieval_evaluator.compute_metrics()

        assert 0 <= metrics["retrieval_mrr"] <= 1, f"retrieval_mrr {metrics['retrieval_mrr']} not in [0, 1]"
        assert 0 <= metrics[
            "retrieval_recall@5"] <= 1, f"retrieval_recall@5 {metrics['retrieval_recall@5']} not in [0, 1]"

    def test_perfect_retrieval_gives_perfect_scores(self, retrieval_evaluator: RetrievalEvaluator, temp_dir: Path):
        """Test that perfect retrieval gives MRR=1.0 and recall@5=1.0."""
        perfect_dataset = [
            {
                "id": "1",
                "query": "Query 1",
                "relevant_ids": ["doc_1"],
                "mock_retrieved_ids": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]  # doc_1 at rank 1
            }
        ]

        dataset_path = temp_dir / "perfect_retrieval.json"
        with open(dataset_path, "w") as f:
            json.dump(perfect_dataset, f)

        retrieval_evaluator.load_model("dummy")
        retrieval_evaluator.load_dataset(str(dataset_path))

        metrics = retrieval_evaluator.compute_metrics()

        assert metrics["retrieval_mrr"] == 1.0
        assert metrics["retrieval_recall@5"] == 1.0


@pytest.mark.benchmark
@pytest.mark.smoke
class TestRetrievalMetricsFunctions:
    def test_compute_mrr_perfect_retrieval(self):
        relevant = [["doc_1"]]
        retrieved = [["doc_1", "doc_2", "doc_3"]]

        mrr = compute_mrr(relevant, retrieved)

        assert mrr == 1.0

    def test_compute_mrr_second_position(self):
        relevant = [["doc_1"]]
        retrieved = [["doc_x", "doc_1", "doc_y"]]

        mrr = compute_mrr(relevant, retrieved)

        assert mrr == 0.5

    def test_compute_mrr_not_found(self):
        relevant = [["doc_1"]]
        retrieved = [["doc_x", "doc_y", "doc_z"]]

        mrr = compute_mrr(relevant, retrieved)

        assert mrr == 0.0

    def test_compute_mrr_multiple_queries(self):
        relevant = [["doc_1"], ["doc_2"]]
        retrieved = [["doc_1"], ["doc_x", "doc_2"]]  # rank 1 + rank 2

        mrr = compute_mrr(relevant, retrieved)

        expected = (1.0 + 0.5) / 2
        assert abs(mrr - expected) < 1e-6

    def test_compute_recall_at_k_all_found(self):
        """Test recall@k is 1.0 when all relevant docs found in top-k."""
        relevant = [["doc_1", "doc_2"]]
        retrieved = [["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]]

        recall = compute_recall_at_k(relevant, retrieved, k=5)

        assert recall == 1.0

    def test_compute_recall_at_k_partial(self):
        """Test recall@k for partial retrieval."""
        relevant = [["doc_1", "doc_2"]]
        retrieved = [["doc_1", "doc_x", "doc_y", "doc_z", "doc_w"]]  # Only doc_1 found

        recall = compute_recall_at_k(relevant, retrieved, k=5)

        assert recall == 0.5

    def test_compute_recall_at_k_none_found(self):
        """Test recall@k is 0.0 when no relevant docs found."""
        relevant = [["doc_1"]]
        retrieved = [["doc_x", "doc_y", "doc_z"]]

        recall = compute_recall_at_k(relevant, retrieved, k=5)

        assert recall == 0.0

    def test_compute_recall_k_boundary(self):
        """Test recall@k respects k boundary."""
        relevant = [["doc_6"]]  # doc_6 is at position 6
        retrieved = [["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"]]

        recall_at_5 = compute_recall_at_k(relevant, retrieved, k=5)
        recall_at_6 = compute_recall_at_k(relevant, retrieved, k=6)

        assert recall_at_5 == 0.0  # doc_6 not in top 5
        assert recall_at_6 == 1.0  # doc_6 in top 6


@pytest.mark.benchmark
@pytest.mark.smoke
class TestRetrievalEvaluatorEdgeCases:
    @pytest.fixture
    def retrieval_config(self) -> Dict[str, Any]:
        return {"smoke_test": True, "device": "cpu"}

    def test_empty_dataset_raises_error(self, retrieval_config: Dict[str, Any]):
        evaluator = RetrievalEvaluator(retrieval_config)
        evaluator.load_model("dummy")
        evaluator.dataset = []

        with pytest.raises(ValueError, match="Dataset not loaded"):
            evaluator.compute_metrics()

    def test_empty_relevant_ids_handled(self, retrieval_config: Dict[str, Any], temp_dir: Path):
        dataset = [
            {
                "id": "1",
                "query": "Query",
                "relevant_ids": [],  # No relevant docs
                "mock_retrieved_ids": ["doc_1", "doc_2"]
            }
        ]

        dataset_path = temp_dir / "empty_relevant.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        evaluator = RetrievalEvaluator(retrieval_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))

        # Should handle
        metrics = evaluator.compute_metrics()
        assert "retrieval_mrr" in metrics

    def test_empty_retrieved_ids_handled(self, retrieval_config: Dict[str, Any], temp_dir: Path):
        """Test handling of empty retrieved_ids (API returns nothing)."""
        dataset = [
            {
                "id": "1",
                "query": "Query",
                "relevant_ids": ["doc_1"],
                "mock_retrieved_ids": []  # No results returned
            }
        ]

        dataset_path = temp_dir / "empty_retrieved.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        evaluator = RetrievalEvaluator(retrieval_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))

        metrics = evaluator.compute_metrics()

        assert metrics["retrieval_mrr"] == 0.0
        assert metrics["retrieval_recall@5"] == 0.0


@pytest.mark.benchmark
@pytest.mark.smoke
class TestRetrievalLatency:
    @pytest.fixture
    def retrieval_config(self) -> Dict[str, Any]:
        return {"smoke_test": True, "device": "cpu"}

    def test_smoke_retrieval_completes_quickly(self, retrieval_config: Dict[str, Any], temp_dir: Path):
        import time

        dataset = [
            {
                "id": str(i),
                "query": f"Query {i}",
                "relevant_ids": [f"doc_{i}"],
                "mock_retrieved_ids": [f"doc_{i}", "doc_x"]
            }
            for i in range(5)
        ]

        dataset_path = temp_dir / "latency_test.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        evaluator = RetrievalEvaluator(retrieval_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))

        start = time.time()
        metrics = evaluator.compute_metrics()
        elapsed_ms = (time.time() - start) * 1000

        # Smoke test should complete in < 1000ms
        assert elapsed_ms < 1000, f"Retrieval took {elapsed_ms:.0f}ms, expected < 1000ms"
