import json
import pytest
from pathlib import Path
from typing import Dict, Any

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.tasks.qa import QAEvaluator
from benchmarks.utils.metrics import compute_f1, compute_exact_match


@pytest.mark.benchmark
@pytest.mark.smoke
class TestQAEvaluatorSmoke:
    @pytest.fixture
    def qa_config(self) -> Dict[str, Any]:
        return {
            "smoke_test": True,
            "device": "cpu",
            "batch_size": 4,
            "tenant_id": "benchmark_tenant"
        }

    @pytest.fixture
    def qa_evaluator(self, qa_config: Dict[str, Any]) -> QAEvaluator:
        return QAEvaluator(qa_config)

    def test_load_model_initializes_api_caller(self, qa_evaluator: QAEvaluator):
        qa_evaluator.load_model("dummy")

        assert qa_evaluator.model is not None
        assert callable(qa_evaluator.model)

    def test_load_dataset_from_file(self, qa_evaluator: QAEvaluator, smoke_qa_data: list, temp_dir: Path):
        dataset_path = temp_dir / "qa_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        qa_evaluator.load_dataset(str(dataset_path))

        assert qa_evaluator.dataset is not None
        assert len(qa_evaluator.dataset) > 0

    def test_smoke_test_limits_dataset_size(self, qa_evaluator: QAEvaluator, temp_dir: Path):
        # Create dataset with more than 5 samples
        large_dataset = [
            {"id": str(i), "question": f"Q{i}?", "answer": f"A{i}", "mock_prediction": f"A{i}"}
            for i in range(20)
        ]

        dataset_path = temp_dir / "large_qa.json"
        with open(dataset_path, "w") as f:
            json.dump(large_dataset, f)

        qa_evaluator.load_dataset(str(dataset_path))

        # smoke_test=True should limit to 5
        assert len(qa_evaluator.dataset) == 5

    def test_compute_metrics_returns_expected_keys(self, qa_evaluator: QAEvaluator, smoke_qa_data: list,
                                                   temp_dir: Path):
        dataset_path = temp_dir / "qa_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        qa_evaluator.load_model("dummy")
        qa_evaluator.load_dataset(str(dataset_path))

        metrics = qa_evaluator.compute_metrics()

        assert "qa_f1" in metrics
        assert "qa_exact_match" in metrics

    def test_metrics_values_in_valid_range(self, qa_evaluator: QAEvaluator, smoke_qa_data: list, temp_dir: Path):
        """Test that F1 and exact match are between 0 and 1."""
        dataset_path = temp_dir / "qa_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        qa_evaluator.load_model("dummy")
        qa_evaluator.load_dataset(str(dataset_path))

        metrics = qa_evaluator.compute_metrics()

        assert 0 <= metrics["qa_f1"] <= 1, f"qa_f1 {metrics['qa_f1']} not in [0, 1]"
        assert 0 <= metrics["qa_exact_match"] <= 1, f"qa_exact_match {metrics['qa_exact_match']} not in [0, 1]"

    def test_mock_predictions_work_correctly(self, qa_evaluator: QAEvaluator, temp_dir: Path):
        """Test that mock predictions from dataset are used in smoke mode."""
        perfect_dataset = [
            {"id": "1", "question": "Q1?", "answer": "correct answer", "mock_prediction": "correct answer"}
        ]

        dataset_path = temp_dir / "perfect_qa.json"
        with open(dataset_path, "w") as f:
            json.dump(perfect_dataset, f)

        qa_evaluator.load_model("dummy")
        qa_evaluator.load_dataset(str(dataset_path))

        metrics = qa_evaluator.compute_metrics()

        # Perfect match should give F1=1.0 and EM=1.0
        assert metrics["qa_f1"] == 1.0
        assert metrics["qa_exact_match"] == 1.0


@pytest.mark.benchmark
@pytest.mark.smoke
class TestQAMetricsFunctions:
    def test_compute_f1_identical_strings(self):
        f1 = compute_f1("the answer", "the answer")
        assert f1 == 1.0

    def test_compute_f1_completely_different(self):
        f1 = compute_f1("apple orange", "banana grape")
        assert f1 == 0.0

    def test_compute_f1_partial_overlap(self):
        f1 = compute_f1("the quick brown fox", "the lazy brown dog")
        assert 0 < f1 < 1

    def test_compute_exact_match_identical(self):
        em = compute_exact_match("answer", "answer")
        assert em == 1.0

    def test_compute_exact_match_different(self):
        em = compute_exact_match("answer", "different")
        assert em == 0.0

    def test_compute_f1_empty_strings(self):
        f1 = compute_f1("", "")
        assert f1 == 1.0  # Both empty = match

    def test_compute_f1_one_empty(self):
        f1 = compute_f1("some text", "")
        assert f1 == 0.0


@pytest.mark.benchmark
@pytest.mark.smoke
class TestQAEvaluatorEdgeCases:
    @pytest.fixture
    def qa_config(self) -> Dict[str, Any]:
        return {"smoke_test": True, "device": "cpu"}

    def test_empty_dataset_raises_error(self, qa_config: Dict[str, Any]):
        evaluator = QAEvaluator(qa_config)
        evaluator.load_model("dummy")
        evaluator.dataset = []

        with pytest.raises(ValueError, match="Dataset not loaded"):
            evaluator.compute_metrics()

    def test_none_dataset_raises_error(self, qa_config: Dict[str, Any]):
        evaluator = QAEvaluator(qa_config)
        evaluator.load_model("dummy")
        # dataset is None by default

        with pytest.raises(ValueError, match="Dataset not loaded"):
            evaluator.compute_metrics()

    def test_handles_missing_answer_field(self, qa_config: Dict[str, Any], temp_dir: Path):
        dataset = [{"id": "1", "question": "Q?", "mock_prediction": "ans"}]  # No 'answer' key

        dataset_path = temp_dir / "no_answer.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        evaluator = QAEvaluator(qa_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))

        # Should handle with empty string default
        metrics = evaluator.compute_metrics()
        assert "qa_f1" in metrics
