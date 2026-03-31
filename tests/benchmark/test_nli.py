import json
import pytest
from pathlib import Path
from typing import Dict, Any

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.tasks.nli import NLIEvaluator


@pytest.mark.benchmark
@pytest.mark.smoke
class TestNLIEvaluatorSmoke:
    @pytest.fixture
    def nli_config(self) -> Dict[str, Any]:
        return {
            "smoke_test": True,
            "device": "cpu",
            "batch_size": 4,
            "tenant_id": "benchmark_tenant"
        }

    @pytest.fixture
    def nli_evaluator(self, nli_config: Dict[str, Any]) -> NLIEvaluator:
        return NLIEvaluator(nli_config)

    def test_load_model_initializes_api_caller(self, nli_evaluator: NLIEvaluator):
        nli_evaluator.load_model("dummy")

        assert nli_evaluator.model is not None
        assert callable(nli_evaluator.model)

    def test_load_dataset_from_file(self, nli_evaluator: NLIEvaluator, smoke_nli_data: list, temp_dir: Path):
        dataset_path = temp_dir / "nli_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_nli_data, f)

        nli_evaluator.load_dataset(str(dataset_path))

        assert nli_evaluator.dataset is not None
        assert len(nli_evaluator.dataset) > 0

    def test_smoke_test_limits_dataset_size(self, nli_evaluator: NLIEvaluator, temp_dir: Path):
        large_dataset = [
            {
                "id": str(i),
                "premise": f"Premise {i}",
                "hypothesis": f"Hypothesis {i}",
                "label": "neutral",
                "mock_prediction": "neutral"
            }
            for i in range(20)
        ]

        dataset_path = temp_dir / "large_nli.json"
        with open(dataset_path, "w") as f:
            json.dump(large_dataset, f)

        nli_evaluator.load_dataset(str(dataset_path))

        assert len(nli_evaluator.dataset) == 5

    def test_compute_metrics_returns_expected_keys(self, nli_evaluator: NLIEvaluator, smoke_nli_data: list,
                                                   temp_dir: Path):
        dataset_path = temp_dir / "nli_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_nli_data, f)

        nli_evaluator.load_model("dummy")
        nli_evaluator.load_dataset(str(dataset_path))

        metrics = nli_evaluator.compute_metrics()

        assert "nli_accuracy" in metrics
        assert "nli_f1_macro" in metrics

    def test_metrics_values_in_valid_range(self, nli_evaluator: NLIEvaluator, smoke_nli_data: list, temp_dir: Path):
        """Test that accuracy and F1 are between 0 and 1."""
        dataset_path = temp_dir / "nli_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_nli_data, f)

        nli_evaluator.load_model("dummy")
        nli_evaluator.load_dataset(str(dataset_path))

        metrics = nli_evaluator.compute_metrics()

        assert 0 <= metrics["nli_accuracy"] <= 1, f"nli_accuracy {metrics['nli_accuracy']} not in [0, 1]"
        assert 0 <= metrics["nli_f1_macro"] <= 1, f"nli_f1_macro {metrics['nli_f1_macro']} not in [0, 1]"

    def test_perfect_predictions_give_perfect_scores(self, nli_evaluator: NLIEvaluator, temp_dir: Path):
        """Test that perfect mock predictions give accuracy=1.0."""
        perfect_dataset = [
            {"id": "1", "premise": "P1", "hypothesis": "H1", "label": "entailment", "mock_prediction": "entailment"},
            {"id": "2", "premise": "P2", "hypothesis": "H2", "label": "contradiction",
             "mock_prediction": "contradiction"},
            {"id": "3", "premise": "P3", "hypothesis": "H3", "label": "neutral", "mock_prediction": "neutral"}
        ]

        dataset_path = temp_dir / "perfect_nli.json"
        with open(dataset_path, "w") as f:
            json.dump(perfect_dataset, f)

        nli_evaluator.load_model("dummy")
        nli_evaluator.load_dataset(str(dataset_path))

        metrics = nli_evaluator.compute_metrics()

        assert metrics["nli_accuracy"] == 1.0
        assert metrics["nli_f1_macro"] == 1.0


@pytest.mark.benchmark
@pytest.mark.smoke
class TestNLILabels:
    @pytest.fixture
    def nli_config(self) -> Dict[str, Any]:
        return {"smoke_test": True, "device": "cpu"}

    def test_all_label_types_handled(self, nli_config: Dict[str, Any], temp_dir: Path):
        """Test all NLI label types (entailment, contradiction, neutral) are handled."""
        dataset = [
            {"id": "1", "premise": "P", "hypothesis": "H", "label": "entailment", "mock_prediction": "entailment"},
            {"id": "2", "premise": "P", "hypothesis": "H", "label": "contradiction",
             "mock_prediction": "contradiction"},
            {"id": "3", "premise": "P", "hypothesis": "H", "label": "neutral", "mock_prediction": "neutral"}
        ]

        dataset_path = temp_dir / "all_labels.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        evaluator = NLIEvaluator(nli_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))

        metrics = evaluator.compute_metrics()

        # Should complete without error
        assert "nli_accuracy" in metrics

    def test_wrong_predictions_lower_accuracy(self, nli_config: Dict[str, Any], temp_dir: Path):
        dataset = [
            {"id": "1", "premise": "P", "hypothesis": "H", "label": "entailment", "mock_prediction": "contradiction"},
            {"id": "2", "premise": "P", "hypothesis": "H", "label": "contradiction", "mock_prediction": "neutral"}
        ]

        dataset_path = temp_dir / "wrong_preds.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        evaluator = NLIEvaluator(nli_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))

        metrics = evaluator.compute_metrics()

        assert metrics["nli_accuracy"] == 0.0


@pytest.mark.benchmark
@pytest.mark.smoke
class TestNLIEvaluatorEdgeCases:
    @pytest.fixture
    def nli_config(self) -> Dict[str, Any]:
        return {"smoke_test": True, "device": "cpu"}

    def test_empty_dataset_raises_error(self, nli_config: Dict[str, Any]):
        evaluator = NLIEvaluator(nli_config)
        evaluator.load_model("dummy")
        evaluator.dataset = []

        with pytest.raises(ValueError, match="Dataset not loaded"):
            evaluator.compute_metrics()

    def test_handles_missing_label_field(self, nli_config: Dict[str, Any], temp_dir: Path):
        dataset = [
            {"id": "1", "premise": "P", "hypothesis": "H", "mock_prediction": "neutral"}
            # No 'label' key - should default to "neutral"
        ]

        dataset_path = temp_dir / "no_label.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        evaluator = NLIEvaluator(nli_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))

        # Should handle with neutral default
        metrics = evaluator.compute_metrics()
        assert "nli_accuracy" in metrics


@pytest.mark.benchmark
@pytest.mark.smoke
class TestNLIDataSamplesFile:
    @pytest.fixture
    def nli_config(self) -> Dict[str, Any]:
        return {"smoke_test": True, "device": "cpu"}

    def test_load_actual_smoke_file(self, nli_config: Dict[str, Any]):
        data_path = Path(__file__).parent.parent.parent / "benchmarks" / "data_samples" / "smoke_nli.json"

        if not data_path.exists():
            pytest.skip("smoke_nli.json not found")

        evaluator = NLIEvaluator(nli_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(data_path))

        assert evaluator.dataset is not None
        assert len(evaluator.dataset) <= 5  # smoke_test limits to 5

    def test_compute_metrics_on_actual_data(self, nli_config: Dict[str, Any]):
        data_path = Path(__file__).parent.parent.parent / "benchmarks" / "data_samples" / "smoke_nli.json"

        if not data_path.exists():
            pytest.skip("smoke_nli.json not found")

        evaluator = NLIEvaluator(nli_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(data_path))

        metrics = evaluator.compute_metrics()

        assert "nli_accuracy" in metrics
        assert "nli_f1_macro" in metrics
        assert 0 <= metrics["nli_accuracy"] <= 1
        assert 0 <= metrics["nli_f1_macro"] <= 1
