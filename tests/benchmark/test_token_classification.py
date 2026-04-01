import json
import pytest
from pathlib import Path
from typing import Dict, Any

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.tasks.token_classification import TokenClassificationEvaluator


@pytest.mark.benchmark
@pytest.mark.smoke
class TestTokenClassificationEvaluatorSmoke:
    @pytest.fixture
    def token_config(self) -> Dict[str, Any]:
        return {
            "smoke_test": True,
            "device": "cpu",
            "batch_size": 4,
            "tenant_id": "benchmark_tenant",
            "ignore_index": -100
        }

    @pytest.fixture
    def token_evaluator(self, token_config: Dict[str, Any]) -> TokenClassificationEvaluator:
        return TokenClassificationEvaluator(token_config)

    def test_load_model_initializes_api_caller(self, token_evaluator: TokenClassificationEvaluator):
        token_evaluator.load_model("dummy")

        assert token_evaluator.model is not None
        assert callable(token_evaluator.model)

    def test_load_dataset_from_file(self, token_evaluator: TokenClassificationEvaluator, smoke_token_data: list,
                                    temp_dir: Path):
        dataset_path = temp_dir / "token_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_token_data, f)

        token_evaluator.load_dataset(str(dataset_path))

        assert token_evaluator.dataset is not None
        assert len(token_evaluator.dataset) > 0

    def test_smoke_test_limits_dataset_size(self, token_evaluator: TokenClassificationEvaluator, temp_dir: Path):
        large_dataset = [
            {
                "id": str(i),
                "text": f"Sample text {i}",
                "tokens": ["word1", "word2"],
                "labels": [0, 1],
                "mock_predictions": [0, 1]
            }
            for i in range(20)
        ]

        dataset_path = temp_dir / "large_token.json"
        with open(dataset_path, "w") as f:
            json.dump(large_dataset, f)

        token_evaluator.load_dataset(str(dataset_path))

        assert len(token_evaluator.dataset) == 5

    def test_compute_metrics_returns_expected_keys(self, token_evaluator: TokenClassificationEvaluator,
                                                   smoke_token_data: list, temp_dir: Path):
        """Test compute_metrics returns precision, recall, F1, and confusion matrix."""
        dataset_path = temp_dir / "token_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_token_data, f)

        token_evaluator.load_model("dummy")
        token_evaluator.load_dataset(str(dataset_path))

        metrics = token_evaluator.compute_metrics()

        assert "token_precision_macro" in metrics
        assert "token_recall_macro" in metrics
        assert "token_f1_macro" in metrics
        assert "confusion_matrix" in metrics

    def test_metrics_values_in_valid_range(self, token_evaluator: TokenClassificationEvaluator, smoke_token_data: list,
                                           temp_dir: Path):
        """Test that precision, recall, F1 are between 0 and 1."""
        dataset_path = temp_dir / "token_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_token_data, f)

        token_evaluator.load_model("dummy")
        token_evaluator.load_dataset(str(dataset_path))

        metrics = token_evaluator.compute_metrics()

        assert 0 <= metrics["token_precision_macro"] <= 1, f"precision {metrics['token_precision_macro']} not in [0, 1]"
        assert 0 <= metrics["token_recall_macro"] <= 1, f"recall {metrics['token_recall_macro']} not in [0, 1]"
        assert 0 <= metrics["token_f1_macro"] <= 1, f"F1 {metrics['token_f1_macro']} not in [0, 1]"

    def test_confusion_matrix_is_list(self, token_evaluator: TokenClassificationEvaluator, smoke_token_data: list,
                                      temp_dir: Path):
        """Test that confusion matrix is returned as a list (JSON serializable)."""
        dataset_path = temp_dir / "token_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_token_data, f)

        token_evaluator.load_model("dummy")
        token_evaluator.load_dataset(str(dataset_path))

        metrics = token_evaluator.compute_metrics()

        assert isinstance(metrics["confusion_matrix"], list)

    def test_perfect_predictions_give_high_scores(self, token_evaluator: TokenClassificationEvaluator, temp_dir: Path):
        """Test that perfect mock predictions give high F1."""
        perfect_dataset = [
            {
                "id": "1",
                "text": "Perfect prediction test",
                "tokens": ["Perfect", "prediction", "test"],
                "labels": [0, 1, 2],
                "mock_predictions": [0, 1, 2]  # Perfect match
            }
        ]

        dataset_path = temp_dir / "perfect_token.json"
        with open(dataset_path, "w") as f:
            json.dump(perfect_dataset, f)

        token_evaluator.load_model("dummy")
        token_evaluator.load_dataset(str(dataset_path))

        metrics = token_evaluator.compute_metrics()

        assert metrics["token_f1_macro"] == 1.0


@pytest.mark.benchmark
@pytest.mark.smoke
class TestTokenClassificationIgnoreIndex:
    @pytest.fixture
    def token_config(self) -> Dict[str, Any]:
        return {"smoke_test": True, "device": "cpu", "ignore_index": -100}

    def test_ignore_index_labels_filtered(self, token_config: Dict[str, Any], temp_dir: Path):
        """Test that labels with ignore_index=-100 are filtered out."""
        dataset = [
            {
                "id": "1",
                "text": "Test with padding",
                "tokens": ["Test", "with", "padding", "[PAD]", "[PAD]"],
                "labels": [0, 1, 0, -100, -100],  # Last two are padding
                "mock_predictions": [0, 1, 0, 0, 0]  # Predictions for all tokens
            }
        ]

        dataset_path = temp_dir / "padded.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        evaluator = TokenClassificationEvaluator(token_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))

        metrics = evaluator.compute_metrics()

        # Should complete without error, ignoring -100 labels
        assert "token_f1_macro" in metrics
        # Perfect predictions on non-padded tokens
        assert metrics["token_f1_macro"] == 1.0


@pytest.mark.benchmark
@pytest.mark.smoke
class TestTokenClassificationPredictionLengthMismatch:
    @pytest.fixture
    def token_config(self) -> Dict[str, Any]:
        return {"smoke_test": True, "device": "cpu", "ignore_index": -100}

    def test_shorter_predictions_padded(self, token_config: Dict[str, Any], temp_dir: Path):
        """Test that shorter predictions are padded with -1."""
        dataset = [
            {
                "id": "1",
                "text": "More labels than predictions",
                "tokens": ["a", "b", "c", "d", "e"],
                "labels": [0, 1, 2, 3, 4],
                "mock_predictions": [0, 1, 2]  # Only 3 predictions for 5 labels
            }
        ]

        dataset_path = temp_dir / "short_preds.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        evaluator = TokenClassificationEvaluator(token_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))

        # Should handle by padding predictions
        metrics = evaluator.compute_metrics()
        assert "token_f1_macro" in metrics

    def test_longer_predictions_truncated(self, token_config: Dict[str, Any], temp_dir: Path):
        """Test that longer predictions are truncated to match labels."""
        dataset = [
            {
                "id": "1",
                "text": "Fewer labels",
                "tokens": ["a", "b"],
                "labels": [0, 1],
                "mock_predictions": [0, 1, 2, 3, 4]  # 5 predictions for 2 labels
            }
        ]

        dataset_path = temp_dir / "long_preds.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        evaluator = TokenClassificationEvaluator(token_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))

        # Should truncate predictions
        metrics = evaluator.compute_metrics()
        assert metrics["token_f1_macro"] == 1.0  # First 2 predictions match


@pytest.mark.benchmark
@pytest.mark.smoke
class TestTokenClassificationEdgeCases:
    @pytest.fixture
    def token_config(self) -> Dict[str, Any]:
        return {"smoke_test": True, "device": "cpu", "ignore_index": -100}

    def test_empty_dataset_raises_error(self, token_config: Dict[str, Any]):
        evaluator = TokenClassificationEvaluator(token_config)
        evaluator.load_model("dummy")
        evaluator.dataset = []

        with pytest.raises(ValueError, match="Dataset not loaded"):
            evaluator.compute_metrics()

    def test_all_ignore_index_returns_zero_metrics(self, token_config: Dict[str, Any], temp_dir: Path):
        """Test handling when all labels are ignore_index."""
        dataset = [
            {
                "id": "1",
                "text": "All padding",
                "tokens": ["[PAD]", "[PAD]"],
                "labels": [-100, -100],  # All ignored
                "mock_predictions": [0, 0]
            }
        ]

        dataset_path = temp_dir / "all_padding.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        evaluator = TokenClassificationEvaluator(token_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))

        metrics = evaluator.compute_metrics()

        # Should return zero metrics gracefully
        assert metrics["token_f1_macro"] == 0.0


@pytest.mark.benchmark
@pytest.mark.smoke
class TestTokenClassificationDataSamplesFile:
    @pytest.fixture
    def token_config(self) -> Dict[str, Any]:
        return {"smoke_test": True, "device": "cpu", "ignore_index": -100}

    def test_load_actual_smoke_file(self, token_config: Dict[str, Any]):
        """Test loading the actual smoke_token.json file."""
        data_path = Path(__file__).parent.parent.parent / "benchmarks" / "data_samples" / "smoke_token.json"

        if not data_path.exists():
            pytest.skip("smoke_token.json not found")

        evaluator = TokenClassificationEvaluator(token_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(data_path))

        assert evaluator.dataset is not None
        assert len(evaluator.dataset) <= 5

    def test_compute_metrics_on_actual_data(self, token_config: Dict[str, Any]):
        """Test compute_metrics on actual smoke data."""
        data_path = Path(__file__).parent.parent.parent / "benchmarks" / "data_samples" / "smoke_token.json"

        if not data_path.exists():
            pytest.skip("smoke_token.json not found")

        evaluator = TokenClassificationEvaluator(token_config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(data_path))

        metrics = evaluator.compute_metrics()

        assert "token_precision_macro" in metrics
        assert "token_recall_macro" in metrics
        assert "token_f1_macro" in metrics
        assert "confusion_matrix" in metrics

        # Validate ranges
        assert 0 <= metrics["token_precision_macro"] <= 1
        assert 0 <= metrics["token_recall_macro"] <= 1
        assert 0 <= metrics["token_f1_macro"] <= 1
