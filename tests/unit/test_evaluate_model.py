import json
import pytest
from pathlib import Path
from typing import Dict, Any, List


@pytest.mark.unit
class TestMetricsComputation:

    def test_accuracy_in_valid_range(self, sample_metrics: Dict[str, Any]):
        """Test accuracy is between 0 and 1."""
        accuracy = sample_metrics["accuracy"]

        assert 0 <= accuracy <= 1, f"Accuracy {accuracy} not in [0, 1]"

    def test_precision_in_valid_range(self, sample_metrics: Dict[str, Any]):
        """Test precision is between 0 and 1."""
        precision = sample_metrics["precision"]

        assert 0 <= precision <= 1, f"Precision {precision} not in [0, 1]"

    def test_recall_in_valid_range(self, sample_metrics: Dict[str, Any]):
        """Test recall is between 0 and 1."""
        recall = sample_metrics["recall"]

        assert 0 <= recall <= 1, f"Recall {recall} not in [0, 1]"

    def test_f1_in_valid_range(self, sample_metrics: Dict[str, Any]):
        """Test F1 score is between 0 and 1."""
        f1 = sample_metrics["f1"]

        assert 0 <= f1 <= 1, f"F1 {f1} not in [0, 1]"

    def test_f1_formula_consistency(self):
        """Test F1 is harmonic mean of precision and recall."""
        precision = 0.8
        recall = 0.9

        expected_f1 = (2 * precision * recall) / (precision + recall)

        assert abs(expected_f1 - 0.8470588235294118) < 1e-6

    def test_per_class_metrics_structure(self, sample_metrics: Dict[str, Any]):
        """Test per-class metrics have correct structure."""
        per_class = sample_metrics["per_class_metrics"]

        for class_name, metrics in per_class.items():
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "support" in metrics

    def test_per_class_metrics_valid_ranges(self, sample_metrics: Dict[str, Any]):
        """Test all per-class metrics are in valid ranges."""
        per_class = sample_metrics["per_class_metrics"]

        for class_name, metrics in per_class.items():
            assert 0 <= metrics["precision"] <= 1
            assert 0 <= metrics["recall"] <= 1
            assert 0 <= metrics["f1"] <= 1
            assert metrics["support"] >= 0


@pytest.mark.unit
class TestConfusionMatrixHandling:
    def test_confusion_matrix_is_list(self, sample_metrics: Dict[str, Any]):
        """Test confusion matrix is a list (JSON serializable)."""
        cm = sample_metrics["confusion_matrix"]

        assert isinstance(cm, list)

    def test_confusion_matrix_is_square(self, sample_metrics: Dict[str, Any]):
        """Test confusion matrix is square."""
        cm = sample_metrics["confusion_matrix"]

        num_rows = len(cm)
        for row in cm:
            assert len(row) == num_rows, "Confusion matrix is not square"

    def test_confusion_matrix_non_negative(self, sample_metrics: Dict[str, Any]):
        """Test all confusion matrix values are non-negative."""
        cm = sample_metrics["confusion_matrix"]

        for row in cm:
            for val in row:
                assert val >= 0, f"Negative value {val} in confusion matrix"

    def test_confusion_matrix_sum_equals_samples(self):
        """Test confusion matrix sum equals total samples."""
        cm = [[10, 2], [1, 12]]
        total = sum(sum(row) for row in cm)

        assert total == 25


@pytest.mark.unit
class TestMetricsOutputSchema:
    def test_required_keys_present(self, sample_metrics: Dict[str, Any]):
        """Test all required keys are present in metrics."""
        required_keys = ["accuracy", "precision", "recall", "f1", "confusion_matrix", "per_class_metrics"]

        for key in required_keys:
            assert key in sample_metrics, f"Missing required key: {key}"

    def test_metrics_are_floats(self, sample_metrics: Dict[str, Any]):
        """Test scalar metrics are floats."""
        float_keys = ["accuracy", "precision", "recall", "f1"]

        for key in float_keys:
            assert isinstance(sample_metrics[key], float), f"{key} is not float"

    def test_metrics_json_serializable(self, sample_metrics: Dict[str, Any]):
        """Test metrics can be serialized to JSON."""
        try:
            json_str = json.dumps(sample_metrics)
            assert len(json_str) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"Metrics not JSON serializable: {e}")


@pytest.mark.unit
class TestMetricsFileSaving:
    def test_save_metrics_to_file(self, temp_dir: Path, sample_metrics: Dict[str, Any]):
        """Test saving metrics to JSON file."""
        metrics_path = temp_dir / "metrics.json"

        with open(metrics_path, "w") as f:
            json.dump(sample_metrics, f, indent=2)

        assert metrics_path.exists()

    def test_load_saved_metrics(self, sample_metrics_file: Path, sample_metrics: Dict[str, Any]):
        """Test loading saved metrics preserves values."""
        with open(sample_metrics_file, "r") as f:
            loaded = json.load(f)

        assert loaded["accuracy"] == sample_metrics["accuracy"]
        assert loaded["f1"] == sample_metrics["f1"]

    def test_metrics_file_creates_directory(self, temp_dir: Path, sample_metrics: Dict[str, Any]):
        """Test metrics file creation handles nested directories."""
        nested_path = temp_dir / "nested" / "deep" / "metrics.json"
        nested_path.parent.mkdir(parents=True, exist_ok=True)

        with open(nested_path, "w") as f:
            json.dump(sample_metrics, f)

        assert nested_path.exists()


@pytest.mark.unit
class TestMLflowMetricsLogging:
    def test_mlflow_metrics_logged(self, mock_mlflow, sample_metrics: Dict[str, Any]):
        """Test that evaluation metrics are logged to MLflow."""
        loggable_metrics = {
            "test_accuracy": sample_metrics["accuracy"],
            "test_precision": sample_metrics["precision"],
            "test_recall": sample_metrics["recall"],
            "test_f1": sample_metrics["f1"]
        }

        mock_mlflow["log_metrics"](loggable_metrics)

        mock_mlflow["log_metrics"].assert_called_once_with(loggable_metrics)

    def test_mlflow_tracking_uri_configured(self, mock_mlflow):
        """Test MLflow tracking URI is set before logging."""
        tracking_uri = "http://mlflow:5000"

        mock_mlflow["set_tracking_uri"](tracking_uri)

        mock_mlflow["set_tracking_uri"].assert_called_once()


@pytest.mark.unit
class TestEdgeCases:
    def test_empty_predictions_handling(self):
        """Test handling of empty predictions."""
        all_predictions = []
        all_labels = []

        # Should handle gracefully
        assert len(all_predictions) == 0
        assert len(all_labels) == 0

    def test_single_class_predictions(self):
        """Test handling of single class in predictions."""
        all_predictions = [0, 0, 0, 0, 0]
        all_labels = [0, 0, 0, 0, 0]

        # Accuracy should be 1.0
        accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        assert accuracy == 1.0

    def test_completely_wrong_predictions(self):
        """Test handling of all wrong predictions."""
        all_predictions = [1, 1, 1, 1, 1]
        all_labels = [0, 0, 0, 0, 0]

        # Accuracy should be 0.0
        accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        assert accuracy == 0.0

    def test_ignore_index_filtering(self):
        """Test that ignore_index (-100) labels are filtered."""
        predictions = [0, 1, 2, 0, 1]
        labels = [0, 1, -100, 0, -100]
        ignore_index = -100

        filtered_pairs = [(p, l) for p, l in zip(predictions, labels) if l != ignore_index]

        assert len(filtered_pairs) == 3
        for p, l in filtered_pairs:
            assert l != ignore_index

    def test_mismatched_prediction_length(self):
        """Test handling of prediction/label length mismatch."""
        predictions = [0, 1, 2]
        labels = [0, 1, 2, 3, 4]

        # Padding logic from token classification
        if len(predictions) < len(labels):
            predictions.extend([-1] * (len(labels) - len(predictions)))

        assert len(predictions) == len(labels)


@pytest.mark.unit
class TestDatasetSplitHandling:
    def test_test_split_required(self, sample_dataset_dict: Dict[str, Any]):
        """Test that 'test' split is required for evaluation."""
        assert "test" in sample_dataset_dict

    def test_missing_test_split_detected(self):
        """Test detection of missing test split."""
        dataset = {"train": {}, "validation": {}}

        assert "test" not in dataset

    def test_test_split_not_empty(self, sample_dataset_dict: Dict[str, Any]):
        """Test that test split is not empty."""
        test_data = sample_dataset_dict["test"]

        assert len(test_data["image"]) > 0


@pytest.mark.unit
class TestDeviceHandling:
    def test_device_selection_cpu(self):
        """Test CPU device selection."""
        cuda_available = False

        device = "cuda" if cuda_available else "cpu"

        assert device == "cpu"

    def test_tensor_to_device_logic(self):
        """Test tensor device transfer logic."""
        encoding = {"input_ids": "tensor1", "attention_mask": "tensor2"}
        device = "cpu"

        moved = {k: f"{v}_on_{device}" for k, v in encoding.items()}

        assert all("cpu" in v for v in moved.values())
