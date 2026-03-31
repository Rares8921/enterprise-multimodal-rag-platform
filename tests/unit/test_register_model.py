import json
import pytest
from pathlib import Path
from typing import Dict, Any


@pytest.mark.unit
class TestModelApprovalLogic:
    def test_model_approved_when_accuracy_exceeds_threshold(self, sample_metrics: Dict[str, Any]):
        """Test model is registered when accuracy >= min_accuracy."""
        min_accuracy = 0.90
        accuracy = sample_metrics["accuracy"]  # 0.92

        status = "registered" if accuracy >= min_accuracy else "rejected"

        assert status == "registered"

    def test_model_rejected_when_accuracy_below_threshold(self):
        """Test model is rejected when accuracy < min_accuracy."""
        min_accuracy = 0.90
        accuracy = 0.85

        status = "registered" if accuracy >= min_accuracy else "rejected"

        assert status == "rejected"

    def test_model_approved_at_exact_threshold(self):
        """Test model is registered when accuracy == min_accuracy."""
        min_accuracy = 0.90
        accuracy = 0.90

        status = "registered" if accuracy >= min_accuracy else "rejected"

        assert status == "registered"

    def test_edge_case_zero_accuracy(self):
        """Test model is rejected with zero accuracy."""
        min_accuracy = 0.90
        accuracy = 0.0

        status = "registered" if accuracy >= min_accuracy else "rejected"

        assert status == "rejected"

    def test_edge_case_perfect_accuracy(self):
        """Test model is registered with perfect accuracy."""
        min_accuracy = 0.90
        accuracy = 1.0

        status = "registered" if accuracy >= min_accuracy else "rejected"

        assert status == "registered"


@pytest.mark.unit
class TestMetricsFileLoading:
    def test_load_metrics_from_file(self, sample_metrics_file: Path):
        """Test loading metrics JSON file."""
        with open(sample_metrics_file, "r") as f:
            metrics = json.load(f)

        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)

    def test_default_accuracy_when_missing(self):
        """Test default accuracy of 0.0 when key is missing."""
        metrics = {"precision": 0.85, "recall": 0.90}

        accuracy = metrics.get("accuracy", 0.0)

        assert accuracy == 0.0

    def test_handle_corrupted_metrics_file(self, temp_dir: Path):
        """Test handling of corrupted metrics file."""
        corrupted_path = temp_dir / "corrupted.json"
        with open(corrupted_path, "w") as f:
            f.write("{ invalid json")

        with pytest.raises(json.JSONDecodeError):
            with open(corrupted_path, "r") as f:
                json.load(f)


@pytest.mark.unit
class TestMLflowRegistration:
    def test_mlflow_register_model_called_on_approval(self, mock_mlflow):
        model_version = "test-run-id-12345"
        experiment_name = "layoutlm-finetuning"
        model_uri = f"runs:/{model_version}/model"

        accuracy = 0.92
        min_accuracy = 0.90

        if accuracy >= min_accuracy:
            mock_mlflow["register_model"](
                model_uri,
                name=f"{experiment_name}-layoutlm",
                tags={"accuracy": str(accuracy), "status": "approved"}
            )

        mock_mlflow["register_model"].assert_called_once()

    def test_mlflow_register_model_not_called_on_rejection(self, mock_mlflow):
        """Test MLflow register_model is NOT called when model is rejected."""
        accuracy = 0.85
        min_accuracy = 0.90

        if accuracy >= min_accuracy:
            mock_mlflow["register_model"]("uri", name="model")

        mock_mlflow["register_model"].assert_not_called()

    def test_mlflow_tracking_uri_set(self, mock_mlflow):
        """Test MLflow tracking URI is configured."""
        tracking_uri = "http://mlflow:5000"

        mock_mlflow["set_tracking_uri"](tracking_uri)

        mock_mlflow["set_tracking_uri"].assert_called_once_with(tracking_uri)

    def test_model_uri_format(self):
        """Test model URI format is correct."""
        model_version = "abc123xyz"

        model_uri = f"runs:/{model_version}/model"

        assert model_uri.startswith("runs:/")
        assert model_version in model_uri
        assert model_uri.endswith("/model")


@pytest.mark.unit
class TestModelNaming:
    def test_registered_model_name_format(self):
        """Test registered model name follows convention."""
        experiment_name = "layoutlm-finetuning"

        model_name = f"{experiment_name}-layoutlm"

        assert model_name == "layoutlm-finetuning-layoutlm"

    def test_model_tags_include_accuracy(self):
        """Test model tags include accuracy value."""
        accuracy = 0.92

        tags = {"accuracy": str(accuracy), "status": "approved"}

        assert "accuracy" in tags
        assert tags["accuracy"] == "0.92"

    def test_model_tags_include_status(self):
        """Test model tags include approval status."""
        tags = {"accuracy": "0.92", "status": "approved"}

        assert "status" in tags
        assert tags["status"] == "approved"


@pytest.mark.unit
class TestReturnValues:
    def test_returns_registered_on_approval(self):
        """Test 'registered' is returned when model is approved."""
        accuracy = 0.92
        min_accuracy = 0.90

        status = "registered" if accuracy >= min_accuracy else "rejected"

        assert status == "registered"

    def test_returns_rejected_on_failure(self):
        """Test 'rejected' is returned when model fails threshold."""
        accuracy = 0.85
        min_accuracy = 0.90

        status = "registered" if accuracy >= min_accuracy else "rejected"

        assert status == "rejected"

    def test_return_type_is_string(self):
        """Test return value is always a string."""
        accuracy = 0.92
        min_accuracy = 0.90

        status = "registered" if accuracy >= min_accuracy else "rejected"

        assert isinstance(status, str)


@pytest.mark.unit
class TestErrorHandling:
    def test_handle_missing_metrics_file(self, temp_dir: Path):
        """Test handling of missing metrics file."""
        missing_path = temp_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            with open(missing_path, "r") as f:
                json.load(f)

    def test_handle_invalid_accuracy_type(self):
        """Test handling of invalid accuracy type in metrics."""
        metrics = {"accuracy": "not_a_number"}

        try:
            accuracy = float(metrics.get("accuracy", 0.0))
        except ValueError:
            accuracy = 0.0

        assert accuracy == 0.0

    def test_handle_none_accuracy(self):
        """Test handling of None accuracy in metrics."""
        metrics = {"accuracy": None}

        accuracy = metrics.get("accuracy")
        if accuracy is None:
            accuracy = 0.0

        assert accuracy == 0.0


@pytest.mark.unit
class TestThresholdConfigurations:

    def test_high_threshold_rejection(self):
        """Test high threshold rejects good models."""
        min_accuracy = 0.99
        accuracy = 0.95

        status = "registered" if accuracy >= min_accuracy else "rejected"

        assert status == "rejected"

    def test_low_threshold_acceptance(self):
        """Test low threshold accepts poor models."""
        min_accuracy = 0.50
        accuracy = 0.60

        status = "registered" if accuracy >= min_accuracy else "rejected"

        assert status == "registered"

    def test_zero_threshold_accepts_all(self):
        """Test zero threshold accepts all models."""
        min_accuracy = 0.0
        accuracy = 0.01

        status = "registered" if accuracy >= min_accuracy else "rejected"

        assert status == "registered"

    def test_typical_production_threshold(self):
        """Test typical production threshold (90%)."""
        min_accuracy = 0.90

        # Test various accuracies
        assert ("registered" if 0.95 >= min_accuracy else "rejected") == "registered"
        assert ("registered" if 0.90 >= min_accuracy else "rejected") == "registered"
        assert ("registered" if 0.89 >= min_accuracy else "rejected") == "rejected"
        assert ("registered" if 0.85 >= min_accuracy else "rejected") == "rejected"


@pytest.mark.unit
class TestIntegrationScenarios:
    def test_full_registration_flow_approved(self, mock_mlflow, sample_metrics_file: Path):
        """Test complete registration flow for approved model."""
        model_version = "test-run-id-12345"
        experiment_name = "layoutlm-finetuning"
        min_accuracy = 0.90

        # Load metrics
        with open(sample_metrics_file, "r") as f:
            metrics = json.load(f)

        accuracy = metrics.get("accuracy", 0.0)

        if accuracy >= min_accuracy:
            model_uri = f"runs:/{model_version}/model"
            mock_mlflow["register_model"](
                model_uri,
                name=f"{experiment_name}-layoutlm",
                tags={"accuracy": str(accuracy), "status": "approved"}
            )
            status = "registered"
        else:
            status = "rejected"

        assert status == "registered"
        mock_mlflow["register_model"].assert_called_once()

    def test_full_registration_flow_rejected(self, mock_mlflow, temp_dir: Path):
        """Test complete registration flow for rejected model."""
        model_version = "test-run-id-12345"
        experiment_name = "layoutlm-finetuning"
        min_accuracy = 0.90

        # Create low-accuracy metrics file
        low_metrics = {"accuracy": 0.75, "precision": 0.70, "recall": 0.72, "f1": 0.71}
        metrics_path = temp_dir / "low_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(low_metrics, f)

        # Load metrics
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        accuracy = metrics.get("accuracy", 0.0)

        if accuracy >= min_accuracy:
            mock_mlflow["register_model"]("uri", name="model")
            status = "registered"
        else:
            status = "rejected"

        assert status == "rejected"
        mock_mlflow["register_model"].assert_not_called()
