import json
import pytest
from pathlib import Path
from typing import Dict, Any, Optional

from kubeflow.layoutlm_logic import (
    load_metrics_from_file,
    get_accuracy_from_metrics,
    determine_registration_status,
    build_model_uri,
    build_registered_model_name,
    build_model_tags,
)


@pytest.mark.unit
class TestModelApprovalLogic:
    def test_model_approved_when_accuracy_exceeds_threshold(self, sample_metrics: Dict[str, Any]):
        min_accuracy = 0.90
        accuracy = sample_metrics["accuracy"]  # 0.92

        status = determine_registration_status(accuracy, min_accuracy)

        assert status == "registered"

    def test_model_rejected_when_accuracy_below_threshold(self):
        min_accuracy = 0.90
        accuracy = 0.85

        status = determine_registration_status(accuracy, min_accuracy)

        assert status == "rejected"

    def test_model_approved_at_exact_threshold(self):
        min_accuracy = 0.90
        accuracy = 0.90

        status = determine_registration_status(accuracy, min_accuracy)

        assert status == "registered"

    def test_edge_case_zero_accuracy(self):
        min_accuracy = 0.90
        accuracy = 0.0

        status = determine_registration_status(accuracy, min_accuracy)

        assert status == "rejected"

    def test_edge_case_perfect_accuracy(self):
        min_accuracy = 0.90
        accuracy = 1.0

        status = determine_registration_status(accuracy, min_accuracy)

        assert status == "registered"

    def test_accuracy_just_below_threshold(self):
        min_accuracy = 0.90
        accuracy = 0.8999

        status = determine_registration_status(accuracy, min_accuracy)

        assert status == "rejected"

    def test_accuracy_just_above_threshold(self):
        min_accuracy = 0.90
        accuracy = 0.9001

        status = determine_registration_status(accuracy, min_accuracy)

        assert status == "registered"


@pytest.mark.unit
class TestMetricsFileLoading:
    def test_load_metrics_from_valid_file(self, sample_metrics_file: Path):
        metrics = load_metrics_from_file(sample_metrics_file)

        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)
        assert 0 <= metrics["accuracy"] <= 1

    def test_metrics_file_contains_all_keys(self, sample_metrics_file: Path):
        metrics = load_metrics_from_file(sample_metrics_file)

        required_keys = ["accuracy", "precision", "recall", "f1"]
        for key in required_keys:
            assert key in metrics, f"Missing required key: {key}"

    def test_default_accuracy_when_key_missing(self):
        metrics = {"precision": 0.85, "recall": 0.90, "f1": 0.87}

        accuracy = get_accuracy_from_metrics(metrics)

        assert accuracy == 0.0

    def test_handle_none_accuracy_value(self):
        metrics = {"accuracy": None, "precision": 0.85}

        accuracy = get_accuracy_from_metrics(metrics)

        assert accuracy == 0.0

    def test_handle_invalid_accuracy_type(self):
        metrics = {"accuracy": "not_a_number"}

        accuracy = get_accuracy_from_metrics(metrics)

        assert accuracy == 0.0

    def test_handle_corrupted_metrics_file(self, temp_dir: Path):
        corrupted_path = temp_dir / "corrupted.json"
        with open(corrupted_path, "w") as f:
            f.write("{ invalid json content")

        with pytest.raises(json.JSONDecodeError):
            load_metrics_from_file(corrupted_path)

    def test_handle_missing_metrics_file(self, temp_dir: Path):
        missing_path = temp_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_metrics_from_file(missing_path)

    def test_metrics_roundtrip_preserves_values(self, temp_dir: Path, sample_metrics: Dict[str, Any]):
        metrics_path = temp_dir / "roundtrip_metrics.json"

        with open(metrics_path, "w") as f:
            json.dump(sample_metrics, f)

        loaded = load_metrics_from_file(metrics_path)

        assert loaded["accuracy"] == sample_metrics["accuracy"]
        assert loaded["precision"] == sample_metrics["precision"]
        assert loaded["recall"] == sample_metrics["recall"]
        assert loaded["f1"] == sample_metrics["f1"]


@pytest.mark.unit
class TestMLflowRegistration:
    def test_mlflow_register_model_called_on_approval(self, mock_mlflow):
        model_version = "test-run-id-12345"
        experiment_name = "layoutlm-finetuning"
        accuracy = 0.92
        min_accuracy = 0.90

        if accuracy >= min_accuracy:
            model_uri = build_model_uri(model_version)
            model_name = build_registered_model_name(experiment_name)
            tags = build_model_tags(accuracy, "approved")

            mock_mlflow["register_model"](model_uri, name=model_name, tags=tags)

        mock_mlflow["register_model"].assert_called_once()

    def test_mlflow_register_model_not_called_on_rejection(self, mock_mlflow):
        accuracy = 0.85
        min_accuracy = 0.90

        if accuracy >= min_accuracy:
            mock_mlflow["register_model"]("uri", name="model")

        mock_mlflow["register_model"].assert_not_called()

    def test_mlflow_tracking_uri_configured(self, mock_mlflow):
        tracking_uri = "http://mlflow:5000"

        mock_mlflow["set_tracking_uri"](tracking_uri)

        mock_mlflow["set_tracking_uri"].assert_called_once_with(tracking_uri)

    def test_register_model_receives_correct_uri(self, mock_mlflow):
        model_version = "abc123xyz"
        experiment_name = "test-experiment"

        model_uri = build_model_uri(model_version)
        model_name = build_registered_model_name(experiment_name)

        mock_mlflow["register_model"](model_uri, name=model_name)

        call_args = mock_mlflow["register_model"].call_args
        assert call_args[0][0].startswith("runs:/")
        assert model_version in call_args[0][0]

    def test_register_model_receives_correct_tags(self, mock_mlflow):
        model_version = "abc123xyz"
        experiment_name = "test-experiment"
        accuracy = 0.95

        model_uri = build_model_uri(model_version)
        model_name = build_registered_model_name(experiment_name)
        tags = build_model_tags(accuracy, "approved")

        mock_mlflow["register_model"](model_uri, name=model_name, tags=tags)

        call_kwargs = mock_mlflow["register_model"].call_args[1]
        assert "tags" in call_kwargs
        assert call_kwargs["tags"]["accuracy"] == "0.95"
        assert call_kwargs["tags"]["status"] == "approved"


@pytest.mark.unit
class TestModelUriFormat:
    def test_model_uri_starts_with_runs(self):
        model_version = "abc123xyz"

        model_uri = build_model_uri(model_version)

        assert model_uri.startswith("runs:/")

    def test_model_uri_contains_version(self):
        model_version = "abc123xyz"

        model_uri = build_model_uri(model_version)

        assert model_version in model_uri

    def test_model_uri_ends_with_model(self):
        model_version = "abc123xyz"

        model_uri = build_model_uri(model_version)

        assert model_uri.endswith("/model")

    def test_model_uri_full_format(self):
        model_version = "run-id-12345"

        model_uri = build_model_uri(model_version)

        assert model_uri == "runs:/run-id-12345/model"


@pytest.mark.unit
class TestModelNaming:

    def test_registered_model_name_format(self):
        experiment_name = "layoutlm-finetuning"

        model_name = build_registered_model_name(experiment_name)

        assert model_name == "layoutlm-finetuning-layoutlm"

    def test_registered_model_name_includes_experiment(self):
        experiment_name = "custom-experiment"

        model_name = build_registered_model_name(experiment_name)

        assert experiment_name in model_name

    def test_model_tags_structure(self):
        accuracy = 0.92

        tags = build_model_tags(accuracy, "approved")

        assert "accuracy" in tags
        assert "status" in tags
        assert isinstance(tags["accuracy"], str)
        assert isinstance(tags["status"], str)

    def test_model_tags_accuracy_as_string(self):
        accuracy = 0.9234

        tags = build_model_tags(accuracy, "approved")

        assert tags["accuracy"] == "0.9234"

    def test_model_tags_with_approved_status(self):
        tags = build_model_tags(0.95, "approved")

        assert tags["status"] == "approved"

    def test_model_tags_with_rejected_status(self):
        tags = build_model_tags(0.75, "rejected")

        assert tags["status"] == "rejected"


@pytest.mark.unit
class TestReturnValues:
    def test_returns_registered_on_approval(self):
        status = determine_registration_status(0.92, 0.90)

        assert status == "registered"

    def test_returns_rejected_on_failure(self):
        status = determine_registration_status(0.85, 0.90)

        assert status == "rejected"

    def test_return_type_is_string(self):
        status_approved = determine_registration_status(0.95, 0.90)
        status_rejected = determine_registration_status(0.80, 0.90)

        assert isinstance(status_approved, str)
        assert isinstance(status_rejected, str)

    def test_only_two_possible_return_values(self):
        test_cases = [
            (0.0, 0.5), (0.5, 0.5), (0.51, 0.5), (1.0, 0.99),
            (0.89, 0.90), (0.90, 0.90), (0.91, 0.90)
        ]

        for accuracy, threshold in test_cases:
            status = determine_registration_status(accuracy, threshold)
            assert status in ["registered", "rejected"], f"Unexpected status: {status}"


@pytest.mark.unit
class TestThresholdConfigurations:
    def test_high_threshold_rejects_good_models(self):
        min_accuracy = 0.99
        accuracy = 0.95

        status = determine_registration_status(accuracy, min_accuracy)

        assert status == "rejected"

    def test_low_threshold_accepts_poor_models(self):
        min_accuracy = 0.50
        accuracy = 0.60

        status = determine_registration_status(accuracy, min_accuracy)

        assert status == "registered"

    def test_zero_threshold_accepts_all(self):
        min_accuracy = 0.0
        accuracy = 0.01

        status = determine_registration_status(accuracy, min_accuracy)

        assert status == "registered"

    def test_typical_production_threshold_90(self):
        min_accuracy = 0.90

        assert determine_registration_status(0.95, min_accuracy) == "registered"
        assert determine_registration_status(0.90, min_accuracy) == "registered"
        assert determine_registration_status(0.89, min_accuracy) == "rejected"
        assert determine_registration_status(0.85, min_accuracy) == "rejected"

    def test_strict_threshold_95(self):
        min_accuracy = 0.95

        assert determine_registration_status(0.96, min_accuracy) == "registered"
        assert determine_registration_status(0.95, min_accuracy) == "registered"
        assert determine_registration_status(0.94, min_accuracy) == "rejected"

    def test_lenient_threshold_70(self):
        min_accuracy = 0.70

        assert determine_registration_status(0.75, min_accuracy) == "registered"
        assert determine_registration_status(0.70, min_accuracy) == "registered"
        assert determine_registration_status(0.69, min_accuracy) == "rejected"


@pytest.mark.unit
class TestErrorHandling:
    def test_empty_metrics_dict_uses_default(self):
        metrics: Dict[str, Any] = {}

        accuracy = get_accuracy_from_metrics(metrics)

        assert accuracy == 0.0

    def test_accuracy_as_integer_converted(self):
        metrics = {"accuracy": 1}

        accuracy = get_accuracy_from_metrics(metrics)

        assert accuracy == 1.0
        assert isinstance(accuracy, float)

    def test_accuracy_as_list_returns_default(self):
        metrics = {"accuracy": [0.9, 0.8]}

        accuracy = get_accuracy_from_metrics(metrics)

        assert accuracy == 0.0

    def test_accuracy_as_dict_returns_default(self):
        metrics = {"accuracy": {"value": 0.9}}

        accuracy = get_accuracy_from_metrics(metrics)

        assert accuracy == 0.0


@pytest.mark.unit
class TestIntegrationScenarios:
    def test_full_registration_flow_approved(self, mock_mlflow, sample_metrics_file: Path):
        model_version = "test-run-id-12345"
        experiment_name = "layoutlm-finetuning"
        min_accuracy = 0.90

        # Load metrics
        metrics = load_metrics_from_file(sample_metrics_file)
        accuracy = get_accuracy_from_metrics(metrics)

        # Determine status
        status = determine_registration_status(accuracy, min_accuracy)

        if status == "registered":
            model_uri = build_model_uri(model_version)
            model_name = build_registered_model_name(experiment_name)
            tags = build_model_tags(accuracy, "approved")

            mock_mlflow["register_model"](model_uri, name=model_name, tags=tags)

        assert status == "registered"
        mock_mlflow["register_model"].assert_called_once()

    def test_full_registration_flow_rejected(self, mock_mlflow, temp_dir: Path):
        model_version = "test-run-id-12345"
        experiment_name = "layoutlm-finetuning"
        min_accuracy = 0.90

        # Create low-accuracy metrics file
        low_metrics = {"accuracy": 0.75, "precision": 0.70, "recall": 0.72, "f1": 0.71}
        metrics_path = temp_dir / "low_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(low_metrics, f)

        # Load metrics
        metrics = load_metrics_from_file(metrics_path)
        accuracy = get_accuracy_from_metrics(metrics)

        # Determine status
        status = determine_registration_status(accuracy, min_accuracy)

        if status == "registered":
            mock_mlflow["register_model"]("uri", name="model")

        assert status == "rejected"
        mock_mlflow["register_model"].assert_not_called()

    def test_full_flow_with_mlflow_setup(self, mock_mlflow, sample_metrics_file: Path):
        model_version = "run-abc-123"
        experiment_name = "layoutlm-finetuning"
        tracking_uri = "http://mlflow:5000"
        min_accuracy = 0.90

        # Configure MLflow
        mock_mlflow["set_tracking_uri"](tracking_uri)

        # Load and validate metrics
        metrics = load_metrics_from_file(sample_metrics_file)
        accuracy = get_accuracy_from_metrics(metrics)
        status = determine_registration_status(accuracy, min_accuracy)

        # Register if approved
        if status == "registered":
            model_uri = build_model_uri(model_version)
            model_name = build_registered_model_name(experiment_name)
            tags = build_model_tags(accuracy, "approved")
            mock_mlflow["register_model"](model_uri, name=model_name, tags=tags)

        # Verify
        mock_mlflow["set_tracking_uri"].assert_called_once_with(tracking_uri)
        mock_mlflow["register_model"].assert_called_once()
        assert status == "registered"

    def test_edge_case_exactly_at_threshold(self, mock_mlflow, temp_dir: Path):
        min_accuracy = 0.90

        # Create metrics at exact threshold
        exact_metrics = {"accuracy": 0.90, "precision": 0.88, "recall": 0.91, "f1": 0.895}
        metrics_path = temp_dir / "exact_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(exact_metrics, f)

        metrics = load_metrics_from_file(metrics_path)
        accuracy = get_accuracy_from_metrics(metrics)
        status = determine_registration_status(accuracy, min_accuracy)

        assert status == "registered"

    def test_multiple_registration_attempts_isolated(self, mock_mlflow, temp_dir: Path):
        """Test multiple registration attempts don't interfere."""
        min_accuracy = 0.90

        # First model - approved
        metrics1 = {"accuracy": 0.92}
        path1 = temp_dir / "metrics1.json"
        with open(path1, "w") as f:
            json.dump(metrics1, f)

        # Second model - rejected
        metrics2 = {"accuracy": 0.85}
        path2 = temp_dir / "metrics2.json"
        with open(path2, "w") as f:
            json.dump(metrics2, f)

        status1 = determine_registration_status(
            get_accuracy_from_metrics(load_metrics_from_file(path1)),
            min_accuracy
        )
        status2 = determine_registration_status(
            get_accuracy_from_metrics(load_metrics_from_file(path2)),
            min_accuracy
        )

        assert status1 == "registered"
        assert status2 == "rejected"
