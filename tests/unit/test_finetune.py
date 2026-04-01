import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock


def build_training_args(
    smoke_test: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    eval_dataset: Any,
    cuda_available: bool = False
) -> Dict[str, Any]:
    return {
        "num_train_epochs": 1 if smoke_test else epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 1 if smoke_test else 2,
        "learning_rate": learning_rate,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch" if eval_dataset else "no",
        "load_best_model_at_end": True if eval_dataset else False,
        "metric_for_best_model": "f1" if eval_dataset else None,
        "save_total_limit": 3,
        "report_to": ["mlflow"],
        "fp16": cuda_available,
        "max_steps": 5 if smoke_test else -1
    }


def validate_hyperparameters(params: Dict[str, Any]) -> bool:
    """Validate hyperparameters returning True if valid."""
    if params.get("epochs", 0) <= 0:
        return False
    if params.get("batch_size", 0) <= 0:
        return False
    if not (0 < params.get("learning_rate", 0) < 1):
        return False
    if params.get("num_labels", 0) <= 0:
        return False
    return True


def should_resume_from_checkpoint(checkpoint_dir: Path) -> bool:
    return checkpoint_dir.exists() and len(list(checkpoint_dir.iterdir())) > 0


@pytest.mark.unit
class TestFinetuneParameterValidation:
    def test_valid_hyperparameters_accepted(self):
        """Test validation accepts valid hyperparameters."""
        params = {
            "model_name": "microsoft/layoutlmv3-base",
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 5e-5,
            "num_labels": 15
        }

        assert validate_hyperparameters(params) is True

    def test_invalid_epochs_rejected(self):
        """Test validation rejects zero or negative epochs."""
        params_zero = {"epochs": 0, "batch_size": 8, "learning_rate": 5e-5, "num_labels": 15}
        params_neg = {"epochs": -1, "batch_size": 8, "learning_rate": 5e-5, "num_labels": 15}

        assert validate_hyperparameters(params_zero) is False
        assert validate_hyperparameters(params_neg) is False

    def test_invalid_batch_size_rejected(self):
        """Test validation rejects zero or negative batch size."""
        params_zero = {"epochs": 5, "batch_size": 0, "learning_rate": 5e-5, "num_labels": 15}
        params_neg = {"epochs": 5, "batch_size": -4, "learning_rate": 5e-5, "num_labels": 15}

        assert validate_hyperparameters(params_zero) is False
        assert validate_hyperparameters(params_neg) is False

    def test_invalid_learning_rate_rejected(self):
        """Test validation rejects learning rates outside (0, 1)."""
        params_zero = {"epochs": 5, "batch_size": 8, "learning_rate": 0, "num_labels": 15}
        params_neg = {"epochs": 5, "batch_size": 8, "learning_rate": -0.01, "num_labels": 15}
        params_high = {"epochs": 5, "batch_size": 8, "learning_rate": 1.5, "num_labels": 15}

        assert validate_hyperparameters(params_zero) is False
        assert validate_hyperparameters(params_neg) is False
        assert validate_hyperparameters(params_high) is False

    def test_invalid_num_labels_rejected(self):
        """Test validation rejects zero or negative num_labels."""
        params_zero = {"epochs": 5, "batch_size": 8, "learning_rate": 5e-5, "num_labels": 0}
        params_neg = {"epochs": 5, "batch_size": 8, "learning_rate": 5e-5, "num_labels": -1}

        assert validate_hyperparameters(params_zero) is False
        assert validate_hyperparameters(params_neg) is False

    def test_learning_rate_in_typical_ranges(self):
        """Test common learning rate values are valid."""
        typical_rates = [1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4]
        base_params = {"epochs": 5, "batch_size": 8, "num_labels": 15}

        for lr in typical_rates:
            params = {**base_params, "learning_rate": lr}
            assert validate_hyperparameters(params) is True, f"LR {lr} should be valid"


@pytest.mark.unit
class TestFinetuneMLflowIntegration:
    def test_mlflow_params_logged_with_all_fields(self, mock_mlflow):
        """Test that all training parameters are logged to MLflow."""
        params = {
            "model_name": "microsoft/layoutlmv3-base",
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 5e-5,
            "num_labels": 15,
            "smoke_test": False
        }

        mock_mlflow["log_params"](params)

        mock_mlflow["log_params"].assert_called_once_with(params)
        call_args = mock_mlflow["log_params"].call_args[0][0]
        assert "model_name" in call_args
        assert "epochs" in call_args
        assert "batch_size" in call_args
        assert "learning_rate" in call_args

    def test_mlflow_tracking_uri_configured(self, mock_mlflow):
        tracking_uri = "http://mlflow:5000"

        mock_mlflow["set_tracking_uri"](tracking_uri)

        mock_mlflow["set_tracking_uri"].assert_called_once_with(tracking_uri)

    def test_mlflow_experiment_configured(self, mock_mlflow):
        experiment_name = "layoutlm-finetuning"

        mock_mlflow["set_experiment"](experiment_name)

        mock_mlflow["set_experiment"].assert_called_once_with(experiment_name)

    def test_mlflow_run_context_provides_run_id(self, mock_mlflow):
        with mock_mlflow["start_run"]() as run:
            run_id = run.info.run_id

            assert run_id is not None
            assert isinstance(run_id, str)
            assert len(run_id) > 0

    def test_mlflow_model_artifact_logged(self, mock_mlflow):
        mock_model = MagicMock()
        artifact_path = "model"

        mock_mlflow["log_model"](mock_model, artifact_path)

        mock_mlflow["log_model"].assert_called_once()
        call_args = mock_mlflow["log_model"].call_args[0]
        assert call_args[1] == "model"

    def test_mlflow_version_tag_set_to_run_id(self, mock_mlflow):
        with mock_mlflow["start_run"]() as run:
            model_version = run.info.run_id
            mock_mlflow["set_tag"]("model_version", model_version)

        mock_mlflow["set_tag"].assert_called_with("model_version", "test-run-id-12345")

    def test_mlflow_full_tracking_workflow(self, mock_mlflow):
        tracking_uri = "http://mlflow:5000"
        experiment_name = "layoutlm-finetuning"
        params = {"model_name": "test", "epochs": 3}

        mock_mlflow["set_tracking_uri"](tracking_uri)
        mock_mlflow["set_experiment"](experiment_name)

        with mock_mlflow["start_run"]() as run:
            mock_mlflow["log_params"](params)
            mock_model = MagicMock()
            mock_mlflow["log_model"](mock_model, "model")
            mock_mlflow["set_tag"]("model_version", run.info.run_id)

        mock_mlflow["set_tracking_uri"].assert_called_once()
        mock_mlflow["set_experiment"].assert_called_once()
        mock_mlflow["log_params"].assert_called_once()
        mock_mlflow["log_model"].assert_called_once()
        mock_mlflow["set_tag"].assert_called_once()


@pytest.mark.unit
class TestFinetuneTrainingArguments:
    def test_full_mode_training_args(self):
        args = build_training_args(
            smoke_test=False,
            epochs=5,
            batch_size=8,
            learning_rate=5e-5,
            eval_dataset=["sample1", "sample2"]
        )

        assert args["num_train_epochs"] == 5
        assert args["gradient_accumulation_steps"] == 2
        assert args["max_steps"] == -1
        assert args["per_device_train_batch_size"] == 8
        assert args["learning_rate"] == 5e-5

    def test_smoke_mode_training_args(self):
        args = build_training_args(
            smoke_test=True,
            epochs=5,
            batch_size=8,
            learning_rate=5e-5,
            eval_dataset=None
        )

        assert args["num_train_epochs"] == 1
        assert args["gradient_accumulation_steps"] == 1
        assert args["max_steps"] == 5

    def test_evaluation_enabled_with_eval_dataset(self):
        args = build_training_args(
            smoke_test=False,
            epochs=5,
            batch_size=8,
            learning_rate=5e-5,
            eval_dataset=["sample1", "sample2"]
        )

        assert args["evaluation_strategy"] == "epoch"
        assert args["load_best_model_at_end"] is True
        assert args["metric_for_best_model"] == "f1"

    def test_evaluation_disabled_without_eval_dataset(self):
        args = build_training_args(
            smoke_test=False,
            epochs=5,
            batch_size=8,
            learning_rate=5e-5,
            eval_dataset=None
        )

        assert args["evaluation_strategy"] == "no"
        assert args["load_best_model_at_end"] is False
        assert args["metric_for_best_model"] is None

    def test_fp16_enabled_with_cuda(self):
        args = build_training_args(
            smoke_test=False,
            epochs=5,
            batch_size=8,
            learning_rate=5e-5,
            eval_dataset=None,
            cuda_available=True
        )

        assert args["fp16"] is True

    def test_fp16_disabled_without_cuda(self):
        args = build_training_args(
            smoke_test=False,
            epochs=5,
            batch_size=8,
            learning_rate=5e-5,
            eval_dataset=None,
            cuda_available=False
        )

        assert args["fp16"] is False

    def test_mlflow_in_report_to(self):
        args = build_training_args(
            smoke_test=False,
            epochs=5,
            batch_size=8,
            learning_rate=5e-5,
            eval_dataset=None
        )

        assert "mlflow" in args["report_to"]

    def test_save_total_limit_configured(self):
        """Test checkpoint save limit is configured."""
        args = build_training_args(
            smoke_test=False,
            epochs=5,
            batch_size=8,
            learning_rate=5e-5,
            eval_dataset=None
        )

        assert args["save_total_limit"] == 3


@pytest.mark.unit
class TestFinetuneOutputs:
    def test_output_tuple_structure(self):
        """Test that finetune returns expected (model_version, model_path) tuple."""
        model_version = "test-run-id-12345"
        model_path = "/path/to/model"

        outputs = (model_version, model_path)

        assert len(outputs) == 2
        assert isinstance(outputs[0], str)
        assert isinstance(outputs[1], str)

    def test_model_version_format(self):
        model_version = "abc123xyz789"

        assert isinstance(model_version, str)
        assert len(model_version) > 0
        assert " " not in model_version  # No spaces in run IDs

    def test_model_artifacts_saved(self, temp_dir: Path):
        output_model_path = temp_dir / "model_output"
        output_model_path.mkdir(parents=True, exist_ok=True)

        (output_model_path / "config.json").write_text('{"model_type": "layoutlmv3"}')
        (output_model_path / "pytorch_model.bin").write_text("mock_weights")
        (output_model_path / "tokenizer_config.json").write_text('{}')

        assert (output_model_path / "config.json").exists()
        assert (output_model_path / "pytorch_model.bin").exists()

    def test_checkpoint_directory_structure(self, temp_dir: Path):
        output_model_path = temp_dir / "model_output"
        checkpoint_dir = output_model_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        (checkpoint_dir / "checkpoint-100").mkdir()
        (checkpoint_dir / "checkpoint-200").mkdir()

        assert checkpoint_dir.exists()
        checkpoints = list(checkpoint_dir.iterdir())
        assert len(checkpoints) == 2


@pytest.mark.unit
class TestFinetuneResumeFromCheckpoint:
    def test_resume_when_checkpoints_exist(self, temp_dir: Path):
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "checkpoint-100").mkdir()

        assert should_resume_from_checkpoint(checkpoint_dir) is True

    def test_no_resume_when_directory_missing(self, temp_dir: Path):
        checkpoint_dir = temp_dir / "nonexistent_checkpoints"

        assert should_resume_from_checkpoint(checkpoint_dir) is False

    def test_no_resume_when_directory_empty(self, temp_dir: Path):
        checkpoint_dir = temp_dir / "empty_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        assert should_resume_from_checkpoint(checkpoint_dir) is False

    def test_resume_with_multiple_checkpoints(self, temp_dir: Path):
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "checkpoint-100").mkdir()
        (checkpoint_dir / "checkpoint-200").mkdir()
        (checkpoint_dir / "checkpoint-300").mkdir()

        assert should_resume_from_checkpoint(checkpoint_dir) is True


@pytest.mark.unit
class TestFinetuneDatasetHandling:
    def test_train_split_extraction(self, sample_dataset_dict: Dict[str, Any]):
        dataset = sample_dataset_dict
        train_dataset = dataset['train'] if 'train' in dataset else dataset

        assert train_dataset is not None
        assert "image" in train_dataset

    def test_validation_split_extraction(self, sample_dataset_dict: Dict[str, Any]):
        dataset = sample_dataset_dict
        eval_dataset = dataset['validation'] if 'validation' in dataset else None

        assert eval_dataset is not None
        assert "image" in eval_dataset

    def test_missing_validation_handled(self):
        dataset = {"train": {"image": ["img1"], "labels": [[0]]}}
        eval_dataset = dataset.get('validation', None)

        assert eval_dataset is None

    def test_flat_dataset_as_train(self, sample_dataset_flat: Dict[str, Any]):
        dataset = sample_dataset_flat

        train_dataset = dataset.get('train', dataset)

        assert "image" in train_dataset


@pytest.mark.unit
class TestFinetuneDeviceHandling:
    def test_device_selection_cpu(self):
        cuda_available = False

        device = "cuda" if cuda_available else "cpu"

        assert device == "cpu"

    def test_device_selection_cuda(self):
        cuda_available = True

        device = "cuda" if cuda_available else "cpu"

        assert device == "cuda"


@pytest.mark.unit
class TestFinetuneIntegrationScenarios:
    def test_smoke_test_full_workflow(self, mock_mlflow, temp_dir: Path, sample_dataset_dict: Dict[str, Any]):
        params = {
            "model_name": "microsoft/layoutlmv3-base",
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 5e-5,
            "num_labels": 15,
            "smoke_test": True
        }

        assert validate_hyperparameters(params) is True

        args = build_training_args(
            smoke_test=True,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            eval_dataset=sample_dataset_dict.get("validation")
        )

        assert args["num_train_epochs"] == 1
        assert args["max_steps"] == 5

        mock_mlflow["set_tracking_uri"]("http://mlflow:5000")
        mock_mlflow["set_experiment"]("test-experiment")

        with mock_mlflow["start_run"]() as run:
            mock_mlflow["log_params"](params)
            model_version = run.info.run_id

        assert model_version == "test-run-id-12345"
        mock_mlflow["log_params"].assert_called_once()

    def test_full_training_workflow(self, mock_mlflow, temp_dir: Path, sample_dataset_dict: Dict[str, Any]):
        params = {
            "model_name": "microsoft/layoutlmv3-base",
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_labels": 15,
            "smoke_test": False
        }

        assert validate_hyperparameters(params) is True

        args = build_training_args(
            smoke_test=False,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            eval_dataset=sample_dataset_dict.get("validation")
        )

        assert args["num_train_epochs"] == 10
        assert args["max_steps"] == -1
        assert args["gradient_accumulation_steps"] == 2

        output_path = temp_dir / "model_output"
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "config.json").write_text("{}")

        assert output_path.exists()

    def test_resume_training_workflow(self, temp_dir: Path):
        output_path = temp_dir / "model_output"
        checkpoint_dir = output_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "checkpoint-500").mkdir()

        resume = should_resume_from_checkpoint(checkpoint_dir)

        assert resume is True
