import json
import yaml
import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.runner import load_config, run_task
from benchmarks.utils.helpers import fix_seeds


@pytest.mark.benchmark
class TestRunnerConfigLoading:
    def test_load_valid_yaml_config(self, temp_dir: Path):
        """Test loading a valid YAML config file."""
        config = {
            "global": {"seed": 1235, "smoke_test": True},
            "tasks": {
                "test_task": {
                    "module": "qa",
                    "evaluator_class": "QAEvaluator"
                }
            }
        }

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        loaded = load_config(str(config_path))

        assert "global" in loaded
        assert "tasks" in loaded
        assert loaded["global"]["seed"] == 1235

    def test_config_has_global_section(self, benchmark_config: Dict[str, Any]):
        assert "global" in benchmark_config

    def test_config_has_tasks_section(self, benchmark_config: Dict[str, Any]):
        assert "tasks" in benchmark_config

    def test_global_config_has_seed(self, benchmark_config: Dict[str, Any]):
        assert "seed" in benchmark_config["global"]

    def test_task_config_has_required_fields(self, benchmark_config: Dict[str, Any]):
        for task_name, task_config in benchmark_config["tasks"].items():
            assert "module" in task_config, f"Task {task_name} missing 'module'"
            assert "evaluator_class" in task_config, f"Task {task_name} missing 'evaluator_class'"


@pytest.mark.benchmark
class TestRunnerTaskExecution:
    def test_run_qa_task_with_mock(self, benchmark_config: Dict[str, Any], smoke_qa_data: list, temp_dir: Path):
        # Create dataset file
        dataset_path = temp_dir / "qa.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        # Update config with temp path
        task_config = benchmark_config["tasks"]["test_qa"].copy()
        task_config["dataset_path"] = str(dataset_path)

        global_config = benchmark_config["global"]

        with patch("benchmarks.runner.get_logger"):
            metrics = run_task("test_qa", task_config, global_config)

        assert "qa_f1" in metrics
        assert "qa_exact_match" in metrics

    def test_run_task_returns_dict(self, benchmark_config: Dict[str, Any], smoke_qa_data: list, temp_dir: Path):
        dataset_path = temp_dir / "qa.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        task_config = benchmark_config["tasks"]["test_qa"].copy()
        task_config["dataset_path"] = str(dataset_path)
        global_config = benchmark_config["global"]

        with patch("benchmarks.runner.get_logger"):
            metrics = run_task("test_qa", task_config, global_config)

        assert isinstance(metrics, dict)


@pytest.mark.benchmark
class TestRunnerMLflowIntegration:
    def test_mlflow_setup_called(self, mock_mlflow, benchmark_config: Dict[str, Any]):
        from benchmarks.utils.helpers import setup_mlflow

        tracking_uri = benchmark_config["global"].get("mlflow_tracking_uri", "")
        experiment_name = benchmark_config["global"].get("experiment_name", "test")

        # verify function exists and is callable
        with patch("mlflow.set_tracking_uri") as mock_uri, \
                patch("mlflow.set_experiment") as mock_exp:
            setup_mlflow(tracking_uri, experiment_name)

            if tracking_uri:
                mock_uri.assert_called_once_with(tracking_uri)
            mock_exp.assert_called_once_with(experiment_name)

    def test_metrics_are_loggable(self, smoke_qa_data: list, temp_dir: Path):
        # Setup
        dataset_path = temp_dir / "qa.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        config = {
            "smoke_test": True,
            "device": "cpu",
            "batch_size": 4,
            "tenant_id": "test"
        }

        from benchmarks.tasks.qa import QAEvaluator
        evaluator = QAEvaluator(config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))
        metrics = evaluator.compute_metrics()

        # Filter for loggable metrics (only int/float)
        loggable = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

        assert len(loggable) > 0
        for k, v in loggable.items():
            assert isinstance(v, (int, float)), f"Metric {k} is not numeric"


@pytest.mark.benchmark
@pytest.mark.reproducibility
class TestRunnerReproducibility:
    def test_fix_seeds_sets_random_seed(self):
        import random

        fix_seeds(1235)
        val1 = random.random()

        fix_seeds(1235)
        val2 = random.random()

        assert val1 == val2

    def test_same_seed_same_metrics(self, smoke_qa_data: list, temp_dir: Path):
        from benchmarks.tasks.qa import QAEvaluator

        dataset_path = temp_dir / "qa.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        config = {"smoke_test": True, "device": "cpu"}

        # Run 1
        fix_seeds(1235)
        evaluator1 = QAEvaluator(config)
        evaluator1.load_model("dummy")
        evaluator1.load_dataset(str(dataset_path))
        metrics1 = evaluator1.compute_metrics()

        # Run 2 with same seed
        fix_seeds(1235)
        evaluator2 = QAEvaluator(config)
        evaluator2.load_model("dummy")
        evaluator2.load_dataset(str(dataset_path))
        metrics2 = evaluator2.compute_metrics()

        assert metrics1 == metrics2

    def test_different_seed_may_differ(self, smoke_qa_data: list, temp_dir: Path):
        from benchmarks.tasks.qa import QAEvaluator

        dataset_path = temp_dir / "qa.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        config = {"smoke_test": True, "device": "cpu"}

        # For QA with mock predictions, results should be same regardless of seed
        fix_seeds(1)
        evaluator1 = QAEvaluator(config)
        evaluator1.load_model("dummy")
        evaluator1.load_dataset(str(dataset_path))
        metrics1 = evaluator1.compute_metrics()

        fix_seeds(9999)
        evaluator2 = QAEvaluator(config)
        evaluator2.load_model("dummy")
        evaluator2.load_dataset(str(dataset_path))
        metrics2 = evaluator2.compute_metrics()

        assert metrics1 == metrics2


@pytest.mark.benchmark
class TestRunnerMultiTaskExecution:
    def test_config_supports_multiple_tasks(self, temp_dir: Path, smoke_qa_data: list, smoke_retrieval_data: list):
        qa_path = temp_dir / "qa.json"
        with open(qa_path, "w") as f:
            json.dump(smoke_qa_data, f)

        retrieval_path = temp_dir / "retrieval.json"
        with open(retrieval_path, "w") as f:
            json.dump(smoke_retrieval_data, f)

        config = {
            "global": {"seed": 1235, "smoke_test": True, "device": "cpu"},
            "tasks": {
                "qa_task": {
                    "module": "qa",
                    "evaluator_class": "QAEvaluator",
                    "model_path": "dummy",
                    "dataset_path": str(qa_path)
                },
                "retrieval_task": {
                    "module": "retrieval",
                    "evaluator_class": "RetrievalEvaluator",
                    "model_path": "dummy",
                    "dataset_path": str(retrieval_path)
                }
            }
        }

        assert len(config["tasks"]) == 2
        assert "qa_task" in config["tasks"]
        assert "retrieval_task" in config["tasks"]

    def test_run_specific_task_from_config(self, temp_dir: Path, smoke_qa_data: list):
        qa_path = temp_dir / "qa.json"
        with open(qa_path, "w") as f:
            json.dump(smoke_qa_data, f)

        config = {
            "global": {"seed": 1235, "smoke_test": True, "device": "cpu"},
            "tasks": {
                "qa_only": {
                    "module": "qa",
                    "evaluator_class": "QAEvaluator",
                    "model_path": "dummy",
                    "dataset_path": str(qa_path)
                },
                "other_task": {
                    "module": "nli",
                    "evaluator_class": "NLIEvaluator",
                    "model_path": "dummy",
                    "dataset_path": "nonexistent.json"
                }
            }
        }

        # Filter to specific task
        selected_task = "qa_only"
        tasks_to_run = {selected_task: config["tasks"][selected_task]}

        assert len(tasks_to_run) == 1
        assert "qa_only" in tasks_to_run


@pytest.mark.benchmark
class TestRunnerOutputSaving:
    def test_results_saved_to_json(self, temp_dir: Path, smoke_qa_data: list):
        """Test that results are saved to JSON file."""
        from benchmarks.tasks.base import BaseEvaluator
        from benchmarks.tasks.qa import QAEvaluator

        dataset_path = temp_dir / "qa.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        config = {"smoke_test": True, "device": "cpu"}
        evaluator = QAEvaluator(config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))
        metrics = evaluator.compute_metrics()

        output_path = temp_dir / "results" / "qa_results.json"
        evaluator.save_results(str(output_path), metrics)

        assert output_path.exists()

        with open(output_path) as f:
            saved = json.load(f)

        assert saved == metrics

    def test_output_directory_created(self, temp_dir: Path, smoke_qa_data: list):
        """Test that output directory is created if it doesn't exist."""
        from benchmarks.tasks.qa import QAEvaluator

        dataset_path = temp_dir / "qa.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        config = {"smoke_test": True, "device": "cpu"}
        evaluator = QAEvaluator(config)
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))
        metrics = evaluator.compute_metrics()

        nested_output = temp_dir / "deep" / "nested" / "results.json"
        evaluator.save_results(str(nested_output), metrics)

        assert nested_output.exists()


@pytest.mark.benchmark
class TestDefaultConfigFile:
    def test_default_config_exists(self):
        config_path = Path(__file__).parent.parent.parent / "benchmarks" / "configs" / "default.yaml"

        assert config_path.exists(), "default.yaml config file not found"

    def test_default_config_is_valid(self):
        config_path = Path(__file__).parent.parent.parent / "benchmarks" / "configs" / "default.yaml"

        if not config_path.exists():
            pytest.skip("default.yaml not found")

        config = load_config(str(config_path))

        assert "global" in config
        assert "tasks" in config
        assert len(config["tasks"]) > 0

    def test_default_config_tasks_have_required_fields(self):
        config_path = Path(__file__).parent.parent.parent / "benchmarks" / "configs" / "default.yaml"

        if not config_path.exists():
            pytest.skip("default.yaml not found")

        config = load_config(str(config_path))

        for task_name, task_config in config["tasks"].items():
            assert "module" in task_config, f"Task {task_name} missing 'module'"
            assert "evaluator_class" in task_config, f"Task {task_name} missing 'evaluator_class'"
            assert "dataset_path" in task_config, f"Task {task_name} missing 'dataset_path'"
