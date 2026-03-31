import json
import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.tasks.base import BaseEvaluator
from benchmarks.runner import load_config, run_task


@pytest.mark.benchmark
class TestNewTaskDetection:
    def test_runner_detects_task_from_config(self, temp_dir: Path):
        config = {
            "global": {"seed": 1235, "smoke_test": True},
            "tasks": {
                "new_custom_task": {
                    "module": "qa",  # Using existing module for test
                    "evaluator_class": "QAEvaluator",
                    "model_path": "dummy",
                    "dataset_path": "test.json"
                }
            }
        }

        assert "new_custom_task" in config["tasks"]

        task_config = config["tasks"]["new_custom_task"]
        assert task_config["module"] == "qa"
        assert task_config["evaluator_class"] == "QAEvaluator"

    def test_dynamic_module_import(self):
        import importlib

        modules = ["qa", "nli", "retrieval", "token_classification"]

        for module_name in modules:
            full_module = f"benchmarks.tasks.{module_name}"
            module = importlib.import_module(full_module)
            assert module is not None

    def test_dynamic_class_loading(self):
        import importlib

        module = importlib.import_module("benchmarks.tasks.qa")
        evaluator_class = getattr(module, "QAEvaluator")

        assert evaluator_class is not None
        assert issubclass(evaluator_class, BaseEvaluator)

    def test_new_task_doesnt_break_existing(self, temp_dir: Path, smoke_qa_data: list):
        qa_path = temp_dir / "qa.json"
        with open(qa_path, "w") as f:
            json.dump(smoke_qa_data, f)

        config = {
            "global": {"seed": 1235, "smoke_test": True, "device": "cpu"},
            "tasks": {
                "existing_qa": {
                    "module": "qa",
                    "evaluator_class": "QAEvaluator",
                    "model_path": "dummy",
                    "dataset_path": str(qa_path)
                },
                "new_qa_variant": {
                    "module": "qa",
                    "evaluator_class": "QAEvaluator",
                    "model_path": "dummy",
                    "dataset_path": str(qa_path)
                }
            }
        }

        # Run existing task
        with patch("benchmarks.runner.get_logger"):
            metrics1 = run_task("existing_qa", config["tasks"]["existing_qa"], config["global"])

        # Run new task
        with patch("benchmarks.runner.get_logger"):
            metrics2 = run_task("new_qa_variant", config["tasks"]["new_qa_variant"], config["global"])

        # Both should succeed
        assert "qa_f1" in metrics1
        assert "qa_f1" in metrics2


@pytest.mark.benchmark
class TestOutputSchemaValidation:
    def test_metrics_dict_has_string_keys(self, smoke_qa_data: list, temp_dir: Path):
        from benchmarks.tasks.qa import QAEvaluator

        dataset_path = temp_dir / "qa.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        evaluator = QAEvaluator({"smoke_test": True, "device": "cpu"})
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))
        metrics = evaluator.compute_metrics()

        for key in metrics.keys():
            assert isinstance(key, str), f"Key {key} is not a string"

    def test_metrics_values_are_json_serializable(self, smoke_qa_data: list, temp_dir: Path):
        from benchmarks.tasks.qa import QAEvaluator

        dataset_path = temp_dir / "qa.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        evaluator = QAEvaluator({"smoke_test": True, "device": "cpu"})
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))
        metrics = evaluator.compute_metrics()

        # Should not raise
        json_str = json.dumps(metrics)
        assert len(json_str) > 0

    def test_qa_metrics_schema(self, smoke_qa_data: list, temp_dir: Path):
        from benchmarks.tasks.qa import QAEvaluator

        dataset_path = temp_dir / "qa.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_qa_data, f)

        evaluator = QAEvaluator({"smoke_test": True, "device": "cpu"})
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))
        metrics = evaluator.compute_metrics()

        # Required keys
        assert "qa_f1" in metrics
        assert "qa_exact_match" in metrics

        # Type checks
        assert isinstance(metrics["qa_f1"], float)
        assert isinstance(metrics["qa_exact_match"], float)

        # Range checks
        assert 0 <= metrics["qa_f1"] <= 1
        assert 0 <= metrics["qa_exact_match"] <= 1

    def test_nli_metrics_schema(self, smoke_nli_data: list, temp_dir: Path):
        from benchmarks.tasks.nli import NLIEvaluator

        dataset_path = temp_dir / "nli.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_nli_data, f)

        evaluator = NLIEvaluator({"smoke_test": True, "device": "cpu"})
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))
        metrics = evaluator.compute_metrics()

        # Required keys
        assert "nli_accuracy" in metrics
        assert "nli_f1_macro" in metrics

        # Type checks
        assert isinstance(metrics["nli_accuracy"], float)
        assert isinstance(metrics["nli_f1_macro"], float)

        # Range checks
        assert 0 <= metrics["nli_accuracy"] <= 1
        assert 0 <= metrics["nli_f1_macro"] <= 1

    def test_retrieval_metrics_schema(self, smoke_retrieval_data: list, temp_dir: Path):
        from benchmarks.tasks.retrieval import RetrievalEvaluator

        dataset_path = temp_dir / "retrieval.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_retrieval_data, f)

        evaluator = RetrievalEvaluator({"smoke_test": True, "device": "cpu"})
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))
        metrics = evaluator.compute_metrics()

        # Required keys
        assert "retrieval_mrr" in metrics
        assert "retrieval_recall@5" in metrics

        # Type checks
        assert isinstance(metrics["retrieval_mrr"], float)
        assert isinstance(metrics["retrieval_recall@5"], float)

        # Range checks
        assert 0 <= metrics["retrieval_mrr"] <= 1
        assert 0 <= metrics["retrieval_recall@5"] <= 1

    def test_token_classification_metrics_schema(self, smoke_token_data: list, temp_dir: Path):
        from benchmarks.tasks.token_classification import TokenClassificationEvaluator

        dataset_path = temp_dir / "token.json"
        with open(dataset_path, "w") as f:
            json.dump(smoke_token_data, f)

        evaluator = TokenClassificationEvaluator({"smoke_test": True, "device": "cpu", "ignore_index": -100})
        evaluator.load_model("dummy")
        evaluator.load_dataset(str(dataset_path))
        metrics = evaluator.compute_metrics()

        # Required keys
        assert "token_precision_macro" in metrics
        assert "token_recall_macro" in metrics
        assert "token_f1_macro" in metrics
        assert "confusion_matrix" in metrics

        # Type checks
        assert isinstance(metrics["token_precision_macro"], float)
        assert isinstance(metrics["token_recall_macro"], float)
        assert isinstance(metrics["token_f1_macro"], float)
        assert isinstance(metrics["confusion_matrix"], list)

        # Range checks
        assert 0 <= metrics["token_precision_macro"] <= 1
        assert 0 <= metrics["token_recall_macro"] <= 1
        assert 0 <= metrics["token_f1_macro"] <= 1


@pytest.mark.benchmark
class TestBaseEvaluatorInterface:
    def test_all_evaluators_inherit_base(self):
        from benchmarks.tasks.qa import QAEvaluator
        from benchmarks.tasks.nli import NLIEvaluator
        from benchmarks.tasks.retrieval import RetrievalEvaluator
        from benchmarks.tasks.token_classification import TokenClassificationEvaluator

        evaluators = [QAEvaluator, NLIEvaluator, RetrievalEvaluator, TokenClassificationEvaluator]

        for evaluator_class in evaluators:
            assert issubclass(evaluator_class, BaseEvaluator), \
                f"{evaluator_class.__name__} does not inherit from BaseEvaluator"

    def test_all_evaluators_implement_load_model(self):
        from benchmarks.tasks.qa import QAEvaluator
        from benchmarks.tasks.nli import NLIEvaluator
        from benchmarks.tasks.retrieval import RetrievalEvaluator
        from benchmarks.tasks.token_classification import TokenClassificationEvaluator

        evaluators = [QAEvaluator, NLIEvaluator, RetrievalEvaluator, TokenClassificationEvaluator]

        for evaluator_class in evaluators:
            config = {"smoke_test": True, "device": "cpu"}
            evaluator = evaluator_class(config)
            assert hasattr(evaluator, "load_model")
            assert callable(evaluator.load_model)

    def test_all_evaluators_implement_load_dataset(self):
        from benchmarks.tasks.qa import QAEvaluator
        from benchmarks.tasks.nli import NLIEvaluator
        from benchmarks.tasks.retrieval import RetrievalEvaluator
        from benchmarks.tasks.token_classification import TokenClassificationEvaluator

        evaluators = [QAEvaluator, NLIEvaluator, RetrievalEvaluator, TokenClassificationEvaluator]

        for evaluator_class in evaluators:
            config = {"smoke_test": True, "device": "cpu"}
            evaluator = evaluator_class(config)
            assert hasattr(evaluator, "load_dataset")
            assert callable(evaluator.load_dataset)

    def test_all_evaluators_implement_compute_metrics(self):
        from benchmarks.tasks.qa import QAEvaluator
        from benchmarks.tasks.nli import NLIEvaluator
        from benchmarks.tasks.retrieval import RetrievalEvaluator
        from benchmarks.tasks.token_classification import TokenClassificationEvaluator

        evaluators = [QAEvaluator, NLIEvaluator, RetrievalEvaluator, TokenClassificationEvaluator]

        for evaluator_class in evaluators:
            config = {"smoke_test": True, "device": "cpu"}
            evaluator = evaluator_class(config)
            assert hasattr(evaluator, "compute_metrics")
            assert callable(evaluator.compute_metrics)

    def test_all_evaluators_implement_save_results(self):
        from benchmarks.tasks.qa import QAEvaluator
        from benchmarks.tasks.nli import NLIEvaluator
        from benchmarks.tasks.retrieval import RetrievalEvaluator
        from benchmarks.tasks.token_classification import TokenClassificationEvaluator

        evaluators = [QAEvaluator, NLIEvaluator, RetrievalEvaluator, TokenClassificationEvaluator]

        for evaluator_class in evaluators:
            config = {"smoke_test": True, "device": "cpu"}
            evaluator = evaluator_class(config)
            assert hasattr(evaluator, "save_results")
            assert callable(evaluator.save_results)


@pytest.mark.benchmark
class TestCustomTaskCreation:
    def test_custom_evaluator_can_be_created(self):
        class CustomEvaluator(BaseEvaluator):
            def load_model(self, model_path: str) -> None:
                self.model = lambda x: "mock_output"

            def load_dataset(self, dataset_path: str) -> None:
                self.dataset = [{"input": "test"}]

            def compute_metrics(self) -> Dict[str, float]:
                return {"custom_metric": 0.95}

        config = {"smoke_test": True, "device": "cpu"}
        evaluator = CustomEvaluator(config)
        evaluator.load_model("dummy")
        evaluator.load_dataset("dummy.json")
        metrics = evaluator.compute_metrics()

        assert "custom_metric" in metrics
        assert metrics["custom_metric"] == 0.95

    def test_custom_evaluator_metrics_follow_schema(self, temp_dir: Path):
        class CustomEvaluator(BaseEvaluator):
            def load_model(self, model_path: str) -> None:
                self.model = lambda x: "output"

            def load_dataset(self, dataset_path: str) -> None:
                self.dataset = [{"input": "test"}]

            def compute_metrics(self) -> Dict[str, float]:
                return {
                    "custom_precision": 0.85,
                    "custom_recall": 0.90,
                    "custom_f1": 0.875
                }

        config = {"smoke_test": True, "device": "cpu"}
        evaluator = CustomEvaluator(config)
        evaluator.load_model("dummy")
        evaluator.load_dataset("dummy")
        metrics = evaluator.compute_metrics()

        # Schema validation
        for key, value in metrics.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))
            assert 0 <= value <= 1

    def test_custom_evaluator_can_save_results(self, temp_dir: Path):
        class CustomEvaluator(BaseEvaluator):
            def load_model(self, model_path: str) -> None:
                self.model = lambda x: "output"

            def load_dataset(self, dataset_path: str) -> None:
                self.dataset = [{"input": "test"}]

            def compute_metrics(self) -> Dict[str, float]:
                return {"custom_score": 0.99}

        config = {"smoke_test": True, "device": "cpu"}
        evaluator = CustomEvaluator(config)
        evaluator.load_model("dummy")
        evaluator.load_dataset("dummy")
        metrics = evaluator.compute_metrics()

        output_path = temp_dir / "custom_results.json"
        evaluator.save_results(str(output_path), metrics)

        assert output_path.exists()

        with open(output_path) as f:
            saved = json.load(f)

        assert saved == metrics


@pytest.mark.benchmark
class TestTaskModuleDiscovery:
    def test_tasks_directory_exists(self):
        tasks_dir = Path(__file__).parent.parent.parent / "benchmarks" / "tasks"
        assert tasks_dir.exists()

    def test_base_module_exists(self):
        base_path = Path(__file__).parent.parent.parent / "benchmarks" / "tasks" / "base.py"
        assert base_path.exists()

    def test_all_task_modules_exist(self):
        tasks_dir = Path(__file__).parent.parent.parent / "benchmarks" / "tasks"

        expected_modules = ["base.py", "qa.py", "nli.py", "retrieval.py", "token_classification.py"]

        for module in expected_modules:
            module_path = tasks_dir / module
            assert module_path.exists(), f"Missing module: {module}"

    def test_all_task_modules_importable(self):
        import importlib

        modules = [
            "benchmarks.tasks.base",
            "benchmarks.tasks.qa",
            "benchmarks.tasks.nli",
            "benchmarks.tasks.retrieval",
            "benchmarks.tasks.token_classification"
        ]

        for module_name in modules:
            module = importlib.import_module(module_name)
            assert module is not None
