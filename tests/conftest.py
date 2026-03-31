import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after test."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_dataset_dict() -> Dict[str, Any]:
    """Valid dataset with train/validation/test splits for LayoutLM."""
    base_sample = {
        "image": "base64_placeholder_image_data",
        "words": ["Contract", "Term", "30", "days"],
        "boxes": [[0, 0, 100, 50], [100, 0, 200, 50], [200, 0, 250, 50], [250, 0, 320, 50]],
        "labels": [0, 1, 2, 0]
    }
    return {
        "train": {
            "image": [base_sample["image"]] * 10,
            "words": [base_sample["words"]] * 10,
            "boxes": [base_sample["boxes"]] * 10,
            "labels": [base_sample["labels"]] * 10
        },
        "validation": {
            "image": [base_sample["image"]] * 5,
            "words": [base_sample["words"]] * 5,
            "boxes": [base_sample["boxes"]] * 5,
            "labels": [base_sample["labels"]] * 5
        },
        "test": {
            "image": [base_sample["image"]] * 5,
            "words": [base_sample["words"]] * 5,
            "boxes": [base_sample["boxes"]] * 5,
            "labels": [base_sample["labels"]] * 5
        }
    }


@pytest.fixture
def sample_dataset_flat() -> Dict[str, Any]:
    """Valid flat dataset (no splits) for LayoutLM."""
    base_sample = {
        "image": "base64_placeholder_image_data",
        "words": ["Invoice", "Total", "500", "USD"],
        "boxes": [[0, 0, 100, 50], [100, 0, 200, 50], [200, 0, 250, 50], [250, 0, 320, 50]],
        "labels": [0, 1, 2, 3]
    }
    return {
        "image": [base_sample["image"]] * 15,
        "words": [base_sample["words"]] * 15,
        "boxes": [base_sample["boxes"]] * 15,
        "labels": [base_sample["labels"]] * 15
    }


@pytest.fixture
def invalid_dataset_missing_keys() -> Dict[str, Any]:
    return {
        "train": {
            "image": ["img1", "img2"],
            "words": [["word1"], ["word2"]],
            # missing 'boxes' and 'labels'
        }
    }


@pytest.fixture
def sample_dataset_file(temp_dir: Path, sample_dataset_dict: Dict[str, Any]) -> Path:
    """Create a sample dataset JSON file."""
    path = temp_dir / "dataset.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_dataset_dict, f)
    return path


@pytest.fixture
def mock_mlflow():
    with patch("mlflow.set_tracking_uri") as mock_uri, \
            patch("mlflow.set_experiment") as mock_exp, \
            patch("mlflow.start_run") as mock_run, \
            patch("mlflow.log_params") as mock_params, \
            patch("mlflow.log_metrics") as mock_metrics, \
            patch("mlflow.set_tag") as mock_tag, \
            patch("mlflow.pytorch.log_model") as mock_log_model, \
            patch("mlflow.register_model") as mock_register:
        mock_run_instance = MagicMock()
        mock_run_instance.info.run_id = "test-run-id-12345"
        mock_run.return_value.__enter__ = MagicMock(return_value=mock_run_instance)
        mock_run.return_value.__exit__ = MagicMock(return_value=False)

        yield {
            "set_tracking_uri": mock_uri,
            "set_experiment": mock_exp,
            "start_run": mock_run,
            "log_params": mock_params,
            "log_metrics": mock_metrics,
            "set_tag": mock_tag,
            "log_model": mock_log_model,
            "register_model": mock_register,
            "run_instance": mock_run_instance
        }


@pytest.fixture
def sample_metrics() -> Dict[str, Any]:
    return {
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.91,
        "f1": 0.90,
        "confusion_matrix": [[10, 2], [1, 12]],
        "per_class_metrics": {
            "class_0": {"precision": 0.91, "recall": 0.83, "f1": 0.87, "support": 12},
            "class_1": {"precision": 0.86, "recall": 0.92, "f1": 0.89, "support": 13}
        }
    }


@pytest.fixture
def sample_metrics_file(temp_dir: Path, sample_metrics: Dict[str, Any]) -> Path:
    path = temp_dir / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_metrics, f)
    return path


@pytest.fixture
def smoke_qa_data() -> list:
    return [
        {
            "id": "1",
            "context": "The standard termination clause requires a 30-day notice period.",
            "question": "How many days notice is required for termination?",
            "answer": "30-day",
            "mock_prediction": "30-day"
        },
        {
            "id": "2",
            "context": "The liability cap is set at 50000 USD for standard breaches.",
            "question": "What is the liability cap?",
            "answer": "50000 USD",
            "mock_prediction": "50000"
        }
    ]


@pytest.fixture
def smoke_nli_data() -> list:
    return [
        {
            "id": "1",
            "premise": "All employees must sign the NDA before starting work.",
            "hypothesis": "New employees need to sign confidentiality agreements.",
            "label": "entailment",
            "mock_prediction": "entailment"
        },
        {
            "id": "2",
            "premise": "The contract expires on December 31, 2024.",
            "hypothesis": "The contract has no expiration date.",
            "label": "contradiction",
            "mock_prediction": "contradiction"
        },
        {
            "id": "3",
            "premise": "Payment terms are net 30.",
            "hypothesis": "The company uses cloud storage.",
            "label": "neutral",
            "mock_prediction": "neutral"
        }
    ]


@pytest.fixture
def smoke_retrieval_data() -> list:
    return [
        {
            "id": "1",
            "query": "Termination notice period rules",
            "relevant_ids": ["doc_term_01", "doc_term_02"],
            "mock_retrieved_ids": ["doc_term_01", "doc_random_99", "doc_term_02", "doc_random_01", "doc_random_02"]
        },
        {
            "id": "2",
            "query": "Standard breaches liability cap",
            "relevant_ids": ["doc_liab_10"],
            "mock_retrieved_ids": ["doc_liab_10", "doc_liab_11", "doc_random_03", "doc_random_04", "doc_random_05"]
        }
    ]


@pytest.fixture
def smoke_token_data() -> list:
    return [
        {
            "id": "1",
            "text": "The standard termination clause requires a 30-day notice period.",
            "tokens": ["The", "standard", "termination", "clause", "requires", "a", "30-day", "notice", "period", "."],
            "labels": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "mock_predictions": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        },
        {
            "id": "2",
            "text": "The liability cap is set at 50000 USD for standard breaches.",
            "tokens": ["The", "liability", "cap", "is", "set", "at", "50000", "USD", "for", "standard", "breaches",
                       "."],
            "labels": [0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
            "mock_predictions": [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
        }
    ]


@pytest.fixture
def benchmark_config() -> Dict[str, Any]:
    return {
        "global": {
            "seed": 1235,
            "smoke_test": True,
            "device": "cpu",
            "batch_size": 4,
            "output_dir": "benchmarks/results",
            "mlflow_tracking_uri": "",
            "experiment_name": "test_benchmarks"
        },
        "tasks": {
            "test_qa": {
                "module": "qa",
                "evaluator_class": "QAEvaluator",
                "model_path": "dummy",
                "dataset_path": "benchmarks/data_samples/smoke_qa.json"
            }
        }
    }


# Markers
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "benchmark: Benchmark/evaluation tests")
    config.addinivalue_line("markers", "smoke: Smoke tests (quick validation)")
    config.addinivalue_line("markers", "reproducibility: Tests for reproducible results")
