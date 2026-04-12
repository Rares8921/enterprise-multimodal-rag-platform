from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

REQUIRED_KEYS: Tuple[str, ...] = ("image", "words", "boxes", "labels")


def _ensure_dict(obj: Any, *, what: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{what} must be a dict")
    return obj


def _validate_required_keys(sample: Mapping[str, Any], required_keys: Iterable[str]) -> None:
    for key in required_keys:
        if key not in sample:
            raise KeyError(f"Missing required key '{key}'")


def validate_dataset_json_schema(data: Any, required_keys: Iterable[str] = REQUIRED_KEYS) -> str:
    """Validate the expected dataset JSON schema.

    Supports either:
      - split dict: {"train": {key: [..], ...}, "validation": ..., "test": ...}
      - flat dict: {key: [..], ...}

    Returns: "split" or "flat".
    """
    data_dict = _ensure_dict(data, what="dataset")

    if "train" in data_dict and isinstance(data_dict.get("train"), dict):
        # Split dict
        for split_name, split_data in data_dict.items():
            if not isinstance(split_data, dict):
                raise ValueError(f"Split '{split_name}' must be a dict")
            _validate_required_keys(split_data, required_keys)
        return "split"

    # Flat dict
    _validate_required_keys(data_dict, required_keys)
    return "flat"


def subset_dataset_json_for_smoke_test(data: Dict[str, Any], max_samples: int = 10) -> Dict[str, Any]:
    """Subset a dataset JSON payload to a maximum number of samples per split."""
    data_dict = _ensure_dict(data, what="dataset")

    if "train" in data_dict and isinstance(data_dict.get("train"), dict):
        subset: Dict[str, Any] = {}
        for split_name, split_data in data_dict.items():
            split_data = _ensure_dict(split_data, what=f"split '{split_name}'")
            subset[split_name] = {k: (v[:max_samples] if isinstance(v, list) else v) for k, v in split_data.items()}
        return subset

    # Flat dict
    return {k: (v[:max_samples] if isinstance(v, list) else v) for k, v in data_dict.items()}


def build_training_args(
    *,
    smoke_test: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    has_eval_dataset: Optional[bool] = None,
    eval_dataset: Any = None,
    cuda_available: bool = False,
) -> Dict[str, Any]:
    """Build TrainingArguments kwargs (pure logic; no transformers import).

    Backward-compatible: unit tests may pass eval_dataset=..., while the pipeline passes has_eval_dataset=...
    """
    if has_eval_dataset is None:
        has_eval_dataset = bool(eval_dataset)

    return {
        "num_train_epochs": 1 if smoke_test else epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 1 if smoke_test else 2,
        "learning_rate": learning_rate,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch" if has_eval_dataset else "no",
        "load_best_model_at_end": True if has_eval_dataset else False,
        "metric_for_best_model": "f1" if has_eval_dataset else None,
        "save_total_limit": 3,
        "report_to": ["mlflow"],
        "fp16": cuda_available,
        "max_steps": 5 if smoke_test else -1,
    }


def validate_hyperparameters(params: Mapping[str, Any]) -> bool:
    """Return True if hyperparameters are in reasonable ranges."""
    try:
        epochs = int(params.get("epochs", 0))
        batch_size = int(params.get("batch_size", 0))
        learning_rate = float(params.get("learning_rate", 0))
        num_labels = int(params.get("num_labels", 0))
    except (TypeError, ValueError):
        return False

    if epochs <= 0:
        return False
    if batch_size <= 0:
        return False
    if not (0 < learning_rate < 1):
        return False
    if num_labels <= 0:
        return False
    return True


def should_resume_from_checkpoint(checkpoint_dir: Path) -> bool:
    return checkpoint_dir.exists() and any(checkpoint_dir.iterdir())


def load_metrics_from_file(metrics_path: Path) -> Dict[str, Any]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_accuracy_from_metrics(metrics: Mapping[str, Any]) -> float:
    accuracy = metrics.get("accuracy", 0.0)
    if accuracy is None:
        return 0.0
    try:
        return float(accuracy)
    except (TypeError, ValueError):
        return 0.0


def determine_registration_status(accuracy: float, min_accuracy: float) -> str:
    return "registered" if accuracy >= min_accuracy else "rejected"


def build_model_uri(model_version: str) -> str:
    return f"runs:/{model_version}/model"


def build_registered_model_name(experiment_name: str) -> str:
    return f"{experiment_name}-layoutlm"


def build_model_tags(accuracy: float, status: str) -> Dict[str, str]:
    return {"accuracy": str(accuracy), "status": status}
