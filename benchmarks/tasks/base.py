import abc
import json
import os
from typing import Dict, Any
from benchmarks.utils.helpers import get_logger

logger = get_logger(__name__)

class BaseEvaluator(abc.ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smoke_test = config.get("smoke_test", False)
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 1)
        self.model = None
        self.dataset = None

    @abc.abstractmethod
    def load_model(self, model_path: str) -> None:
        pass

    @abc.abstractmethod
    def load_dataset(self, dataset_path: str) -> None:
        pass

    @abc.abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        pass

    def save_results(self, output_path: str, metrics: Dict[str, float]) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Results saved to {output_path}")