import json
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score
from benchmarks.tasks.base import BaseEvaluator
from benchmarks.utils.helpers import get_logger, get_http_client

logger = get_logger(__name__)


class NLIEvaluator(BaseEvaluator):
    def load_model(self, model_path: str) -> None:
        logger.info(f"Initializing NLI API client targeting {model_path}")
        self.session = get_http_client()

        def api_caller(item: Dict[str, Any]) -> str:
            if self.smoke_test and model_path == "dummy":
                return item.get("mock_prediction", "neutral")

            payload = {
                "premise": item.get("premise", ""),
                "hypothesis": item.get("hypothesis", ""),
                "tenant_id": self.config.get("tenant_id", "benchmark_tenant")
            }

            headers = {"Content-Type": "application/json"}
            api_key = self.config.get("api_key")
            if api_key:
                headers["X-API-Key"] = api_key

            response = self.session.post(model_path, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json().get("label", "neutral")

        self.model = api_caller

    def load_dataset(self, dataset_path: str) -> None:
        logger.info(f"Loading NLI dataset from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        if self.smoke_test:
            self.dataset = self.dataset[:5]

    def compute_metrics(self) -> Dict[str, float]:
        if not self.dataset:
            raise ValueError("Dataset not loaded.")

        predictions = []
        references = []

        for idx, item in enumerate(self.dataset):
            try:
                predictions.append(self.model(item))
            except Exception as e:
                logger.error(f"Failed NLI inference on item {idx}: {e}")

                predictions.append("error_fallback")

            references.append(item.get("label", "neutral"))

        accuracy = float(accuracy_score(references, predictions))
        f1 = float(f1_score(references, predictions, average="macro", zero_division=0))

        metrics = {
            "nli_accuracy": accuracy,
            "nli_f1_macro": f1
        }
        logger.info(f"NLI Metrics: {metrics}")
        return metrics