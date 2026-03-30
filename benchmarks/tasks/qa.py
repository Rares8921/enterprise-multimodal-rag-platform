import json
from typing import Dict, Any
from benchmarks.tasks.base import BaseEvaluator
from benchmarks.utils.metrics import compute_f1, compute_exact_match
from benchmarks.utils.helpers import get_logger, get_http_client

logger = get_logger(__name__)


class QAEvaluator(BaseEvaluator):
    def load_model(self, model_path: str) -> None:
        logger.info(f"Initializing QA API client targeting {model_path}")
        self.session = get_http_client()

        def api_caller(item: Dict[str, Any]) -> str:
            if self.smoke_test and model_path == "dummy":
                return item.get("mock_prediction", "")

            payload = {
                "query": item.get("question", ""),
                "tenant_id": self.config.get("tenant_id", "benchmark_tenant"),
                "top_k": 5,
                "include_citations": False
            }

            headers = {"Content-Type": "application/json"}
            api_key = self.config.get("api_key")
            if api_key:
                headers["X-API-Key"] = api_key

            response = self.session.post(model_path, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json().get("answer", "")

        self.model = api_caller

    def load_dataset(self, dataset_path: str) -> None:
        logger.info(f"Loading QA dataset from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        if self.smoke_test:
            self.dataset = self.dataset[:5]

    def compute_metrics(self) -> Dict[str, float]:
        if not self.dataset:
            raise ValueError("Dataset not loaded.")

        f1_scores, em_scores = [], []

        for idx, item in enumerate(self.dataset):
            try:
                prediction = self.model(item)
                ground_truth = item.get("answer", "")

                f1_scores.append(compute_f1(prediction, ground_truth))
                em_scores.append(compute_exact_match(prediction, ground_truth))
            except Exception as e:
                logger.error(f"Failed QA inference on item {idx}: {e}")
                f1_scores.append(0.0)
                em_scores.append(0.0)

        metrics = {
            "qa_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "qa_exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0
        }
        logger.info(f"QA Metrics: {metrics}")
        return metrics