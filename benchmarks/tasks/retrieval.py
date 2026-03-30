import json
from typing import Dict, Any
from benchmarks.tasks.base import BaseEvaluator
from benchmarks.utils.metrics import compute_mrr, compute_recall_at_k
from benchmarks.utils.helpers import get_logger, get_http_client

logger = get_logger(__name__)


class RetrievalEvaluator(BaseEvaluator):
    def load_model(self, model_path: str) -> None:
        logger.info(f"Initializing Retrieval API client targeting {model_path}")
        self.session = get_http_client()

        def api_caller(item: Dict[str, Any]) -> list:
            if self.smoke_test and model_path == "dummy":
                return item.get("mock_retrieved_ids", [])

            payload = {
                "query": item.get("query", ""),
                "tenant_id": self.config.get("tenant_id", "benchmark_tenant"),
                "top_k": 10,
                "include_citations": True
            }

            headers = {"Content-Type": "application/json"}
            api_key = self.config.get("api_key")
            if api_key:
                headers["X-API-Key"] = api_key

            response = self.session.post(model_path, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            # id of docs from citations block
            citations = response.json().get("citations", [])
            return [cit.get("doc_id") for cit in citations if "doc_id" in cit]

        self.model = api_caller

    def load_dataset(self, dataset_path: str) -> None:
        logger.info(f"Loading Retrieval dataset from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        if self.smoke_test:
            self.dataset = self.dataset[:5]

    def compute_metrics(self) -> Dict[str, float]:
        if not self.dataset:
            raise ValueError("Dataset not loaded.")

        relevant, retrieved = [], []

        for idx, item in enumerate(self.dataset):
            try:
                relevant.append(item.get("relevant_ids", []))
                retrieved.append(self.model(item))
            except Exception as e:
                logger.error(f"Failed Retrieval inference on item {idx}: {e}")
                relevant.append(item.get("relevant_ids", []))
                retrieved.append([])

        metrics = {
            "retrieval_mrr": compute_mrr(relevant, retrieved),
            "retrieval_recall@5": compute_recall_at_k(relevant, retrieved, k=5)
        }
        logger.info(f"Retrieval Metrics: {metrics}")
        return metrics