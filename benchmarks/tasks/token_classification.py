import json
from typing import Dict, Any
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from benchmarks.tasks.base import BaseEvaluator
from benchmarks.utils.helpers import get_logger, get_http_client

logger = get_logger(__name__)


class TokenClassificationEvaluator(BaseEvaluator):
    def load_model(self, model_path: str) -> None:
        logger.info(f"Initializing Token Classification API client targeting {model_path}")
        self.session = get_http_client()

        def api_caller(item: Dict[str, Any]) -> list:
            if self.smoke_test and model_path in ["dummy", "layoutlm-v3-finetuned"]:
                return item.get("mock_predictions", [])

            payload = {
                "text": item.get("text", ""),
                "tokens": item.get("tokens", []),
                "tenant_id": self.config.get("tenant_id", "benchmark_tenant")
            }

            headers = {"Content-Type": "application/json"}
            api_key = self.config.get("api_key")
            if api_key:
                headers["X-API-Key"] = api_key

            response = self.session.post(model_path, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json().get("predictions", [])

        self.model = api_caller

    def load_dataset(self, dataset_path: str) -> None:
        logger.info(f"Loading Token Classification dataset from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        if self.smoke_test:
            self.dataset = self.dataset[:5]

    def compute_metrics(self) -> Dict[str, Any]:
        if not self.dataset:
            raise ValueError("Dataset not loaded.")

        flat_preds = []
        flat_refs = []
        ignore_index = self.config.get("ignore_index", -100)

        for idx, item in enumerate(self.dataset):
            try:
                preds = self.model(item)
                refs = item.get("labels", [])

                # too few tokens -> penalty
                if len(preds) < len(refs):
                    preds.extend([-1] * (len(refs) - len(preds)))
                elif len(preds) > len(refs):
                    preds = preds[:len(refs)]

                for p, r in zip(preds, refs):
                    if r != ignore_index:
                        flat_preds.append(p)
                        flat_refs.append(r)
            except Exception as e:
                logger.error(f"Failed Token Classification inference on item {idx}: {e}")

        if not flat_refs:
            logger.warning("No valid references found or all API calls failed.")
            return {"token_precision_macro": 0.0, "token_recall_macro": 0.0, "token_f1_macro": 0.0,
                    "confusion_matrix": []}

        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_refs, flat_preds, average="macro", zero_division=0
        )

        cm = confusion_matrix(flat_refs, flat_preds)

        metrics = {
            "token_precision_macro": float(precision),
            "token_recall_macro": float(recall),
            "token_f1_macro": float(f1),
            "confusion_matrix": cm.tolist()
        }
        logger.info(f"Token Classification Metrics: F1 Macro = {metrics['token_f1_macro']}")
        return metrics