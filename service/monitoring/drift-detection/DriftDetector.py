from typing import Dict, List, Any
from scipy import stats
import numpy as np

class DriftDetector:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

    async def detect_data_drift(self, feature_name: str, current_data: List[float], reference_data: List[float]) -> Dict[str, Any]:
        # PSI as primary, K-S as auxiliary
        if not current_data or not reference_data:
            return {'drift_detected': False, 'psi': 0.0, 'severity': 'none'}

        psi = self._calculate_psi(reference_data, current_data)
        statistic, p_value = stats.ks_2samp(reference_data, current_data)

        drift_detected = psi > 0.2

        return {
            'feature': feature_name,
            'drift_detected': bool(drift_detected),
            'ks_statistic': float(statistic),
            'p_value': float(p_value),
            'psi': float(psi),
            'severity': 'high' if psi > 0.2 else 'medium' if psi > 0.1 else 'low'
        }

    def _calculate_psi(self, reference: List[float], current: List[float], bins: int = 10) -> float:
        # population stability index
        ref_hist, bin_edges = np.histogram(reference, bins=bins)
        curr_hist, _ = np.histogram(current, bins=bin_edges)

        ref_pct = ref_hist / len(reference)
        curr_pct = curr_hist / len(current)

        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)

        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi)

    async def detect_concept_drift(self, recent_feedback: float, baseline_feedback: float, threshold: float = 0.05) -> Dict[str, Any]:
        # concept drift by monitoring actual ground truth / user feedback
        accuracy_drop = baseline_feedback - recent_feedback
        drift_detected = accuracy_drop > threshold

        return {
            'drift_detected': bool(drift_detected),
            'accuracy_drop': float(accuracy_drop),
            'recent_accuracy': float(recent_feedback),
            'baseline_accuracy': float(baseline_feedback),
            'severity': 'critical' if accuracy_drop > 0.10 else 'warning'
        }

    async def detect_embedding_drift(self, recent_embeddings: np.ndarray, reference_embeddings: np.ndarray) -> Dict[str, Any]:
        # drift in embedding space using centroid dist
        if len(recent_embeddings) == 0 or len(reference_embeddings) == 0:
            return {'drift_detected': False, 'centroid_distance': 0.0}

        curr_centroid = np.mean(recent_embeddings, axis=0)
        ref_centroid = np.mean(reference_embeddings, axis=0)

        curr_norm = curr_centroid / (np.linalg.norm(curr_centroid) + 1e-10)
        ref_norm = ref_centroid / (np.linalg.norm(ref_centroid) + 1e-10)

        cosine_sim = np.dot(curr_norm, ref_norm)
        cosine_distance = 1.0 - cosine_sim

        drift_detected = cosine_distance > 0.15

        return {
            'drift_detected': bool(drift_detected),
            'centroid_distance': float(cosine_distance),
            'recent_variance': float(np.var(recent_embeddings)),
            'reference_variance': float(np.var(reference_embeddings))
        }