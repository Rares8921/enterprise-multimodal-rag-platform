from dataclasses import dataclass
from typing import Optional

from complexity_analyzer import ComplexityResult, QueryComplexityAnalyzer
from config import Settings

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    model: str
    reason: str
    complexity: ComplexityResult
    context_length: int


class ModelRouter:
    def __init__(self, settings: Settings, complexity_analyzer: QueryComplexityAnalyzer):
        self.settings = settings
        self.complexity_analyzer = complexity_analyzer

        # Cost per 1M tokens
        self.costs = {
            'gemini': {'input': 3.50, 'output': 10.50},  # Gemini 1.5 Pro
            'mistral': {'input': 0.70, 'output': 2.10}  # Mistral 8x7B (self-hosted)
        }

        # Model capabilities
        self.capabilities = {
            'gemini': {
                'max_context': 128000,
                'reasoning_score': 0.95,
                'speed_score': 0.75
            },
            'mistral': {
                'max_context': 32000,
                'reasoning_score': 0.80,
                'speed_score': 0.90
            }
        }

    def route(self, query: str, context_length: int, doc_type: str, force_model: Optional[str] = None) -> RoutingDecision:
        """
        Args:
            query: User query
            context_length: Length of context in tokens
            doc_type: Type of document
            force_model: Force specific model (for testing/comparison)

        Returns:
            Routing decision with selected model, reason, and complexity analysis
        """
        if force_model and force_model != "auto":
            if force_model not in self.costs:
                raise ValueError(f"Unsupported forced model: {force_model}")
            complexity = self.complexity_analyzer.analyze(query, doc_type)
            return RoutingDecision(
                model=force_model,
                reason="forced",
                complexity=complexity,
                context_length=max(0, int(context_length)),
            )

        # Analyze query complexity
        complexity = self.complexity_analyzer.analyze(query, doc_type)
        safe_context_length = max(0, int(context_length))
        threshold = getattr(self.settings, "complexity_threshold", 0.7)
        query_lower = (query or "").lower()

        if safe_context_length > self.capabilities['mistral']['max_context']:
            # Context too large for Mistral
            decision = 'gemini'
            reason = 'context_size'
        elif complexity.score >= threshold:
            # High complexity needs stronger reasoning
            decision = 'gemini'
            reason = 'high_complexity'
        elif 'compare' in query_lower or 'analyze' in query_lower or 'explain' in query_lower:
            # Analytical queries benefit from Gemini
            decision = 'gemini'
            reason = 'analytical_query'
        else:
            # Simple queries
            decision = 'mistral'
            reason = 'cost_efficient'

        logger.info(
            "Model routing: %s (reason: %s, complexity: %.2f, level: %s)",
            decision,
            reason,
            complexity.score,
            complexity.level,
        )

        return RoutingDecision(
            model=decision,
            reason=reason,
            complexity=complexity,
            context_length=safe_context_length,
        )

    def select_model(self, query: str, context_length: int, doc_type: str, force_model: Optional[str] = None) -> str:
        return self.route(query, context_length, doc_type, force_model).model

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        # Estimate cost for a request
        if model not in self.costs:
            raise ValueError(f"Unsupported model for cost estimation: {model}")

        cost_per_million = self.costs[model]
        safe_input_tokens = max(0, int(input_tokens))
        safe_output_tokens = max(0, int(output_tokens))
        cost = (safe_input_tokens * cost_per_million['input'] +
                safe_output_tokens * cost_per_million['output']) / 1_000_000
        return cost
