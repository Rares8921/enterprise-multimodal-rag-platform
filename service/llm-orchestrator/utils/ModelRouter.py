from typing import Optional

from ..complexity_analyzer import QueryComplexityAnalyzer
from ..config import Settings

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def select_model(self, query: str, context_length: int, doc_type: str, force_model: Optional[str] = None) -> str:
        """
        Args:
            query: User query
            context_length: Length of context in tokens
            doc_type: Type of document
            force_model: Force specific model (for testing/comparison)

        Returns:
            Model name to use
        """
        if force_model and force_model != "auto":
            return force_model

        # Analyze query complexity
        complexity_score = self.complexity_analyzer.analyze(query, doc_type)

        if context_length > self.capabilities['mistral']['max_context']:
            # Context too large for Mistral
            decision = 'gemini'
            reason = 'context_size'
        elif complexity_score > 0.7:
            # High complexity needs stronger reasoning
            decision = 'gemini'
            reason = 'high_complexity'
        elif 'compare' in query.lower() or 'analyze' in query.lower() or 'explain' in query.lower():
            # Analytical queries benefit from Gemini
            decision = 'gemini'
            reason = 'analytical_query'
        else:
            # Simple queries
            decision = 'mistral'
            reason = 'cost_efficient'

        logger.info(f"Model routing: {decision} (reason: {reason}, complexity: {complexity_score:.2f})")

        return decision

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        #Estimate cost for a request
        cost_per_million = self.costs[model]
        cost = (input_tokens * cost_per_million['input'] +
                output_tokens * cost_per_million['output']) / 1_000_000
        return cost