import re
from typing import Dict, Any, List


class QueryComplexityAnalyzer:
    def __init__(self):
        self.analytical_patterns = [
            re.compile(pattern) for pattern in [
                r'\banalyze\b', r'\bevaluate\b', r'\bassess\b', r'\bjustify\b',
                r'\bimplications\b', r'\bconsequences\b', r'\brelationship\b',
                r'\bimpact\b', r'\bsignificance\b', r'\bcomprehensive\b', r'\bsynthesize\b'
            ]
        ]

        self.comparison_patterns = [
            re.compile(pattern) for pattern in [
                r'\bcompare\b', r'\bdifference between\b', r'\bvs\.?\b', r'\bversus\b', r'\bcontrast\b'
            ]
        ]

        self.factual_prefixes = [
            re.compile(pattern) for pattern in [
                r'^what is\b', r'^who is\b', r'^where is\b', r'^when is\b', r'^when did\b',
                r'^list\b', r'^find\b', r'^show\b', r'^get\b', r'^define\b'
            ]
        ]

        self.reasoning_prefixes = [
            re.compile(pattern) for pattern in [
                r'^why\b', r'^how\b'
            ]
        ]

        self.conditional_patterns = [
            re.compile(pattern) for pattern in [
                r'\bif\b', r'\bcondition\b', r'\bdepends\b'
            ]
        ]

        self.legal_complex_terms = [
            'liability', 'indemnification', 'jurisdiction', 'breach',
            'termination', 'arbitration', 'damages', 'covenant'
        ]

        self.financial_complex_terms = [
            'amortization', 'depreciation', 'valuation', 'derivatives',
            'liquidity', 'solvency', 'consolidated', 'hedging'
        ]

    def analyze(self, query: str, doc_type: str) -> Dict[str, Any]:
        """
        Analyze query complexity

        Args:
            query: User query
            doc_type: Document type

        Returns:
            Dictionary containing complexity score, level, and detected signals
        """
        if not query or not query.strip():
            return {
                "score": 0.0,
                "level": "low",
                "signals": ["empty_query"]
            }

        query_lower = query.lower().strip()
        score = 0.3
        signals: List[str] = []

        for pattern in self.factual_prefixes:
            if pattern.search(query_lower):
                signals.append("factual_prefix")
                score -= 0.15
                break

        for pattern in self.reasoning_prefixes:
            if pattern.search(query_lower):
                signals.append("reasoning_required")
                score += 0.1
                break

        analytical_matches = sum(1 for pattern in self.analytical_patterns if pattern.search(query_lower))
        if analytical_matches > 0:
            score += min(0.3, analytical_matches * 0.15)
            signals.append(f"analytical_keywords:{analytical_matches}")

        comparison_matches = sum(1 for pattern in self.comparison_patterns if pattern.search(query_lower))
        if comparison_matches > 0:
            score += 0.15
            signals.append("comparison")

        conditional_matches = sum(1 for pattern in self.conditional_patterns if pattern.search(query_lower))
        if conditional_matches > 0:
            score += 0.1
            signals.append(f"conditional_logic:{conditional_matches}")

        word_count = len(query_lower.split())
        if word_count > 25:
            score += 0.2
            signals.append("length:long")
        elif word_count > 15:
            score += 0.1
            signals.append("length:medium")
        elif word_count < 6:
            score -= 0.1
            signals.append("length:short")

        clauses = len(re.findall(r'[.?!]+', query))
        if clauses > 1:
            score += 0.15
            signals.append("multi_sentence")

        if doc_type == 'legal_contract':
            legal_count = sum(1 for term in self.legal_complex_terms if term in query_lower)
            if legal_count > 0:
                score += min(0.2, legal_count * 0.1)
                signals.append(f"domain_terms_legal:{legal_count}")
        elif doc_type == 'financial_report':
            financial_count = sum(1 for term in self.financial_complex_terms if term in query_lower)
            if financial_count > 0:
                score += min(0.2, financial_count * 0.1)
                signals.append(f"domain_terms_financial:{financial_count}")

        final_score = max(0.0, min(1.0, float(score)))

        if final_score < 0.4:
            level = "low"
        elif final_score < 0.7:
            level = "medium"
        else:
            level = "high"

        return {
            "score": round(final_score, 2),
            "level": level,
            "signals": signals
        }