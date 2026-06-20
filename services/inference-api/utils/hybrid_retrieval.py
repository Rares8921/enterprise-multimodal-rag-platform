import math
import re
from collections import Counter
from typing import Any, Iterable


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _metadata(match: Any) -> dict[str, Any]:
    metadata = _field(match, "metadata", {}) or {}
    if isinstance(metadata, dict):
        return metadata
    return dict(metadata)


def _as_dict(match: Any) -> dict[str, Any]:
    if isinstance(match, dict):
        return dict(match)
    return {
        "id": _field(match, "id"),
        "score": _field(match, "score", 0.0),
        "metadata": _metadata(match),
    }


def _normalize(values: Iterable[float]) -> list[float]:
    values = list(values)
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [1.0 if value > 0 else 0.0 for value in values]

    return [(value - min_value) / (max_value - min_value) for value in values]


def bm25_scores(query: str, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    query_terms = tokenize(query)
    tokenized_docs = [tokenize(doc) for doc in documents]
    if not query_terms or not tokenized_docs:
        return [0.0 for _ in documents]

    doc_count = len(tokenized_docs)
    avg_doc_len = sum(len(doc) for doc in tokenized_docs) / max(1, doc_count)
    doc_freq: Counter[str] = Counter()
    for doc in tokenized_docs:
        doc_freq.update(set(doc))

    scores: list[float] = []
    for doc in tokenized_docs:
        term_counts = Counter(doc)
        doc_len = len(doc) or 1
        score = 0.0
        for term in query_terms:
            if term_counts[term] == 0:
                continue
            idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            numerator = term_counts[term] * (k1 + 1)
            denominator = term_counts[term] + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
            score += idf * numerator / denominator
        scores.append(score)

    return scores


def hybrid_rerank(
    query: str,
    matches: list[Any],
    *,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict[str, Any]]:
    if not matches:
        return []
    if vector_weight < 0 or bm25_weight < 0:
        raise ValueError("Hybrid retrieval weights must be non-negative")
    if vector_weight == 0 and bm25_weight == 0:
        raise ValueError("At least one hybrid retrieval weight must be positive")

    total_weight = vector_weight + bm25_weight
    vector_weight = vector_weight / total_weight
    bm25_weight = bm25_weight / total_weight

    match_dicts = [_as_dict(match) for match in matches]
    vector_scores = [float(match.get("score") or 0.0) for match in match_dicts]
    documents = [str((match.get("metadata") or {}).get("text", "")) for match in match_dicts]
    lexical_scores = bm25_scores(query, documents)

    normalized_vector = _normalize(vector_scores)
    normalized_bm25 = _normalize(lexical_scores)

    reranked = []
    for match, vector_score, bm25_score, norm_vector, norm_bm25 in zip(
        match_dicts,
        vector_scores,
        lexical_scores,
        normalized_vector,
        normalized_bm25,
    ):
        hybrid_score = vector_weight * norm_vector + bm25_weight * norm_bm25
        enriched = dict(match)
        enriched["vector_score"] = vector_score
        enriched["bm25_score"] = bm25_score
        enriched["hybrid_score"] = hybrid_score
        reranked.append(enriched)

    return sorted(reranked, key=lambda item: item["hybrid_score"], reverse=True)
