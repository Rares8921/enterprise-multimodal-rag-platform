import math
import re
from collections import Counter
from typing import Any, Iterable


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
ACCESSION_PATTERN = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
SEC_TICKERS = {"AAPL", "AMZN", "JPM", "MSFT", "NVDA", "XOM"}

SEC_SECTION_PATTERNS: list[tuple[str, str, tuple[re.Pattern[str], ...]]] = [
    (
        "item_1a_risk_factors",
        "Item 1A Risk Factors",
        (
            re.compile(r"\bitem\s*1a\b", re.IGNORECASE),
            re.compile(r"\brisk\s+factors?\b", re.IGNORECASE),
        ),
    ),
    (
        "item_7a_market_risk",
        "Item 7A Quantitative and Qualitative Disclosures About Market Risk",
        (
            re.compile(r"\bitem\s*7a\b", re.IGNORECASE),
            re.compile(r"\bmarket\s+risk\b", re.IGNORECASE),
        ),
    ),
    (
        "item_9a_controls",
        "Item 9A Controls and Procedures",
        (
            re.compile(r"\bitem\s*9a\b", re.IGNORECASE),
            re.compile(r"\bcontrols?\s+and\s+procedures?\b", re.IGNORECASE),
        ),
    ),
    (
        "item_7_mda",
        "Item 7 Management's Discussion and Analysis",
        (
            re.compile(r"\bitem\s*7(?!\s*a|a)\b", re.IGNORECASE),
            re.compile(r"\bmanagement'?s?\s+discussion\b", re.IGNORECASE),
            re.compile(r"\bmd\s*&\s*a\b", re.IGNORECASE),
            re.compile(r"\bresults\s+of\s+operations\b", re.IGNORECASE),
        ),
    ),
    (
        "item_8_financial_statements",
        "Item 8 Financial Statements",
        (
            re.compile(r"\bitem\s*8\b", re.IGNORECASE),
            re.compile(r"\bfinancial\s+statements?\b", re.IGNORECASE),
        ),
    ),
    (
        "item_1_business",
        "Item 1 Business",
        (
            re.compile(r"\bitem\s*1(?!\s*a|a)\b", re.IGNORECASE),
            re.compile(r"\bbusiness\b", re.IGNORECASE),
        ),
    ),
]


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


def infer_sec_query_metadata(query: str) -> dict[str, Any]:
    """Infer SEC metadata from explicit query text only."""
    query = query or ""
    inferred: dict[str, Any] = {}

    for section_id, section_name, patterns in SEC_SECTION_PATTERNS:
        if any(pattern.search(query) for pattern in patterns):
            inferred["section_id"] = section_id
            inferred["section_name"] = section_name
            break

    upper_query = query.upper()
    for ticker in SEC_TICKERS:
        if re.search(rf"\b{re.escape(ticker)}\b", upper_query):
            inferred["ticker"] = ticker
            break

    accession_match = ACCESSION_PATTERN.search(query)
    if accession_match:
        inferred["accession_number"] = accession_match.group(0)

    year_match = YEAR_PATTERN.search(query)
    if year_match:
        inferred["filing_year"] = int(year_match.group(1))

    return inferred


def _metadata_bool(metadata: dict[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _normalized_metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if value is None:
        return ""
    return str(value).strip()


def sec_metadata_score(
    inferred: dict[str, Any],
    metadata: dict[str, Any],
    *,
    section_boost: float = 1.0,
    wrong_section_penalty: float = 0.25,
    unknown_section_penalty: float = 0.10,
    ticker_boost: float = 0.30,
    wrong_ticker_penalty: float = 0.35,
    accession_boost: float = 0.50,
    wrong_accession_penalty: float = 0.35,
    filing_year_boost: float = 0.20,
    wrong_filing_year_penalty: float = 0.20,
    table_of_contents_penalty: float = 0.60,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    desired_section = inferred.get("section_id")
    actual_section = _normalized_metadata_value(metadata, "section_id")
    if desired_section:
        if actual_section == desired_section:
            score += section_boost
            reasons.append("section_match")
        elif actual_section == "unknown":
            score -= unknown_section_penalty
            reasons.append("unknown_section")
        elif actual_section:
            score -= wrong_section_penalty
            reasons.append("section_mismatch")

        if _metadata_bool(metadata, "is_table_of_contents"):
            score -= table_of_contents_penalty
            reasons.append("table_of_contents_penalty")

    desired_ticker = inferred.get("ticker")
    actual_ticker = _normalized_metadata_value(metadata, "ticker").upper()
    if desired_ticker and actual_ticker:
        if actual_ticker == desired_ticker:
            score += ticker_boost
            reasons.append("ticker_match")
        else:
            score -= wrong_ticker_penalty
            reasons.append("ticker_mismatch")

    desired_accession = inferred.get("accession_number")
    actual_accession = _normalized_metadata_value(metadata, "accession_number")
    if desired_accession and actual_accession:
        if actual_accession == desired_accession:
            score += accession_boost
            reasons.append("accession_match")
        else:
            score -= wrong_accession_penalty
            reasons.append("accession_mismatch")

    desired_year = inferred.get("filing_year")
    actual_year = metadata.get("filing_year")
    if desired_year and actual_year:
        try:
            actual_year = int(actual_year)
        except (TypeError, ValueError):
            actual_year = None
        if actual_year == desired_year:
            score += filing_year_boost
            reasons.append("filing_year_match")
        elif actual_year:
            score -= wrong_filing_year_penalty
            reasons.append("filing_year_mismatch")

    return score, reasons


def sec_aware_rerank(
    query: str,
    matches: list[Any],
    *,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    metadata_weight: float = 0.5,
) -> list[dict[str, Any]]:
    if metadata_weight < 0:
        raise ValueError("SEC metadata weight must be non-negative")

    reranked = hybrid_rerank(
        query,
        matches,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )
    inferred = infer_sec_query_metadata(query)
    if not inferred or metadata_weight == 0:
        for match in reranked:
            match["sec_metadata_score"] = 0.0
            match["sec_metadata_reasons"] = []
            match["sec_aware_score"] = match["hybrid_score"]
            match["inferred_sec_query"] = inferred
        return reranked

    enriched = []
    for match in reranked:
        metadata = match.get("metadata") or {}
        metadata_score, reasons = sec_metadata_score(inferred, metadata)
        updated = dict(match)
        updated["sec_metadata_score"] = metadata_score
        updated["sec_metadata_reasons"] = reasons
        updated["sec_aware_score"] = match["hybrid_score"] + metadata_weight * metadata_score
        updated["inferred_sec_query"] = inferred
        enriched.append(updated)

    return sorted(enriched, key=lambda item: item["sec_aware_score"], reverse=True)
