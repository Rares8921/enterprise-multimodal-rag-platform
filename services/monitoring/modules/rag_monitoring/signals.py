from __future__ import annotations

import re
from typing import Any, Dict, Optional


_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+", re.MULTILINE)
_BRACKET_CIT_RE = re.compile(r"\[(\d{1,4})\]")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_SOURCE_RE = re.compile(r"\bsource(s)?\b\s*:", re.IGNORECASE)


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"1", "true", "yes", "y"}:
            return True
        if v in {"0", "false", "no", "n"}:
            return False
    return None


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p and p.strip()]
    return parts


def _citation_stats_from_text(answer_text: str) -> Dict[str, Any]:
    sents = _split_sentences(answer_text)
    if not sents:
        return {
            "rag_answer_sentences": 0,
            "rag_citations_count": 0,
            "rag_has_citations": False,
            "rag_citation_coverage": None,
        }

    cited_sentences = 0
    citations_total = 0

    for s in sents:
        bracket = len(_BRACKET_CIT_RE.findall(s))
        url = len(_URL_RE.findall(s))
        src = 1 if _SOURCE_RE.search(s) else 0
        citations_in_sentence = bracket + url + src
        citations_total += citations_in_sentence
        if citations_in_sentence > 0:
            cited_sentences += 1

    coverage = cited_sentences / max(len(sents), 1)
    return {
        "rag_answer_sentences": len(sents),
        "rag_citations_count": citations_total,
        "rag_has_citations": citations_total > 0,
        "rag_citation_coverage": float(coverage),
    }


def extract_rag_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract/compute RAG monitoring fields.

    Preference order:
      1) explicit `rag_*` fields
      2) derived from `citations` list
      3) derived from `answer_text`

    Note: we intentionally do NOT persist raw answer text.
    """

    out: Dict[str, Any] = {
        "rag_answer_sentences": _safe_int(payload.get("rag_answer_sentences")),
        "rag_citations_count": _safe_int(payload.get("rag_citations_count")),
        "rag_has_citations": _safe_bool(payload.get("rag_has_citations")),
        "rag_citation_coverage": _safe_float(payload.get("rag_citation_coverage")),
        "rag_groundedness_score": _safe_float(
            payload.get("rag_groundedness_score", payload.get("groundedness_score"))
        ),
    }

    citations = payload.get("citations")
    if out["rag_citations_count"] is None and isinstance(citations, list):
        out["rag_citations_count"] = len(citations)
        if out["rag_has_citations"] is None:
            out["rag_has_citations"] = len(citations) > 0

    answer_text = payload.get("answer_text") or payload.get("rag_answer_text")
    if isinstance(answer_text, str) and (
        out["rag_answer_sentences"] is None
        or out["rag_citations_count"] is None
        or out["rag_has_citations"] is None
        or out["rag_citation_coverage"] is None
    ):
        derived = _citation_stats_from_text(answer_text)
        for k, v in derived.items():
            if out.get(k) is None:
                out[k] = v

    return out
