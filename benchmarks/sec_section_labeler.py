"""Generate conservative SEC 10-K section labels for rendered public filings.

The output is intended for retrieval evaluation. It records approximate page ranges
for common 10-K sections from rendered PDF text and does not store raw filing text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.corpus_manifest import (  # noqa: E402
    LoadedCorpus,
    load_corpus_manifest,
    validate_manifest_payload,
)

DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "corpora" / "sec_edgar_rendered_manifest.generated.json"
DEFAULT_PDF_ROOT = REPO_ROOT / "benchmarks" / "corpora" / "local_pdfs"
DEFAULT_LABELS_OUT = REPO_ROOT / "benchmarks" / "corpora" / "sec_edgar_section_labels.generated.json"
DEFAULT_MANIFEST_OUT = REPO_ROOT / "benchmarks" / "corpora" / "sec_edgar_section_manifest.generated.json"


@dataclass(frozen=True)
class SectionSpec:
    section_id: str
    section_name: str
    heading_pattern: re.Pattern[str]
    query_phrase: str
    expected_hints: list[str]


SECTION_SPECS = [
    SectionSpec(
        section_id="item_1_business",
        section_name="Item 1. Business",
        heading_pattern=re.compile(r"\bItem\s+1\.\s+Business\b", re.IGNORECASE),
        query_phrase="Item 1 Business",
        expected_hints=["business"],
    ),
    SectionSpec(
        section_id="item_1a_risk_factors",
        section_name="Item 1A. Risk Factors",
        heading_pattern=re.compile(r"\bItem\s+1A\.\s+Risk\s+Factors\b", re.IGNORECASE),
        query_phrase="Item 1A Risk Factors",
        expected_hints=["risk factors"],
    ),
    SectionSpec(
        section_id="item_7_mda",
        section_name="Item 7. Management's Discussion and Analysis",
        heading_pattern=re.compile(
            r"\bItem\s+7\.\s+Management'?s\s+Discussion\s+and\s+Analysis\b",
            re.IGNORECASE,
        ),
        query_phrase="Item 7 Management's Discussion and Analysis",
        expected_hints=["management", "discussion", "analysis"],
    ),
    SectionSpec(
        section_id="item_7a_market_risk",
        section_name="Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
        heading_pattern=re.compile(
            r"\bItem\s+7A\.\s+Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk\b",
            re.IGNORECASE,
        ),
        query_phrase="Item 7A market risk disclosures",
        expected_hints=["market risk"],
    ),
    SectionSpec(
        section_id="item_8_financial_statements",
        section_name="Item 8. Financial Statements",
        heading_pattern=re.compile(r"\bItem\s+8\.\s+Financial\s+Statements\b", re.IGNORECASE),
        query_phrase="Item 8 Financial Statements",
        expected_hints=["financial statements"],
    ),
    SectionSpec(
        section_id="item_9a_controls",
        section_name="Item 9A. Controls and Procedures",
        heading_pattern=re.compile(r"\bItem\s+9A\.\s+Controls\s+and\s+Procedures\b", re.IGNORECASE),
        query_phrase="Item 9A Controls and Procedures",
        expected_hints=["controls", "procedures"],
    ),
]

SECTION_BY_ID = {spec.section_id: spec for spec in SECTION_SPECS}
LIMITATIONS = [
    "Section labels are generated from rendered SEC PDF text with conservative heading regexes.",
    "Page ranges are approximate and should be interpreted as retrieval labels, not legal or financial correctness labels.",
    "The extractor skips table-of-contents hits when detected, but rendered HTML/PDF quirks may still affect page mapping.",
    "No raw SEC filing text is written to the generated labels or manifest.",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_section_labels(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("SEC section labeling requires pypdf") from exc

    reader = PdfReader(str(pdf_path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return extract_section_labels_from_pages(pages)


def extract_section_labels_from_pages(pages: list[str]) -> list[dict[str, Any]]:
    starts: dict[str, int] = {}
    for page_number, page_text in enumerate(pages, start=1):
        if _looks_like_table_of_contents(page_text):
            continue
        normalized = _normalize_text(page_text)
        for spec in SECTION_SPECS:
            if spec.section_id in starts:
                continue
            if spec.heading_pattern.search(normalized):
                starts[spec.section_id] = page_number

    found = [
        {
            "section_id": spec.section_id,
            "section_name": spec.section_name,
            "start_page": starts[spec.section_id],
        }
        for spec in SECTION_SPECS
        if spec.section_id in starts
    ]
    found.sort(key=lambda item: (item["start_page"], _section_order(item["section_id"])))

    page_count = len(pages)
    labels: list[dict[str, Any]] = []
    for index, item in enumerate(found):
        start_page = item["start_page"]
        end_page = _conservative_end_page(item, found[index + 1] if index + 1 < len(found) else None, page_count)
        confidence = "high" if end_page > start_page or item["section_id"] == "item_9a_controls" else "medium"
        labels.append({
            **item,
            "end_page": min(end_page, page_count),
            "confidence": confidence,
            "method": "regex_heading_in_rendered_pdf_text",
            "limitations": LIMITATIONS,
        })
    return labels


def build_section_manifest_payload(loaded: LoadedCorpus, labels_by_document: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    source_payload = json.loads(loaded.manifest_path.read_text(encoding="utf-8-sig"))
    source_payload["corpus_id"] = f"{source_payload['corpus_id']}_section_eval"
    source_payload["corpus_name"] = f"{source_payload['corpus_name']} - Section Retrieval Evaluation"
    source_metadata = source_payload.setdefault("source_metadata", {})
    source_metadata["section_labeling"] = {
        "generated_at_utc": utc_timestamp(),
        "method": "conservative regex heading extraction over rendered SEC PDF text",
        "section_ids": [spec.section_id for spec in SECTION_SPECS],
        "limitations": LIMITATIONS,
    }

    for document in source_payload["documents"]:
        labels = labels_by_document.get(document["document_id"], [])
        metadata = document.setdefault("source_metadata", {})
        metadata["sec_section_labels"] = labels
        metadata["sec_section_label_count"] = len(labels)

    source_payload["queries"] = _section_queries(source_payload["documents"], labels_by_document)
    validate_manifest_payload(source_payload)
    return source_payload


def build_labels_artifact(loaded: LoadedCorpus, labels_by_document: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "artifact_name": "sec_10k_section_labels",
        "generated_at_utc": utc_timestamp(),
        "source_manifest": str(loaded.manifest_path),
        "document_count": len(loaded.manifest.documents),
        "section_label_count": sum(len(labels) for labels in labels_by_document.values()),
        "method": "conservative regex heading extraction over rendered SEC PDF text",
        "limitations": LIMITATIONS,
        "documents": [
            {
                "document_id": document.document_id,
                "filename": document.filename,
                "sections": labels_by_document.get(document.document_id, []),
            }
            for document in loaded.manifest.documents
        ],
    }


def _section_queries(documents: list[dict[str, Any]], labels_by_document: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for document in documents:
        document_id = document["document_id"]
        metadata = document.get("source_metadata", {})
        filing_label = _filing_label(metadata)
        for label in labels_by_document.get(document_id, []):
            spec = SECTION_BY_ID.get(label["section_id"])
            if not spec:
                continue
            query_id = f"{document_id}_{label['section_id']}"
            pages = list(range(label["start_page"], label["end_page"] + 1))
            queries.append({
                "query_id": query_id,
                "query": f"In {filing_label}, where is {spec.query_phrase} discussed?",
                "category": f"sec_section_{label['section_id']}",
                "target_document_ids": [document_id],
                "relevant_pages": pages,
                "relevant_sections": [label["section_id"]],
                "relevant_chunk_ids": [],
                "expected_answer_hints": spec.expected_hints,
                "citation_required": True,
            })
    return queries


def _filing_label(metadata: dict[str, Any]) -> str:
    ticker = metadata.get("ticker") or "the company"
    form_type = metadata.get("form_type") or "10-K"
    filing_date = metadata.get("filing_date")
    accession = metadata.get("accession_number")
    parts = [str(ticker), str(form_type)]
    if filing_date:
        parts.append(str(filing_date))
    if accession:
        parts.append(str(accession))
    return " ".join(parts)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or " ").strip()


def _looks_like_table_of_contents(text: str) -> bool:
    normalized = _normalize_text(text).lower()
    item_count = len(re.findall(r"\bitem\s+\d+[a-z]?\.?\b", normalized))
    if "table of contents" in normalized[:2000] and item_count >= 3:
        return True
    # Some rendered SEC filings split the table of contents across pages without
    # repeating the words "Table of Contents". A dense list of item headings is
    # safer to skip than to treat as a section start.
    return item_count >= 6


def _section_order(section_id: str) -> int:
    for index, spec in enumerate(SECTION_SPECS):
        if spec.section_id == section_id:
            return index
    return len(SECTION_SPECS)


def _conservative_end_page(current: dict[str, Any], next_item: dict[str, Any] | None, page_count: int) -> int:
    start_page = current["start_page"]
    current_order = _section_order(current["section_id"])
    if next_item is None:
        return page_count if current["section_id"] == "item_9a_controls" else start_page

    next_start = next_item["start_page"]
    if next_start <= start_page:
        return start_page
    next_order = _section_order(next_item["section_id"])
    if next_order == current_order + 1:
        return next_start - 1
    return start_page


def _write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SEC 10-K section labels and a section-level retrieval manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_corpus_manifest(args.manifest, pdf_root=args.pdf_root, require_files=True)
    labels_by_document = {
        document.document_id: extract_section_labels(loaded.document_paths[document.document_id])
        for document in loaded.manifest.documents
    }
    labels_artifact = build_labels_artifact(loaded, labels_by_document)
    manifest_payload = build_section_manifest_payload(loaded, labels_by_document)
    _write_json(args.labels_out, labels_artifact, args.overwrite)
    _write_json(args.manifest_out, manifest_payload, args.overwrite)
    summary = {
        "documents": len(loaded.manifest.documents),
        "section_labels": labels_artifact["section_label_count"],
        "queries": len(manifest_payload["queries"]),
        "labels_out": str(args.labels_out),
        "manifest_out": str(args.manifest_out),
        "limitations": LIMITATIONS,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()