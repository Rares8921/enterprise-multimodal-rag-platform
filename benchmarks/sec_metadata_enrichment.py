"""Enrich SEC Pinecone vectors with manifest-derived section metadata.

This utility copies vectors from an existing namespace into a new namespace while
adding public SEC manifest metadata and conservative section labels. It does not
re-embed documents and does not write raw filing text to reports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.corpus_manifest import load_manifest_schema  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "corpora" / "sec_edgar_section_manifest.generated.json"
DEFAULT_INGESTION_RUN = REPO_ROOT / "benchmarks" / "corpora" / "results" / "document_rag_eval_ingest_sec_ingest.json"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "corpora" / "results" / "sec_metadata_enrichment_v2.json"
DEFAULT_SOURCE_NAMESPACE = "tenant_eval_local"
DEFAULT_TARGET_NAMESPACE = "tenant_eval_sec_sections_v2"
ITEM_HEADING_PATTERN = re.compile(r"\bitem\s+\d+[a-z]?\.?\b", re.IGNORECASE)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_ingestion_mapping(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for row in payload.get("ingestion", {}).get("documents", []):
        manifest_id = row.get("manifest_document_id")
        service_id = row.get("service_document_id")
        if manifest_id and service_id:
            mapping[str(manifest_id)] = str(service_id)
    return mapping


def build_document_lookup(manifest_path: Path, ingestion_mapping: dict[str, str]) -> dict[str, dict[str, Any]]:
    manifest = load_manifest_schema(manifest_path)
    lookup: dict[str, dict[str, Any]] = {}
    for document in manifest.documents:
        source_metadata = document.source_metadata or {}
        service_document_id = ingestion_mapping.get(document.document_id)
        record = {
            "document_id": document.document_id,
            "service_document_id": service_document_id,
            "filename": document.filename,
            "filename_basename": Path(document.filename).name,
            "doc_type": document.doc_type,
            "ticker": source_metadata.get("ticker"),
            "company_name": source_metadata.get("company_name"),
            "filing_date": source_metadata.get("filing_date"),
            "filing_year": _filing_year(source_metadata.get("filing_date")),
            "form_type": source_metadata.get("form_type"),
            "accession_number": source_metadata.get("accession_number"),
            "section_labels": source_metadata.get("sec_section_labels", []),
        }
        for key in [
            document.document_id,
            service_document_id,
            document.filename,
            Path(document.filename).name,
        ]:
            if key:
                lookup[str(key)] = record
    return lookup


def enrich_metadata(metadata: dict[str, Any], document_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(metadata)
    record = _match_document_record(metadata, document_lookup)
    page_number = _coerce_int(metadata.get("page_number", metadata.get("page")))
    if page_number is not None:
        enriched["page_number"] = page_number
        enriched["page"] = page_number

    if record:
        enriched.update({
            "manifest_document_id": record["document_id"],
            "document_id": record["document_id"],
            "service_document_id": record.get("service_document_id") or metadata.get("doc_id"),
            "source_filename": record["filename"],
            "ticker": record.get("ticker") or "unknown",
            "company_name": record.get("company_name") or "unknown",
            "filing_date": record.get("filing_date") or "unknown",
            "filing_year": record.get("filing_year") or "unknown",
            "form_type": record.get("form_type") or "unknown",
            "accession_number": record.get("accession_number") or "unknown",
        })
        label = section_for_page(record.get("section_labels", []), page_number)
    else:
        label = None

    if label:
        enriched.update({
            "section_id": label["section_id"],
            "section_name": label.get("section_name") or label["section_id"],
            "section_confidence": label.get("confidence", "unknown"),
        })
    else:
        enriched.update({
            "section_id": "unknown",
            "section_name": "unknown",
            "section_confidence": "unknown",
        })

    text = str(metadata.get("text", ""))
    enriched["is_table_of_contents"] = is_table_of_contents_chunk(text, label)
    enriched["metadata_enrichment"] = "sec_section_manifest_v2"
    return enriched


def section_for_page(section_labels: Iterable[dict[str, Any]], page_number: int | None) -> dict[str, Any] | None:
    if page_number is None:
        return None
    for label in section_labels or []:
        start_page = _coerce_int(label.get("start_page"))
        end_page = _coerce_int(label.get("end_page")) or start_page
        if start_page is None or end_page is None:
            continue
        if start_page <= page_number <= end_page:
            return label
    return None


def is_table_of_contents_chunk(text: str, section_label: dict[str, Any] | None) -> bool:
    normalized = " ".join((text or "").lower().split())
    if "table of contents" in normalized[:500]:
        return True
    heading_count = len(ITEM_HEADING_PATTERN.findall(normalized[:1200]))
    return section_label is None and heading_count >= 5


def reindex_namespace(
    *,
    index: Any,
    document_lookup: dict[str, dict[str, Any]],
    source_namespace: str,
    target_namespace: str,
    batch_size: int = 100,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    ids = list(_iter_vector_ids(index, namespace=source_namespace, limit=limit))
    stats: Counter[str] = Counter()
    per_document: Counter[str] = Counter()

    for batch_ids in _batches(ids, batch_size):
        fetched = _fetch_vectors(index, batch_ids, namespace=source_namespace)
        upsert_payload = []
        for vector_id, vector in fetched.items():
            metadata = dict(vector.get("metadata") or {})
            values = vector.get("values")
            if values is None:
                raise RuntimeError(f"Fetched vector {vector_id} did not include values")
            enriched_metadata = enrich_metadata(metadata, document_lookup)
            stats["vectors"] += 1
            if enriched_metadata.get("section_id") != "unknown":
                stats["section_mapped"] += 1
            else:
                stats["section_unknown"] += 1
            if enriched_metadata.get("is_table_of_contents"):
                stats["table_of_contents"] += 1
            per_document[str(enriched_metadata.get("document_id", "unknown"))] += 1
            upsert_payload.append((vector_id, values, enriched_metadata))
        if upsert_payload and not dry_run:
            index.upsert(vectors=upsert_payload, namespace=target_namespace)

    return {
        "vector_count": stats["vectors"],
        "section_mapped_count": stats["section_mapped"],
        "section_unknown_count": stats["section_unknown"],
        "table_of_contents_count": stats["table_of_contents"],
        "per_document_vector_counts": dict(sorted(per_document.items())),
    }


def _iter_vector_ids(index: Any, *, namespace: str, limit: int | None) -> Iterable[str]:
    yielded = 0
    for page in index.list(namespace=namespace):
        for vector_id in list(page):
            yield vector_id
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _fetch_vectors(index: Any, ids: list[str], *, namespace: str) -> dict[str, dict[str, Any]]:
    response = index.fetch(ids=ids, namespace=namespace)
    raw_vectors = response.get("vectors", {}) if isinstance(response, dict) else getattr(response, "vectors", {})
    normalized: dict[str, dict[str, Any]] = {}
    for vector_id, vector in raw_vectors.items():
        if isinstance(vector, dict):
            normalized[vector_id] = {
                "values": vector.get("values"),
                "metadata": vector.get("metadata", {}),
            }
        else:
            normalized[vector_id] = {
                "values": getattr(vector, "values", None),
                "metadata": getattr(vector, "metadata", {}) or {},
            }
    return normalized


def _batches(items: list[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _match_document_record(metadata: dict[str, Any], document_lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        metadata.get("manifest_document_id"),
        metadata.get("document_id"),
        metadata.get("doc_id"),
        metadata.get("source_filename"),
        metadata.get("filename"),
        Path(str(metadata.get("filename", ""))).name if metadata.get("filename") else None,
    ]
    for candidate in candidates:
        if candidate is not None and str(candidate) in document_lookup:
            return document_lookup[str(candidate)]
    return None


def _filing_year(filing_date: Any) -> str | None:
    if not filing_date:
        return None
    match = re.search(r"\b(20\d{2}|19\d{2})\b", str(filing_date))
    return match.group(1) if match else None


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_pinecone_index(index_name: str):
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise RuntimeError("Pinecone SDK is required for metadata enrichment") from exc
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is required")
    return Pinecone(api_key=api_key).Index(index_name)


def delete_namespace_if_exists(index: Any, namespace: str) -> bool:
    try:
        index.delete(delete_all=True, namespace=namespace)
        return True
    except Exception as exc:
        message = str(exc).lower()
        if "namespace not found" in message or "not found" in message or "404" in message:
            return False
        raise


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy SEC vectors to a new namespace with enriched section metadata.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ingestion-run", type=Path, default=DEFAULT_INGESTION_RUN)
    parser.add_argument("--pinecone-index", default=os.getenv("PINECONE_INDEX", "doc-intelligence"))
    parser.add_argument("--source-namespace", default=DEFAULT_SOURCE_NAMESPACE)
    parser.add_argument("--target-namespace", default=DEFAULT_TARGET_NAMESPACE)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-namespace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestion_mapping = load_ingestion_mapping(args.ingestion_run)
    document_lookup = build_document_lookup(args.manifest, ingestion_mapping)
    index = _load_pinecone_index(args.pinecone_index)
    target_deleted = False
    if args.overwrite_namespace and not args.dry_run:
        target_deleted = delete_namespace_if_exists(index, args.target_namespace)
    stats = reindex_namespace(
        index=index,
        document_lookup=document_lookup,
        source_namespace=args.source_namespace,
        target_namespace=args.target_namespace,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    report = {
        "artifact_name": "sec_metadata_enrichment",
        "timestamp_utc": utc_timestamp(),
        "manifest": str(args.manifest),
        "ingestion_run": str(args.ingestion_run),
        "pinecone_index": args.pinecone_index,
        "source_namespace": args.source_namespace,
        "target_namespace": args.target_namespace,
        "dry_run": args.dry_run,
        "overwrote_target_namespace": target_deleted,
        "metadata_fields_added": [
            "manifest_document_id",
            "document_id",
            "service_document_id",
            "section_id",
            "section_name",
            "section_confidence",
            "ticker",
            "company_name",
            "filing_date",
            "filing_year",
            "form_type",
            "accession_number",
            "page_number",
            "is_table_of_contents",
            "metadata_enrichment",
        ],
        "stats": stats,
        "limitations": [
            "Vectors are copied from the existing namespace; documents are not re-acquired or re-embedded.",
            "Section metadata is mapped from approximate page ranges in the public SEC section manifest.",
            "Chunks outside known section page ranges are marked section_id=unknown rather than guessed.",
            "This does not create chunk-level correctness labels or production retrieval-quality evidence by itself.",
        ],
    }
    write_report(args.report_out, report)
    print(json.dumps({"status": "ok", "stats": stats, "report": str(args.report_out)}, indent=2))


if __name__ == "__main__":
    main()
