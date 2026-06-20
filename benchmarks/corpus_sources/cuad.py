import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from benchmarks.corpus_manifest import validate_manifest_payload
from benchmarks.public_corpus_sources import PublicCorpusSource, load_public_source_registry


DEFAULT_CUAD_LOCAL_DIR = Path("benchmarks/corpora/local_pdfs/cuad")
DEFAULT_CUAD_MANIFEST = Path("benchmarks/corpora/cuad_manifest.generated.json")
MANUAL_CUAD_INSTRUCTIONS = """CUAD acquisition needs a local CUAD metadata file and contract PDFs, or explicit per-document download URLs.
Suggested workflow:
1. Review CUAD usage terms at https://github.com/TheAtticusProject/cuad.
2. Download the CUAD release outside git.
3. Place contract PDFs under benchmarks/corpora/local_pdfs/cuad/.
4. Run this adapter with --metadata-json pointing at CUAD metadata and --no-download to generate a manifest.
5. Run validate-only before any ingestion call.
"""


@dataclass(frozen=True)
class CuadDocumentSpec:
    document_id: str
    title: str
    filename: str
    source_url: str | None = None
    local_source_path: Path | None = None
    clause_categories: list[str] = field(default_factory=list)
    source_note: str = "CUAD / Atticus Project public legal contract corpus"


def load_cuad_document_specs(metadata_path: Path, *, sample_size: int) -> list[CuadDocumentSpec]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and isinstance(payload.get("documents"), list):
        raw_documents = payload["documents"]
    elif isinstance(payload, list):
        raw_documents = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        raw_documents = payload["data"]
    else:
        raise ValueError("CUAD metadata must contain a documents list, be a list, or use a CUAD/SQuAD-style data list")

    specs: list[CuadDocumentSpec] = []
    for index, raw in enumerate(raw_documents[:sample_size], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"CUAD metadata row {index} must be an object")
        specs.append(_parse_cuad_document(raw, index))
    return specs


def build_cuad_manifest(
    specs: list[CuadDocumentSpec],
    *,
    source: PublicCorpusSource | None = None,
    corpus_id: str = "cuad_public_sample",
    corpus_name: str = "CUAD Public Legal Contract Sample",
) -> dict[str, Any]:
    source = source or load_public_source_registry().by_id("cuad_atticus")
    documents = []
    queries = []
    for spec in specs:
        documents.append({
            "document_id": spec.document_id,
            "filename": f"cuad/{spec.filename}",
            "doc_type": "legal_contract",
            "source_type": "public_cuad",
            "source_note": _source_note(spec),
            "source_metadata": {
                **source.to_manifest_source_metadata(),
                "source_url": spec.source_url or source.source_url,
                "source_format": "pdf",
            },
            "page_count": None,
            "allowed_to_commit": False,
        })
        for clause in spec.clause_categories[:3]:
            query_id = f"{spec.document_id}_{_slug(clause)}"
            queries.append({
                "query_id": query_id,
                "query": f"Which part of {spec.title} discusses {clause}?",
                "category": "cuad_clause_template",
                "target_document_ids": [spec.document_id],
                "relevant_pages": [],
                "relevant_chunk_ids": [],
                "expected_answer_hints": [clause],
                "citation_required": True,
            })

    return {
        "corpus_id": corpus_id,
        "corpus_name": corpus_name,
        "mode": "public",
        "source_metadata": source.to_manifest_source_metadata(),
        "documents": documents,
        "queries": queries,
    }


def prepare_cuad_corpus(
    *,
    metadata_json: Path,
    output_pdf_dir: Path = DEFAULT_CUAD_LOCAL_DIR,
    manifest_out: Path = DEFAULT_CUAD_MANIFEST,
    sample_size: int = 10,
    download: bool = False,
    copy_local_files: bool = False,
    overwrite: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    registry = load_public_source_registry()
    source = registry.by_id("cuad_atticus")
    specs = load_cuad_document_specs(metadata_json, sample_size=sample_size)
    output_pdf_dir.mkdir(parents=True, exist_ok=True)

    actions: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        for spec in specs:
            target_path = output_pdf_dir / spec.filename
            action = {
                "document_id": spec.document_id,
                "filename": spec.filename,
                "target_path": str(target_path),
                "status": "manifest_only",
            }
            if target_path.exists() and not overwrite:
                action["status"] = "exists"
            elif copy_local_files and spec.local_source_path:
                _copy_local_pdf(spec.local_source_path, target_path)
                action["status"] = "copied"
            elif download and spec.source_url:
                _download_pdf(client, spec.source_url, target_path)
                action["status"] = "downloaded"
            elif download and not spec.source_url:
                action["status"] = "manual_required"
                action["warning"] = "No source_url is available in CUAD metadata for automatic download"
            actions.append(action)

    manifest = build_cuad_manifest(specs, source=source)
    validate_manifest_payload(manifest)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "source_id": source.source_id,
        "metadata_json": str(metadata_json),
        "manifest_out": str(manifest_out),
        "output_pdf_dir": str(output_pdf_dir),
        "sample_size": sample_size,
        "document_count": len(specs),
        "query_count": len(manifest["queries"]),
        "actions": actions,
        "manual_instructions": MANUAL_CUAD_INSTRUCTIONS,
        "limitations": source.limitations,
    }


def _parse_cuad_document(raw: dict[str, Any], index: int) -> CuadDocumentSpec:
    title = str(raw.get("title") or raw.get("name") or raw.get("document_name") or f"CUAD contract {index}").strip()
    document_id = str(raw.get("document_id") or raw.get("id") or _slug(title) or f"cuad_{index:03d}").strip()
    filename = str(raw.get("filename") or raw.get("pdf_filename") or f"{_slug(title) or document_id}.pdf").strip()
    filename = _pdf_filename(filename)
    source_url = raw.get("source_url") or raw.get("url") or raw.get("pdf_url")
    local_source = raw.get("local_source_path") or raw.get("local_path")
    clause_categories = _clause_categories(raw)
    source_note = str(raw.get("source_note") or "CUAD / Atticus Project public legal contract corpus").strip()
    return CuadDocumentSpec(
        document_id=_slug(document_id) or f"cuad_{index:03d}",
        title=title,
        filename=filename,
        source_url=str(source_url).strip() if source_url else None,
        local_source_path=Path(local_source) if local_source else None,
        clause_categories=clause_categories,
        source_note=source_note,
    )


def _clause_categories(raw: dict[str, Any]) -> list[str]:
    if isinstance(raw.get("clause_categories"), list):
        return [str(item).strip() for item in raw["clause_categories"] if str(item).strip()]
    if isinstance(raw.get("qas"), list):
        return [str(item.get("id") or item.get("question") or "").strip() for item in raw["qas"] if isinstance(item, dict) and str(item.get("id") or item.get("question") or "").strip()]
    categories: set[str] = set()
    for paragraph in raw.get("paragraphs", []) if isinstance(raw.get("paragraphs"), list) else []:
        for qa in paragraph.get("qas", []) if isinstance(paragraph, dict) else []:
            if isinstance(qa, dict):
                value = qa.get("id") or qa.get("question")
                if value:
                    categories.add(str(value).strip())
    return sorted(categories)


def _copy_local_pdf(source_path: Path, target_path: Path) -> None:
    resolved = source_path.resolve()
    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"CUAD local_source_path must point to a PDF: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(resolved, target_path)


def _download_pdf(client: httpx.Client, source_url: str, target_path: Path) -> None:
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported CUAD download URL: {source_url}")
    response = client.get(source_url)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not source_url.lower().endswith(".pdf"):
        raise ValueError(f"CUAD download did not look like a PDF: {source_url}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(response.content)


def _source_note(spec: CuadDocumentSpec) -> str:
    if spec.source_url:
        return f"{spec.source_note}; source_url={spec.source_url}"
    return spec.source_note


def _pdf_filename(value: str) -> str:
    name = Path(value).name
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return _safe_filename(name)


def _safe_filename(value: str) -> str:
    stem = Path(value).stem
    suffix = Path(value).suffix or ".pdf"
    return f"{_slug(stem) or 'document'}{suffix.lower()}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
