import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VALID_CORPUS_MODES = {"public", "synthetic", "private_local"}
VALID_DOCUMENT_TYPES = {"legal_contract", "financial_report", "other"}
VALID_SOURCE_FORMATS = {"pdf", "html", "rendered_pdf"}
SOURCE_FORMAT_SUFFIXES = {
    "pdf": {".pdf"},
    "rendered_pdf": {".pdf"},
    "html": {".html", ".htm", ".txt", ".xml"},
}


class ManifestValidationError(ValueError):
    """Raised when a corpus manifest does not match the expected schema."""


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    filename: str
    doc_type: str
    source_type: str
    source_note: str
    page_count: int | None
    allowed_to_commit: bool
    source_format: str = "pdf"
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusQuery:
    query_id: str
    query: str
    category: str
    target_document_ids: list[str]
    relevant_pages: list[int]
    relevant_chunk_ids: list[str]
    expected_answer_hints: list[str]
    citation_required: bool


@dataclass(frozen=True)
class CorpusManifest:
    corpus_id: str
    corpus_name: str
    mode: str
    documents: list[CorpusDocument]
    queries: list[CorpusQuery]
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_ids(self) -> set[str]:
        return {document.document_id for document in self.documents}

    def summary(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "corpus_name": self.corpus_name,
            "mode": self.mode,
            "document_count": len(self.documents),
            "query_count": len(self.queries),
            "doc_types": sorted({document.doc_type for document in self.documents}),
            "source_formats": sorted({document.source_format for document in self.documents}),
            "query_categories": sorted({query.category for query in self.queries}),
            "source_metadata": self.source_metadata,
        }


@dataclass(frozen=True)
class LoadedCorpus:
    manifest: CorpusManifest
    manifest_path: Path
    pdf_root: Path
    document_paths: dict[str, Path]
    warnings: list[str]

    def summary(self) -> dict[str, Any]:
        return {
            **self.manifest.summary(),
            "manifest_path": str(self.manifest_path),
            "pdf_root": str(self.pdf_root),
            "file_root": str(self.pdf_root),
            "warnings": self.warnings,
            "documents": [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "doc_type": document.doc_type,
                    "source_format": document.source_format,
                    "exists": self.document_paths[document.document_id].exists(),
                    "allowed_to_commit": document.allowed_to_commit,
                }
                for document in self.manifest.documents
            ],
        }


def read_manifest_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def validate_manifest_payload(payload: dict[str, Any]) -> CorpusManifest:
    required_root = {"corpus_id", "corpus_name", "mode", "documents", "queries"}
    _require_keys(payload, required_root, "manifest")

    corpus_id = _required_string(payload, "corpus_id", "manifest")
    corpus_name = _required_string(payload, "corpus_name", "manifest")
    mode = _required_string(payload, "mode", "manifest")
    if mode not in VALID_CORPUS_MODES:
        raise ManifestValidationError(f"manifest.mode must be one of {sorted(VALID_CORPUS_MODES)}")

    source_metadata = _optional_object(payload.get("source_metadata", {}), "manifest.source_metadata")
    raw_documents = payload["documents"]
    raw_queries = payload["queries"]
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ManifestValidationError("manifest.documents must be a non-empty list")
    if not isinstance(raw_queries, list):
        raise ManifestValidationError("manifest.queries must be a list")

    documents = [_parse_document(item, index) for index, item in enumerate(raw_documents)]
    document_ids = [document.document_id for document in documents]
    duplicates = _duplicates(document_ids)
    if duplicates:
        raise ManifestValidationError(f"Duplicate document_id values: {duplicates}")

    queries = [_parse_query(item, index, set(document_ids)) for index, item in enumerate(raw_queries)]
    query_ids = [query.query_id for query in queries]
    duplicate_queries = _duplicates(query_ids)
    if duplicate_queries:
        raise ManifestValidationError(f"Duplicate query_id values: {duplicate_queries}")

    return CorpusManifest(
        corpus_id=corpus_id,
        corpus_name=corpus_name,
        mode=mode,
        documents=documents,
        queries=queries,
        source_metadata=source_metadata,
    )


def load_manifest_schema(path: Path) -> CorpusManifest:
    return validate_manifest_payload(read_manifest_json(path))


def load_corpus_manifest(
    manifest_path: Path,
    *,
    pdf_root: Path | None = None,
    require_files: bool = True,
) -> LoadedCorpus:
    resolved_manifest_path = manifest_path.resolve()
    manifest = load_manifest_schema(resolved_manifest_path)
    resolved_pdf_root = (pdf_root or resolved_manifest_path.parent / "local_pdfs").resolve()

    warnings = _privacy_warnings(manifest)
    document_paths = {
        document.document_id: _resolve_corpus_file_path(document, resolved_pdf_root)
        for document in manifest.documents
    }

    if require_files:
        missing = [
            f"{document.document_id}: {document_paths[document.document_id]}"
            for document in manifest.documents
            if not document_paths[document.document_id].is_file()
        ]
        if missing:
            raise ManifestValidationError(f"Referenced PDF files are missing: {missing}")

    return LoadedCorpus(
        manifest=manifest,
        manifest_path=resolved_manifest_path,
        pdf_root=resolved_pdf_root,
        document_paths=document_paths,
        warnings=warnings,
    )


def format_corpus_summary(loaded: LoadedCorpus) -> str:
    summary = loaded.summary()
    lines = [
        f"Corpus: {summary['corpus_name']} ({summary['corpus_id']})",
        f"Mode: {summary['mode']}",
        f"Manifest: {summary['manifest_path']}",
        f"File root: {summary['file_root']}",
        f"Documents: {summary['document_count']}",
        f"Queries: {summary['query_count']}",
        f"Source formats: {', '.join(summary['source_formats']) if summary['source_formats'] else 'none'}",
    ]
    if loaded.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in loaded.warnings)
    return "\n".join(lines)


def _parse_document(raw: dict[str, Any], index: int) -> CorpusDocument:
    context = f"documents[{index}]"
    required = {
        "document_id",
        "filename",
        "doc_type",
        "source_type",
        "source_note",
        "allowed_to_commit",
    }
    _require_keys(raw, required, context)

    document_id = _required_string(raw, "document_id", context)
    filename = _required_string(raw, "filename", context)
    _validate_relative_filename(filename, context)

    doc_type = _required_string(raw, "doc_type", context)
    if doc_type not in VALID_DOCUMENT_TYPES:
        raise ManifestValidationError(f"{context}.doc_type must be one of {sorted(VALID_DOCUMENT_TYPES)}")

    source_metadata = _optional_object(raw.get("source_metadata", {}), f"{context}.source_metadata")
    source_format = _source_format(raw, source_metadata, filename, context)

    page_count = raw.get("page_count")
    if page_count is not None and (not isinstance(page_count, int) or page_count <= 0):
        raise ManifestValidationError(f"{context}.page_count must be a positive integer when provided")

    allowed_to_commit = raw["allowed_to_commit"]
    if not isinstance(allowed_to_commit, bool):
        raise ManifestValidationError(f"{context}.allowed_to_commit must be a boolean")

    return CorpusDocument(
        document_id=document_id,
        filename=filename,
        doc_type=doc_type,
        source_type=_required_string(raw, "source_type", context),
        source_note=_required_string(raw, "source_note", context),
        page_count=page_count,
        allowed_to_commit=allowed_to_commit,
        source_format=source_format,
        source_metadata=source_metadata,
    )


def _parse_query(raw: dict[str, Any], index: int, document_ids: set[str]) -> CorpusQuery:
    context = f"queries[{index}]"
    required = {"query_id", "query", "category", "target_document_ids", "citation_required"}
    _require_keys(raw, required, context)

    target_document_ids = _string_list(raw["target_document_ids"], f"{context}.target_document_ids")
    if not target_document_ids:
        raise ManifestValidationError(f"{context}.target_document_ids must not be empty")

    missing_targets = sorted(set(target_document_ids) - document_ids)
    if missing_targets:
        raise ManifestValidationError(f"{context}.target_document_ids reference unknown documents: {missing_targets}")

    citation_required = raw["citation_required"]
    if not isinstance(citation_required, bool):
        raise ManifestValidationError(f"{context}.citation_required must be a boolean")

    return CorpusQuery(
        query_id=_required_string(raw, "query_id", context),
        query=_required_string(raw, "query", context),
        category=_required_string(raw, "category", context),
        target_document_ids=target_document_ids,
        relevant_pages=_int_list(raw.get("relevant_pages", []), f"{context}.relevant_pages"),
        relevant_chunk_ids=_string_list(raw.get("relevant_chunk_ids", []), f"{context}.relevant_chunk_ids"),
        expected_answer_hints=_string_list(raw.get("expected_answer_hints", []), f"{context}.expected_answer_hints"),
        citation_required=citation_required,
    )


def _require_keys(raw: dict[str, Any], required: set[str], context: str) -> None:
    if not isinstance(raw, dict):
        raise ManifestValidationError(f"{context} must be an object")
    missing = sorted(required - set(raw))
    if missing:
        raise ManifestValidationError(f"{context} missing required keys: {missing}")


def _required_string(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list):
        raise ManifestValidationError(f"{context} must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ManifestValidationError(f"{context} must contain only non-empty strings")
    return [item.strip() for item in value]


def _int_list(value: Any, context: str) -> list[int]:
    if not isinstance(value, list):
        raise ManifestValidationError(f"{context} must be a list")
    if any(not isinstance(item, int) or item <= 0 for item in value):
        raise ManifestValidationError(f"{context} must contain only positive integers")
    return value


def _optional_object(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{context} must be an object when provided")
    return value


def _source_format(raw: dict[str, Any], source_metadata: dict[str, Any], filename: str, context: str) -> str:
    value = raw.get("source_format") or source_metadata.get("source_format") or _infer_source_format(filename)
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{context}.source_format must be a non-empty string when provided")
    source_format = value.strip().lower()
    if source_format not in VALID_SOURCE_FORMATS:
        raise ManifestValidationError(f"{context}.source_format must be one of {sorted(VALID_SOURCE_FORMATS)}")
    suffix = Path(filename).suffix.lower()
    if suffix not in SOURCE_FORMAT_SUFFIXES[source_format]:
        raise ManifestValidationError(
            f"{context}.filename suffix {suffix!r} is not valid for source_format {source_format!r}"
        )
    return source_format


def _infer_source_format(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in SOURCE_FORMAT_SUFFIXES["html"]:
        return "html"
    raise ManifestValidationError("filename must use a supported source format suffix: .pdf, .html, .htm, .txt, or .xml")


def _validate_relative_filename(filename: str, context: str) -> None:
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ManifestValidationError(f"{context}.filename must be a relative path inside the local corpus file directory")


def _resolve_corpus_file_path(document: CorpusDocument, pdf_root: Path) -> Path:
    path = (pdf_root / document.filename).resolve()
    if pdf_root not in path.parents and path != pdf_root:
        raise ManifestValidationError(f"{document.document_id}.filename resolves outside the configured corpus file root")
    return path


def _privacy_warnings(manifest: CorpusManifest) -> list[str]:
    warnings: list[str] = []
    if manifest.mode == "private_local":
        warnings.append("Manifest mode is private_local; keep PDFs and generated reports out of git by default.")
        unsafe = [document.document_id for document in manifest.documents if document.allowed_to_commit]
        if unsafe:
            raise ManifestValidationError(
                "private_local manifests cannot mark documents as allowed_to_commit=true: "
                f"{unsafe}"
            )
    elif any(not document.allowed_to_commit for document in manifest.documents):
        warnings.append("At least one document is marked allowed_to_commit=false; do not copy it into tracked paths.")
    if any(document.source_format == "html" for document in manifest.documents):
        warnings.append("At least one document is HTML; PDF ingestion may require local rendering/conversion before ingest mode.")
    return warnings


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
