import json

import pytest

from benchmarks.corpus_manifest import (
    ManifestValidationError,
    format_corpus_summary,
    load_corpus_manifest,
    validate_manifest_payload,
)


def _manifest_payload(**overrides):
    payload = {
        "corpus_id": "unit_corpus",
        "corpus_name": "Unit Test Corpus",
        "mode": "private_local",
        "documents": [
            {
                "document_id": "doc_001",
                "filename": "doc_001.pdf",
                "doc_type": "legal_contract",
                "source_type": "unit_test_pdf",
                "source_note": "Temporary test fixture.",
                "page_count": 2,
                "allowed_to_commit": False,
            }
        ],
        "queries": [
            {
                "query_id": "q001",
                "query": "What notice period applies?",
                "category": "legal_clause",
                "target_document_ids": ["doc_001"],
                "relevant_pages": [1],
                "relevant_chunk_ids": [],
                "expected_answer_hints": ["notice"],
                "citation_required": True,
            }
        ],
    }
    payload.update(overrides)
    return payload


def _write_manifest(temp_dir, payload):
    path = temp_dir / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.benchmark
def test_valid_manifest_loads_with_local_pdf(temp_dir):
    pdf_root = temp_dir / "local_pdfs"
    pdf_root.mkdir()
    (pdf_root / "doc_001.pdf").write_bytes(b"%PDF-1.4\n% unit test pdf placeholder\n")
    manifest_path = _write_manifest(temp_dir, _manifest_payload())

    loaded = load_corpus_manifest(manifest_path, pdf_root=pdf_root, require_files=True)

    assert loaded.manifest.corpus_id == "unit_corpus"
    assert loaded.document_paths["doc_001"].is_file()
    assert loaded.summary()["document_count"] == 1
    assert loaded.warnings == [
        "Manifest mode is private_local; keep PDFs and generated reports out of git by default."
    ]


@pytest.mark.benchmark
def test_missing_pdf_fails_validation(temp_dir):
    pdf_root = temp_dir / "local_pdfs"
    pdf_root.mkdir()
    manifest_path = _write_manifest(temp_dir, _manifest_payload())

    with pytest.raises(ManifestValidationError, match="Referenced PDF files are missing"):
        load_corpus_manifest(manifest_path, pdf_root=pdf_root, require_files=True)


@pytest.mark.benchmark
def test_duplicate_document_id_fails_schema_validation():
    payload = _manifest_payload()
    payload["documents"].append(dict(payload["documents"][0]))

    with pytest.raises(ManifestValidationError, match="Duplicate document_id"):
        validate_manifest_payload(payload)


@pytest.mark.benchmark
def test_query_referencing_missing_document_fails_schema_validation():
    payload = _manifest_payload()
    payload["queries"][0]["target_document_ids"] = ["missing_doc"]

    with pytest.raises(ManifestValidationError, match="unknown documents"):
        validate_manifest_payload(payload)


@pytest.mark.benchmark
def test_private_local_manifest_rejects_commit_safe_flag(temp_dir):
    payload = _manifest_payload()
    payload["documents"][0]["allowed_to_commit"] = True
    manifest_path = _write_manifest(temp_dir, payload)

    with pytest.raises(ManifestValidationError, match="private_local manifests cannot mark"):
        load_corpus_manifest(manifest_path, require_files=False)


@pytest.mark.benchmark
def test_private_local_warning_and_summary_without_files(temp_dir):
    manifest_path = _write_manifest(temp_dir, _manifest_payload())

    loaded = load_corpus_manifest(manifest_path, require_files=False)
    summary = format_corpus_summary(loaded)

    assert "Mode: private_local" in summary
    assert "Warnings:" in summary
    assert "Documents: 1" in summary


@pytest.mark.benchmark
def test_schema_validation_rejects_bad_filename():
    payload = _manifest_payload()
    payload["documents"][0]["filename"] = "../outside.pdf"

    with pytest.raises(ManifestValidationError, match="relative path"):
        validate_manifest_payload(payload)
