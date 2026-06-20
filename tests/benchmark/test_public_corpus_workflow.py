import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.corpus_manifest import load_corpus_manifest, validate_manifest_payload
from benchmarks.corpus_sources.cuad import prepare_cuad_corpus
from benchmarks.corpus_sources.sec_edgar import prepare_sec_corpus
from benchmarks.e2e_document_rag_eval import preflight, write_json_report, write_markdown_report
from benchmarks.generate_synthetic_pdf_corpus import generate_synthetic_pdf_corpus
from benchmarks.promote_document_rag_report import ReportPromotionError, promote_report
from benchmarks.public_corpus_sources import load_public_source_registry


def test_public_source_registry_schema():
    registry = load_public_source_registry()
    source_ids = {source.source_id for source in registry.sources}
    assert {"cuad_atticus", "sec_edgar"}.issubset(source_ids)
    assert registry.by_id("cuad_atticus").raw_files_may_be_committed is False
    assert registry.by_id("sec_edgar").recommended_local_storage_path.endswith("sec_edgar/")


def test_cuad_manifest_generation_from_mock_metadata(tmp_path):
    metadata = tmp_path / "cuad_metadata.json"
    metadata.write_text(json.dumps({
        "documents": [
            {
                "document_id": "Mock Contract",
                "title": "Mock Contract",
                "filename": "Mock Contract.pdf",
                "clause_categories": ["Termination", "Governing Law"],
            }
        ]
    }), encoding="utf-8")
    manifest_out = tmp_path / "cuad_manifest.json"
    report = prepare_cuad_corpus(
        metadata_json=metadata,
        output_pdf_dir=tmp_path / "local_pdfs" / "cuad",
        manifest_out=manifest_out,
        sample_size=1,
    )
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    validate_manifest_payload(manifest)
    assert report["document_count"] == 1
    assert manifest["mode"] == "public"
    assert manifest["documents"][0]["filename"].startswith("cuad/")
    assert manifest["documents"][0]["allowed_to_commit"] is False
    assert len(manifest["queries"]) == 2


def test_cuad_download_without_source_url_is_manual_required(tmp_path):
    metadata = tmp_path / "cuad_metadata.json"
    metadata.write_text(json.dumps({"documents": [{"title": "No URL Contract", "filename": "No URL Contract.pdf"}]}), encoding="utf-8")
    report = prepare_cuad_corpus(
        metadata_json=metadata,
        output_pdf_dir=tmp_path / "local_pdfs" / "cuad",
        manifest_out=tmp_path / "manifest.json",
        sample_size=1,
        download=True,
    )
    assert report["actions"][0]["status"] == "manual_required"


def test_sec_manifest_generation_from_mock_filing_metadata(tmp_path):
    filings = tmp_path / "sec_filings.json"
    filings.write_text(json.dumps({
        "filings": [
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "cik": "0000320193",
                "accession_number": "0000320193-24-000123",
                "form_type": "10-K",
                "filing_date": "2024-11-01",
                "primary_document": "aapl-20240928.htm",
                "source_format": "html",
            }
        ]
    }), encoding="utf-8")
    manifest_out = tmp_path / "sec_manifest.json"
    report = prepare_sec_corpus(
        filings_json=filings,
        output_file_dir=tmp_path / "local_pdfs" / "sec_edgar",
        manifest_out=manifest_out,
        sample_size=1,
    )
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    parsed = validate_manifest_payload(manifest)
    assert report["document_count"] == 1
    assert parsed.documents[0].source_format == "html"
    assert manifest["documents"][0]["filename"].endswith(".htm")
    assert manifest["documents"][0]["source_metadata"]["ticker"] == "AAPL"


def test_sec_fetch_requires_user_agent_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        prepare_sec_corpus(
            company_list=Path("benchmarks/corpora/sec_edgar_sample_companies.json"),
            tickers=["AAPL"],
            fetch_metadata=True,
            output_file_dir=tmp_path / "local_pdfs" / "sec_edgar",
            manifest_out=tmp_path / "manifest.json",
            sample_size=1,
        )


def test_sec_requires_metadata_or_fetch_flag(tmp_path):
    with pytest.raises(ValueError, match="--filings-json"):
        prepare_sec_corpus(
            output_file_dir=tmp_path / "local_pdfs" / "sec_edgar",
            manifest_out=tmp_path / "manifest.json",
        )


def test_preflight_validate_only_report_generation(tmp_path):
    manifest = _manifest(tmp_path, mode="synthetic")
    args = _preflight_args(tmp_path, manifest, target="validate-only", skip_file_check=True)
    report = preflight(args)
    assert report["preflight"]["required_failure_count"] == 0
    json_path = write_json_report(report, tmp_path / "reports", "preflight_test")
    md_path = json_path.with_suffix(".md")
    write_markdown_report(md_path, report)
    assert json_path.exists()
    assert "## Preflight" in md_path.read_text(encoding="utf-8")


def test_preflight_missing_local_pdf_for_ingest(tmp_path):
    manifest = _manifest(tmp_path, mode="synthetic")
    args = _preflight_args(tmp_path, manifest, target="ingest", skip_file_check=False)
    report = preflight(args)
    local_file_check = next(check for check in report["preflight"]["checks"] if check["name"] == "local_files")
    assert local_file_check["status"] == "fail"
    assert report["preflight"]["required_failure_count"] >= 1


def test_preflight_missing_pinecone_config_for_retrieve(tmp_path):
    manifest = _manifest(tmp_path, mode="synthetic")
    args = _preflight_args(tmp_path, manifest, target="retrieve", skip_file_check=True)
    args.pinecone_api_key = ""
    args.pinecone_index = ""
    report = preflight(args)
    failures = {check["name"] for check in report["preflight"]["checks"] if check["status"] == "fail"}
    assert {"pinecone_api_key", "pinecone_index"}.issubset(failures)


def test_preflight_private_corpus_warning(tmp_path):
    manifest = _manifest(tmp_path, mode="private_local")
    args = _preflight_args(tmp_path, manifest, target="validate-only", skip_file_check=True)
    report = preflight(args)
    warnings = [check["message"] for check in report["preflight"]["checks"] if check["status"] == "warn"]
    assert any("private_local" in warning for warning in warnings)


def test_synthetic_pdf_manifest_generation_and_labels(tmp_path):
    pdf_root = tmp_path / "local_pdfs"
    report = generate_synthetic_pdf_corpus(
        output_pdf_dir=pdf_root / "synthetic_smoke",
        manifest_out=tmp_path / "synthetic_manifest.json",
        overwrite=True,
        seed=3,
        num_docs=6,
    )
    loaded = load_corpus_manifest(Path(report["manifest_out"]), pdf_root=pdf_root, require_files=True)
    assert loaded.manifest.mode == "synthetic"
    assert len(loaded.manifest.documents) == 6
    assert all(query.relevant_pages for query in loaded.manifest.queries)


def test_report_promotion_sanitizes_paths_and_refuses_private(tmp_path):
    synthetic_report = {
        "mode": "validate-only",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "git_commit": "abc123",
        "corpus": {
            "mode": "synthetic",
            "corpus_id": "synthetic_test",
            "corpus_name": "Synthetic Test",
            "document_count": 1,
            "query_count": 1,
            "manifest_path": str(tmp_path / "manifest.json"),
            "pdf_root": str(tmp_path / "local_pdfs"),
            "documents": [{"document_id": "doc", "filename": "synthetic/doc.pdf", "doc_type": "legal_contract", "text_preview": "secret"}],
        },
        "answer": {"queries": [{"query_id": "q1", "query": "sensitive query", "answer": "raw answer"}]},
        "limitations": ["local only"],
        "unsupported_claims": ["production quality"],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(synthetic_report), encoding="utf-8")
    output_json = tmp_path / "sanitized.json"
    promote_report(report_path, output_markdown=tmp_path / "sanitized.md", output_json=output_json)
    sanitized = json.loads(output_json.read_text(encoding="utf-8"))
    assert "manifest_path" not in sanitized["corpus"]
    assert sanitized["answer"]["queries"][0]["query"] == "[removed-query-text]"
    assert sanitized["answer"]["queries"][0]["answer"] == "[removed-content-field]"

    synthetic_report["corpus"]["mode"] = "private_local"
    report_path.write_text(json.dumps(synthetic_report), encoding="utf-8")
    with pytest.raises(ReportPromotionError):
        promote_report(report_path, output_markdown=tmp_path / "private.md")


def _manifest(tmp_path: Path, *, mode: str) -> Path:
    payload = {
        "corpus_id": f"{mode}_test",
        "corpus_name": f"{mode} Test Corpus",
        "mode": mode,
        "documents": [
            {
                "document_id": "doc_001",
                "filename": "doc_001.pdf",
                "doc_type": "legal_contract",
                "source_type": mode,
                "source_note": "test fixture",
                "source_format": "pdf",
                "allowed_to_commit": False,
            }
        ],
        "queries": [
            {
                "query_id": "q1",
                "query": "What is tested?",
                "category": "smoke",
                "target_document_ids": ["doc_001"],
                "relevant_pages": [1],
                "relevant_chunk_ids": [],
                "expected_answer_hints": ["test"],
                "citation_required": True,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _preflight_args(tmp_path: Path, manifest: Path, *, target: str, skip_file_check: bool) -> SimpleNamespace:
    return SimpleNamespace(
        mode="preflight",
        preflight_target=target,
        public_source=None,
        manifest=manifest,
        pdf_root=tmp_path / "local_pdfs",
        output_dir=tmp_path / "reports",
        skip_file_check=skip_file_check,
        ingestion_url="http://127.0.0.1:9",
        query_api_url="http://127.0.0.1:9",
        request_timeout_seconds=0.1,
        pinecone_api_key="test-key",
        pinecone_index="test-index",
        pinecone_namespace=None,
        tenant_id="tenant_test",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
    )
