import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.corpus_manifest import load_corpus_manifest, validate_manifest_payload
from benchmarks.corpus_sources.cuad import prepare_cuad_corpus
from benchmarks.corpus_sources.sec_edgar import prepare_sec_corpus
from benchmarks.e2e_document_rag_eval import _git_corpus_safety_checks, answer, preflight, write_json_report, write_markdown_report
from benchmarks.generate_synthetic_pdf_corpus import generate_synthetic_pdf_corpus
from benchmarks.promote_document_rag_report import ReportPromotionError, promote_report
from benchmarks.public_corpus_sources import load_public_source_registry
from benchmarks.render_sec_html_to_pdf import render_sec_html_manifest_to_pdf


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


def test_preflight_git_safety_allows_sanitized_public_summaries(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="\n".join([
            "benchmarks/corpora/results/sanitized_sec_retrieval_summary.md",
            "benchmarks/corpora/results/sanitized_sec_section_retrieval_v2_summary.md",
        ]))

    monkeypatch.setattr("benchmarks.e2e_document_rag_eval.subprocess.run", fake_run)
    checks = _git_corpus_safety_checks()

    assert checks == [{
        "name": "git_corpus_safety",
        "status": "pass",
        "required": True,
        "message": "No raw corpus files or generated local reports are tracked under corpus storage.",
    }]


def test_preflight_git_safety_rejects_raw_local_reports(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="benchmarks/corpora/results/document_rag_eval_answer_raw.json")

    monkeypatch.setattr("benchmarks.e2e_document_rag_eval.subprocess.run", fake_run)
    checks = _git_corpus_safety_checks()

    assert checks[0]["status"] == "fail"
    assert "document_rag_eval_answer_raw.json" in checks[0]["message"]


def test_answer_mode_applies_configured_provider_delay(tmp_path, monkeypatch):
    payload = json.loads(_manifest(tmp_path, mode="synthetic").read_text(encoding="utf-8"))
    payload["queries"].append({
        **payload["queries"][0],
        "query_id": "q2",
        "query": "What else is tested?",
    })
    manifest = tmp_path / "answer_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    sleeps = []

    def fake_call_answer_query(client, args, headers, query, request_payload):
        return {
            "query_id": query.query_id,
            "category": query.category,
            "status": "ok",
            "answer_non_empty": True,
            "citation_present": True,
            "citation_required": query.citation_required,
            "expected_hint_overlap": 1.0,
            "model_used": "gemini",
            "tokens_used": 1,
            "confidence_score": 0.8,
            "latency_ms": 1.0,
            "retrieval_count": 1,
        }

    monkeypatch.setattr("benchmarks.e2e_document_rag_eval._call_answer_query", fake_call_answer_query)
    monkeypatch.setattr("benchmarks.e2e_document_rag_eval.time.sleep", sleeps.append)
    args = SimpleNamespace(
        mode="answer",
        manifest=manifest,
        pdf_root=tmp_path / "local_pdfs",
        skip_file_check=True,
        ingestion_run=None,
        api_key=None,
        bearer_token=None,
        query_api_url="http://127.0.0.1:9",
        request_timeout_seconds=0.1,
        tenant_id="tenant_test",
        ingestion_url=None,
        status_url=None,
        pinecone_index=None,
        embedding_model=None,
        answer_top_k=5,
        model_choice="gemini",
        agent=None,
        retrieval_candidate_pool=100,
        sec_aware_rerank=True,
        sec_metadata_weight=0.5,
        answer_disable_target_doc_filter=True,
        answer_delay_seconds=1.25,
    )

    report = answer(args)

    assert sleeps == [1.25]
    assert report["answer"]["query_count"] == 2
    assert report["answer"]["answer_delay_seconds"] == 1.25
    assert report["answer"]["failure_count"] == 0


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


def test_sec_html_rendering_generates_rendered_manifest(tmp_path):
    pdf_root = tmp_path / "local_pdfs"
    source_dir = pdf_root / "sec_edgar"
    source_dir.mkdir(parents=True)
    (source_dir / "sample.htm").write_text(
        "<html><head><style>ignored</style></head><body><h1>Risk Factors</h1><p>Revenue and liquidity risk are disclosed.</p></body></html>",
        encoding="utf-8",
    )
    source_manifest = tmp_path / "sec_manifest.json"
    source_manifest.write_text(json.dumps({
        "corpus_id": "sec_test",
        "corpus_name": "SEC Test",
        "mode": "public",
        "source_metadata": {"source_id": "sec_edgar", "source_format": "html"},
        "documents": [
            {
                "document_id": "sec_test_doc",
                "filename": "sec_edgar/sample.htm",
                "doc_type": "financial_report",
                "source_type": "public_sec_edgar",
                "source_note": "SEC test filing; source_format=html",
                "source_metadata": {
                    "source_id": "sec_edgar",
                    "ticker": "TEST",
                    "accession_number": "0000000000-26-000001",
                    "form_type": "10-K",
                    "filing_date": "2026-01-01",
                    "source_url": "https://www.sec.gov/example",
                    "source_format": "html",
                },
                "allowed_to_commit": False,
            }
        ],
        "queries": [
            {
                "query_id": "q1",
                "query": "What risk is disclosed?",
                "category": "sec_risk_factors_template",
                "target_document_ids": ["sec_test_doc"],
                "relevant_pages": [],
                "relevant_chunk_ids": [],
                "expected_answer_hints": ["risk"],
                "citation_required": True,
            }
        ],
    }), encoding="utf-8")

    manifest_out = tmp_path / "rendered_manifest.json"
    report = render_sec_html_manifest_to_pdf(
        source_manifest=source_manifest,
        pdf_root=pdf_root,
        output_pdf_dir=pdf_root / "sec_edgar_rendered",
        manifest_out=manifest_out,
        overwrite=True,
    )
    loaded = load_corpus_manifest(manifest_out, pdf_root=pdf_root, require_files=True)
    rendered_doc = loaded.manifest.documents[0]
    assert report["document_count"] == 1
    assert rendered_doc.source_format == "rendered_pdf"
    assert rendered_doc.filename == "sec_edgar_rendered/sec_test_doc.pdf"
    assert rendered_doc.source_metadata["original_source_format"] == "html"
    assert rendered_doc.source_metadata["rendered_format"] == "pdf"
    assert rendered_doc.page_count and rendered_doc.page_count >= 1

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

def test_report_promotion_preserves_retrieval_granularity_metadata(tmp_path):
    report = {
        "mode": "retrieve",
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "git_commit": "abc123",
        "corpus": {
            "mode": "public",
            "corpus_id": "sec_public_test",
            "corpus_name": "SEC Public Test",
            "document_count": 2,
            "query_count": 2,
        },
        "retrieval": {
            "strategy": "pinecone_vector_candidates_plus_bm25_hybrid_rerank",
            "candidate_pool_size": 25,
            "top_k": 5,
            "label_granularity_counts": {"section": 2},
            "candidate_pool_miss_count": 1,
            "overall": {"recall@1": 0.5, "recall@3": 1.0, "recall@5": 1.0, "mrr": 0.75, "ndcg@5": 0.8},
            "queries": [{"query_id": "q1", "query": "sensitive query", "top_results": [{"text_preview": "raw text"}]}],
        },
        "limitations": ["local only"],
        "unsupported_claims": ["production quality"],
    }
    report_path = tmp_path / "retrieval_report.json"
    output_md = tmp_path / "sanitized.md"
    output_json = tmp_path / "sanitized.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    promote_report(report_path, output_markdown=output_md, output_json=output_json)

    sanitized = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert sanitized["retrieval"]["label_granularity_counts"] == {"section": 2}
    assert sanitized["retrieval"]["queries"][0]["query"] == "[removed-query-text]"
    assert sanitized["retrieval"]["queries"][0]["top_results"][0]["text_preview"] == "[removed-content-field]"
    assert 'Label granularity counts: `{"section": 2}`' in markdown
    assert "Candidate pool misses: `1`" in markdown

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

