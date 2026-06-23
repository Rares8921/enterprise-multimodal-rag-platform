import argparse
import importlib.util
import json
import math
import mimetypes
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.corpus_manifest import LoadedCorpus, ManifestValidationError, format_corpus_summary, load_corpus_manifest


DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "corpora" / "example_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "corpora" / "results"
DEFAULT_PDF_ROOT = REPO_ROOT / "benchmarks" / "corpora" / "local_pdfs"
HYBRID_MODULE_PATH = REPO_ROOT / "services" / "inference-api" / "utils" / "hybrid_retrieval.py"
INGESTION_SUPPORTED_DOC_TYPES = {"legal_contract", "financial_report"}
TERMINAL_STATUSES = {"ocr_complete", "embedding_complete", "indexed", "completed", "failed"}
INGESTION_SUPPORTED_SUFFIXES = {".pdf"}
METRIC_KEYS = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]

def _load_hybrid_module():
    spec = importlib.util.spec_from_file_location("document_rag_eval_hybrid_utils", HYBRID_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["document_rag_eval_hybrid_utils"] = module
    spec.loader.exec_module(module)
    return module


hybrid_utils = _load_hybrid_module()
hybrid_rerank = hybrid_utils.hybrid_rerank
sec_aware_rerank = getattr(hybrid_utils, "sec_aware_rerank", None)


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_base_report(args: argparse.Namespace, loaded: LoadedCorpus) -> dict[str, Any]:
    return {
        "benchmark_name": "document_rag_real_service_evaluation",
        "mode": args.mode,
        "timestamp_utc": utc_timestamp(),
        "git_commit": get_git_commit(),
        "command": "python " + " ".join(sys.argv),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "runner": "local real-service document RAG evaluation harness",
        },
        "corpus": loaded.summary(),
        "services": {
            "ingestion_url": args.ingestion_url if args.mode == "ingest" else None,
            "status_url": args.status_url if args.mode == "ingest" else None,
            "pinecone_index": args.pinecone_index if args.mode == "retrieve" else None,
            "pinecone_namespace": _pinecone_namespace(args) if args.mode == "retrieve" else None,
            "embedding_model": args.embedding_model if args.mode == "retrieve" else None,
            "query_api_url": args.query_api_url if args.mode == "answer" else None,
            "tenant_id": args.tenant_id,
        },
        "limitations": [
            "This harness runs against local or explicitly configured services; it is not production evidence.",
            "Raw PDFs are expected to live in an ignored local directory and are not committed by default.",
            "Manifest labels may be incomplete until real indexing produces page or chunk labels.",
            "Ingestion and retrieval results do not prove legal correctness, financial correctness, QPS, uptime, or cost savings.",
        ],
        "unsupported_claims": [
            "production usage",
            "customer data evaluation",
            "real users",
            "production legal or financial correctness",
            "production retrieval quality",
            "uptime, QPS, SLA, or incident recovery",
            "real cost savings",
            "provider accuracy",
        ],
    }


def validate_only(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_corpus_manifest(
        args.manifest,
        pdf_root=args.pdf_root,
        require_files=not args.skip_file_check,
    )
    report = build_base_report(args, loaded)
    report["validation"] = {
        "status": "passed",
        "file_check": not args.skip_file_check,
        "warnings": loaded.warnings,
    }
    print(format_corpus_summary(loaded))
    return report


def ingest(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_corpus_manifest(args.manifest, pdf_root=args.pdf_root, require_files=True)
    report = build_base_report(args, loaded)
    headers = _auth_headers(args)
    document_results: list[dict[str, Any]] = []

    with httpx.Client(timeout=args.request_timeout_seconds) as client:
        for document in loaded.manifest.documents:
            local_path = loaded.document_paths[document.document_id]
            if document.doc_type not in INGESTION_SUPPORTED_DOC_TYPES:
                document_results.append({
                    "manifest_document_id": document.document_id,
                    "filename": document.filename,
                    "status": "skipped",
                    "error": f"doc_type {document.doc_type!r} is not supported by the ingestion API",
                })
                continue
            if local_path.suffix.lower() not in INGESTION_SUPPORTED_SUFFIXES:
                document_results.append({
                    "manifest_document_id": document.document_id,
                    "filename": document.filename,
                    "status": "skipped",
                    "source_format": getattr(document, "source_format", "unknown"),
                    "error": "ingest mode only uploads PDF files; render or convert this source before ingestion",
                })
                continue

            result = _upload_document(client, args, headers, loaded, document.document_id, local_path)
            if args.poll_status and result.get("service_document_id"):
                result["status_polls"] = _poll_document_status(
                    client,
                    args,
                    headers,
                    result["service_document_id"],
                )
                if result["status_polls"]:
                    result["final_status"] = result["status_polls"][-1].get("status")
            document_results.append(result)

    successes = sum(1 for row in document_results if row.get("status") == "uploaded")
    failures = sum(1 for row in document_results if row.get("status") == "failed")
    skipped = sum(1 for row in document_results if row.get("status") == "skipped")
    report["ingestion"] = {
        "document_count": len(document_results),
        "success_count": successes,
        "failure_count": failures,
        "skipped_count": skipped,
        "poll_status": args.poll_status,
        "documents": document_results,
    }

    if failures:
        report["limitations"].append("One or more documents failed ingestion; downstream retrieval evaluation should not be run until failures are resolved.")

    return report



def retrieve(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_corpus_manifest(
        args.manifest,
        pdf_root=args.pdf_root,
        require_files=not args.skip_file_check,
    )
    report = build_base_report(args, loaded)
    ingestion_mapping = _load_ingestion_mapping(args.ingestion_run) if args.ingestion_run else {}
    index = _load_pinecone_index(args)
    model = _load_embedding_model(args.embedding_model)
    namespace = _pinecone_namespace(args)

    section_labels_by_document = _section_labels_by_document(loaded.manifest.documents)
    document_identity_lookup = _document_identity_lookup(loaded.manifest.documents, ingestion_mapping)

    rows: list[dict[str, Any]] = []
    for query in loaded.manifest.queries:
        vector = model.encode(query.query, convert_to_numpy=True, normalize_embeddings=True)
        raw_results = index.query(
            vector=vector.tolist() if hasattr(vector, "tolist") else vector,
            top_k=args.retrieval_candidate_pool,
            namespace=namespace,
            include_metadata=True,
        )
        matches = _matches_from_pinecone_response(raw_results)
        if args.sec_aware_rerank:
            if sec_aware_rerank is None:
                raise RuntimeError("SEC-aware reranking is unavailable in the hybrid retrieval utility")
            reranked = sec_aware_rerank(
                query.query,
                matches,
                vector_weight=args.vector_weight,
                bm25_weight=args.bm25_weight,
                metadata_weight=args.sec_metadata_weight,
            )
        else:
            reranked = hybrid_rerank(
                query.query,
                matches,
                vector_weight=args.vector_weight,
                bm25_weight=args.bm25_weight,
            )
        rows.append(_evaluate_retrieval_query(
            query,
            reranked,
            ingestion_mapping,
            args.top_k,
            section_labels_by_document,
            document_identity_lookup,
        ))

    report["retrieval"] = {
        "mode": "pinecone_real_service",
        "strategy": "pinecone_vector_candidates_plus_bm25_sec_aware_rerank" if args.sec_aware_rerank else "pinecone_vector_candidates_plus_bm25_hybrid_rerank",
        "top_k": args.top_k,
        "candidate_pool_size": args.retrieval_candidate_pool,
        "vector_weight": args.vector_weight,
        "bm25_weight": args.bm25_weight,
        "sec_aware_rerank": args.sec_aware_rerank,
        "sec_metadata_weight": args.sec_metadata_weight if args.sec_aware_rerank else None,
        "ingestion_run": str(args.ingestion_run) if args.ingestion_run else None,
        "overall": _average_metrics(rows),
        "by_category": _category_metrics(rows),
        "label_granularity_counts": dict(Counter(row["label_granularity"] for row in rows)),
        "candidate_pool_miss_count": sum(1 for row in rows if not row["candidate_pool_contains_relevant"]),
        "metric_deduplication": "document/page/section/chunk label identity before metric calculation",
        "missed_queries_top5": [
            {
                "query_id": row["query_id"],
                "category": row["category"],
                "label_granularity": row["label_granularity"],
                "target_document_ids": row["target_document_ids"],
                "top_results": row["top_results"],
            }
            for row in rows
            if row["metrics"]["recall@5"] == 0.0
        ],
        "queries": rows,
    }
    report["limitations"].extend([
        "Retrieval mode requires a live Pinecone index populated by the ingestion/OCR/layout/embedding pipeline.",
        "Document-level labels are weaker than section-level, page-level, or chunk-level labels; metric interpretation depends on manifest label granularity.",
        "Retrieval metrics deduplicate repeated Pinecone chunks by the active label granularity before scoring.",
        "This is local real-service evidence only and must not be described as production retrieval quality.",
    ])
    return report

def answer(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_corpus_manifest(
        args.manifest,
        pdf_root=args.pdf_root,
        require_files=not args.skip_file_check,
    )
    report = build_base_report(args, loaded)
    ingestion_mapping = _load_ingestion_mapping(args.ingestion_run) if args.ingestion_run else {}
    headers = _auth_headers(args)
    document_types = {document.document_id: document.doc_type for document in loaded.manifest.documents}
    previous_answer = _load_answer_resume_source(args.answer_retry_failed_from) if args.answer_retry_failed_from else None
    previous_rows = previous_answer.get("queries", []) if previous_answer else []
    queries_to_run = list(loaded.manifest.queries)
    if previous_answer:
        failed_query_ids = {
            row.get("query_id")
            for row in previous_rows
            if row.get("status") == "failed" and row.get("query_id")
        }
        queries_to_run = [query for query in loaded.manifest.queries if query.query_id in failed_query_ids]

    retry_rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.request_timeout_seconds) as client:
        for index, query in enumerate(queries_to_run):
            request_payload = _build_answer_request_payload(args, query, ingestion_mapping, document_types)
            row = _call_answer_query_with_retries(client, args, headers, query, request_payload)
            retry_rows.append(row)
            if args.answer_delay_seconds > 0 and index < len(queries_to_run) - 1:
                time.sleep(args.answer_delay_seconds)

    rows = _merge_answer_rows(loaded.manifest.queries, previous_rows, retry_rows) if previous_answer else retry_rows
    answer_payload = {
        "mode": "optional_real_service_answer_proxy",
        **_answer_metric_summary(rows, args),
        "queries": rows,
    }
    if previous_answer:
        answer_payload["resume"] = {
            "source_report": Path(args.answer_retry_failed_from).name,
            "source_metrics": _compact_answer_metrics(previous_answer),
            "retry_query_count": len(retry_rows),
            "retry_metrics": _answer_metric_summary(retry_rows, args),
            "combined_metrics": _compact_answer_metrics(answer_payload),
        }
    report["answer"] = answer_payload
    report["limitations"].extend([
        "Answer mode may call real LLM providers through the query service and can incur cost depending on local configuration.",
        "Answer evaluation is a lightweight proxy: non-empty answer, citation presence, and expected-hint overlap only.",
        "Expected-hint overlap is not semantic correctness and must not be described as legal or financial accuracy.",
        "Use --answer-delay-seconds for live providers with request-per-minute limits; delayed runs trade runtime for fewer provider quota failures.",
        "Use --answer-retry-failed-from only to transparently retry failed prior rows; combined reports keep the full manifest denominator and preserve remaining failures.",
    ])
    return report

def preflight(args: argparse.Namespace) -> dict[str, Any]:
    report = _preflight_base_report(args)
    checks: list[dict[str, Any]] = []
    loaded: LoadedCorpus | None = None

    try:
        loaded = load_corpus_manifest(args.manifest, pdf_root=args.pdf_root, require_files=False)
        report["corpus"] = loaded.summary()
        checks.append(_preflight_check("manifest_schema", "pass", "Manifest schema is valid.", required=True))
    except ManifestValidationError as exc:
        checks.append(_preflight_check("manifest_schema", "fail", str(exc), required=True))
        report["preflight"] = {"target": args.preflight_target, "public_source": args.public_source, "checks": checks, "required_failure_count": 1, "warning_count": 0, "status": "failed"}
        return report

    missing_files = [
        {"document_id": document.document_id, "path": str(loaded.document_paths[document.document_id])}
        for document in loaded.manifest.documents
        if not loaded.document_paths[document.document_id].is_file()
    ]
    files_required = args.preflight_target in {"ingest", "retrieve", "answer"} and not args.skip_file_check
    if missing_files and files_required:
        checks.append(_preflight_check("local_files", "fail", f"Missing referenced local files: {missing_files}", required=True))
    elif missing_files:
        checks.append(_preflight_check("local_files", "warn", f"Missing local files; skipped or optional for this preflight target: {missing_files}", required=False))
    else:
        checks.append(_preflight_check("local_files", "pass", "All referenced local files exist." if loaded.manifest.documents else "No documents to check.", required=files_required))

    for warning in loaded.warnings:
        checks.append(_preflight_check("corpus_warning", "warn", warning, required=False))

    checks.append(_output_dir_preflight(args.output_dir))
    checks.extend(_git_corpus_safety_checks())

    if args.preflight_target == "ingest":
        checks.append(_endpoint_preflight("ingestion_url", args.ingestion_url, args.request_timeout_seconds, required=True))
    if args.preflight_target == "retrieve":
        checks.extend(_pinecone_preflight(args))
    if args.preflight_target == "answer":
        checks.append(_endpoint_preflight("query_api_url", args.query_api_url, args.request_timeout_seconds, required=True))
    if args.preflight_target == "acquisition" or _manifest_uses_sec(loaded):
        checks.append(_sec_user_agent_preflight(required=args.preflight_target == "acquisition" and args.public_source == "sec_edgar"))

    required_failure_count = sum(1 for check in checks if check["required"] and check["status"] == "fail")
    report["preflight"] = {
        "target": args.preflight_target,
        "public_source": args.public_source,
        "checks": checks,
        "required_failure_count": required_failure_count,
        "warning_count": sum(1 for check in checks if check["status"] == "warn"),
        "status": "failed" if required_failure_count else "passed",
    }
    return report


def _preflight_base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "benchmark_name": "document_rag_preflight",
        "mode": "preflight",
        "timestamp_utc": utc_timestamp(),
        "git_commit": get_git_commit(),
        "command": "python " + " ".join(sys.argv),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "runner": "local document RAG preflight",
        },
        "corpus": {
            "manifest_path": str(args.manifest),
            "pdf_root": str(args.pdf_root),
            "file_root": str(args.pdf_root),
        },
        "services": {
            "ingestion_url": args.ingestion_url if args.preflight_target == "ingest" else None,
            "pinecone_index": args.pinecone_index if args.preflight_target == "retrieve" else None,
            "pinecone_namespace": _pinecone_namespace(args) if args.preflight_target == "retrieve" else None,
            "embedding_model": args.embedding_model if args.preflight_target == "retrieve" else None,
            "query_api_url": args.query_api_url if args.preflight_target == "answer" else None,
            "tenant_id": args.tenant_id,
        },
        "limitations": [
            "Preflight checks readiness only; they do not ingest, retrieve, evaluate answer quality, or prove production behavior.",
            "Endpoint reachability checks do not prove service correctness or downstream dependencies.",
            "Git safety checks only inspect tracked corpus artifacts in this repository worktree.",
        ],
        "unsupported_claims": [
            "production usage",
            "customer data evaluation",
            "real users",
            "production retrieval quality",
            "legal or financial correctness",
            "uptime, QPS, SLA, or real cost savings",
        ],
    }


def _preflight_check(name: str, status: str, message: str, *, required: bool) -> dict[str, Any]:
    return {"name": name, "status": status, "required": required, "message": message}


def _output_dir_preflight(output_dir: Path) -> dict[str, Any]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return _preflight_check("output_dir_writable", "pass", f"Output directory is writable: {output_dir}", required=True)
    except Exception as exc:
        return _preflight_check("output_dir_writable", "fail", f"Output directory is not writable: {exc}", required=True)


def _git_corpus_safety_checks() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "benchmarks/corpora/local_pdfs", "benchmarks/corpora/results"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        tracked = [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]
    except Exception as exc:
        return [_preflight_check("git_corpus_safety", "warn", f"Could not inspect tracked corpus artifacts: {exc}", required=False)]
    unsafe = []
    for path in tracked:
        name = Path(path).name
        is_sanitized_summary = (
            path.startswith("benchmarks/corpora/results/")
            and name.startswith("sanitized_")
            and name.endswith("_summary.md")
        )
        if path.endswith(".gitkeep") or is_sanitized_summary:
            continue
        if (
            path.startswith("benchmarks/corpora/local_pdfs/")
            or path.startswith("benchmarks/corpora/results/")
            or path.lower().endswith((".pdf", ".html", ".htm"))
            or "document_rag_eval_" in path
        ):
            unsafe.append(path)
    if unsafe:
        return [_preflight_check("git_corpus_safety", "fail", f"Potential raw/generated corpus artifacts are tracked: {unsafe}", required=True)]
    return [_preflight_check("git_corpus_safety", "pass", "No raw corpus files or generated local reports are tracked under corpus storage.", required=True)]


def _endpoint_preflight(name: str, url: str, timeout_seconds: float, *, required: bool) -> dict[str, Any]:
    if not url:
        return _preflight_check(name, "fail" if required else "warn", "No URL configured.", required=required)
    try:
        with httpx.Client(timeout=min(timeout_seconds, 5.0), follow_redirects=True) as client:
            response = client.get(url.rstrip("/"))
        if response.status_code < 500:
            return _preflight_check(name, "pass", f"Endpoint responded with HTTP {response.status_code}: {url}", required=required)
        return _preflight_check(name, "fail", f"Endpoint returned HTTP {response.status_code}: {url}", required=required)
    except httpx.RequestError as exc:
        return _preflight_check(name, "fail" if required else "warn", f"Endpoint is not reachable: {exc}", required=required)


def _pinecone_preflight(args: argparse.Namespace) -> list[dict[str, Any]]:
    checks = []
    checks.append(_preflight_check(
        "pinecone_api_key",
        "pass" if args.pinecone_api_key else "fail",
        "PINECONE_API_KEY is configured." if args.pinecone_api_key else "PINECONE_API_KEY or --pinecone-api-key is required for retrieve mode.",
        required=True,
    ))
    checks.append(_preflight_check(
        "pinecone_index",
        "pass" if args.pinecone_index else "fail",
        f"Pinecone index configured: {args.pinecone_index}" if args.pinecone_index else "PINECONE_INDEX or --pinecone-index is required for retrieve mode.",
        required=True,
    ))
    checks.append(_preflight_check(
        "pinecone_namespace",
        "pass" if _pinecone_namespace(args) else "fail",
        f"Pinecone namespace configured: {_pinecone_namespace(args)}" if _pinecone_namespace(args) else "Pinecone namespace or tenant_id is required.",
        required=True,
    ))
    checks.append(_preflight_check(
        "embedding_model",
        "pass" if args.embedding_model else "fail",
        f"Embedding model configured: {args.embedding_model}" if args.embedding_model else "Embedding model is required for retrieve mode.",
        required=True,
    ))
    return checks


def _manifest_uses_sec(loaded: LoadedCorpus) -> bool:
    root_source = loaded.manifest.source_metadata.get("source_id")
    if root_source == "sec_edgar":
        return True
    return any(document.source_type == "public_sec_edgar" or document.source_metadata.get("source_id") == "sec_edgar" for document in loaded.manifest.documents)


def _sec_user_agent_preflight(*, required: bool) -> dict[str, Any]:
    value = os.getenv("SEC_USER_AGENT")
    if value:
        return _preflight_check("sec_user_agent", "pass", "SEC_USER_AGENT is configured for SEC access.", required=required)
    return _preflight_check("sec_user_agent", "fail" if required else "warn", "SEC_USER_AGENT is not configured; SEC acquisition should not run without a real contact User-Agent.", required=required)
def write_json_report(report: dict[str, Any], output_dir: Path, run_id: str | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"document_rag_eval_{report['mode']}_{suffix}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path




def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    corpus = report["corpus"]
    lines = [
        "# Document RAG Evaluation Report",
        "",
        f"Mode: `{report['mode']}`",
        f"Timestamp UTC: `{report['timestamp_utc']}`",
        f"Git commit: `{report['git_commit']}`",
        f"Command: `{report['command']}`",
        f"Manifest: `{corpus['manifest_path']}`",
        f"PDF root: `{corpus['pdf_root']}`",
        f"Corpus: `{corpus['corpus_name']}` (`{corpus['corpus_id']}`)",
        f"Corpus mode: `{corpus['mode']}`",
        f"Documents: `{corpus['document_count']}`",
        f"Queries: `{corpus['query_count']}`",
        "",
        "## Services",
        "",
    ]
    for key, value in report.get("services", {}).items():
        if value is not None:
            lines.append(f"- `{key}`: `{value}`")
    if all(value is None for value in report.get("services", {}).values()):
        lines.append("- No service calls were made in this mode.")

    if "validation" in report:
        lines.extend([
            "",
            "## Validation",
            "",
            f"Status: `{report['validation']['status']}`",
            f"File check: `{report['validation']['file_check']}`",
        ])

    if "preflight" in report:
        preflight_payload = report["preflight"]
        lines.extend([
            "",
            "## Preflight",
            "",
            f"Target: `{preflight_payload['target']}`",
            f"Status: `{preflight_payload['status']}`",
            f"Required failures: `{preflight_payload['required_failure_count']}`",
            f"Warnings: `{preflight_payload['warning_count']}`",
            "",
            "| Check | Status | Required | Message |",
            "|---|---|---:|---|",
        ])
        for check in preflight_payload.get("checks", []):
            lines.append(f"| `{check['name']}` | `{check['status']}` | `{check['required']}` | {check['message']} |")

    if "ingestion" in report:
        ingestion = report["ingestion"]
        lines.extend([
            "",
            "## Ingestion",
            "",
            f"Documents: `{ingestion['document_count']}`",
            f"Successes: `{ingestion['success_count']}`",
            f"Failures: `{ingestion['failure_count']}`",
            f"Skipped: `{ingestion['skipped_count']}`",
        ])

    if "retrieval" in report:
        retrieval = report["retrieval"]
        metrics = retrieval["overall"]
        lines.extend([
            "",
            "## Retrieval",
            "",
            f"Mode: `{retrieval['mode']}`",
            f"Strategy: `{retrieval['strategy']}`",
            f"Candidate pool size: `{retrieval['candidate_pool_size']}`",
            f"Top K: `{retrieval['top_k']}`",
            f"Label granularity counts: `{json.dumps(retrieval.get('label_granularity_counts', {}), sort_keys=True)}`",
            f"Candidate pool misses: `{retrieval['candidate_pool_miss_count']}`",
            "",
            "| Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |",
            "|---:|---:|---:|---:|---:|",
            f"| {metrics['recall@1']:.4f} | {metrics['recall@3']:.4f} | {metrics['recall@5']:.4f} | {metrics['mrr']:.4f} | {metrics['ndcg@5']:.4f} |",
        ])

    if "answer" in report:
        answer_payload = report["answer"]
        lines.extend([
            "",
            "## Answer Proxy",
            "",
            f"Queries: `{answer_payload['query_count']}`",
            f"Failures: `{answer_payload['failure_count']}`",
            f"Non-empty answer rate: `{answer_payload['non_empty_answer_rate']}`",
            f"Citation presence rate for required citations: `{answer_payload['citation_presence_rate_required']}`",
            f"Average expected-hint overlap: `{answer_payload['average_expected_hint_overlap']}`",
            f"Estimated tokens used: `{answer_payload['estimated_tokens_used']}`",
        ])

    query_rows = _report_query_rows(report)
    if query_rows:
        lines.extend([
            "",
            "## Per-Query Results",
            "",
            "| Query | Category | Status | Metric Summary |",
            "|---|---|---|---|",
        ])
        for row in query_rows:
            lines.append(f"| `{row['query_id']}` | {row['category']} | {row['status']} | {row['summary']} |")

    lines.extend([
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    lines.extend([
        "",
        "## Unsupported Claims",
        "",
    ])
    lines.extend(f"- {item}" for item in report.get("unsupported_claims", []))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_report(path: Path, report: dict[str, Any]) -> None:
    import csv

    rows = _report_query_rows(report)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "category", "status", "summary"])
        writer.writeheader()
        writer.writerows(rows)


def _report_query_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    if "retrieval" in report:
        rows = []
        for query in report["retrieval"].get("queries", []):
            metrics = query["metrics"]
            rows.append({
                "query_id": query["query_id"],
                "category": query["category"],
                "status": query["label_granularity"],
                "summary": f"R@5={metrics['recall@5']:.4f}; MRR={metrics['mrr']:.4f}; nDCG@5={metrics['ndcg@5']:.4f}",
            })
        return rows
    if "answer" in report:
        return [
            {
                "query_id": query["query_id"],
                "category": query["category"],
                "status": query["status"],
                "summary": f"non_empty={query.get('answer_non_empty')}; citations={query.get('citation_present')}; hint_overlap={query.get('expected_hint_overlap')}",
            }
            for query in report["answer"].get("queries", [])
        ]
    if "ingestion" in report:
        return [
            {
                "query_id": row["manifest_document_id"],
                "category": row.get("filename", ""),
                "status": row["status"],
                "summary": row.get("service_document_id") or row.get("error", ""),
            }
            for row in report["ingestion"].get("documents", [])
        ]
    return []
def recall_at_k(ranked_relevance: list[bool], relevant_count: int, k: int) -> float:
    if relevant_count <= 0:
        return 0.0
    return min(sum(1 for item in ranked_relevance[:k] if item), relevant_count) / relevant_count


def reciprocal_rank(ranked_relevance: list[bool]) -> float:
    for rank, is_relevant in enumerate(ranked_relevance, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_relevance: list[bool], relevant_count: int, k: int) -> float:
    dcg = 0.0
    for rank, is_relevant in enumerate(ranked_relevance[:k], start=1):
        if is_relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(relevant_count, k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0
def _upload_document(
    client: httpx.Client,
    args: argparse.Namespace,
    headers: dict[str, str],
    loaded: LoadedCorpus,
    manifest_document_id: str,
    local_path: Path,
) -> dict[str, Any]:
    document = next(item for item in loaded.manifest.documents if item.document_id == manifest_document_id)
    metadata = {
        "corpus_id": loaded.manifest.corpus_id,
        "manifest_document_id": document.document_id,
        "source_type": document.source_type,
        "source_note": document.source_note,
        "page_count": document.page_count,
    }
    mime_type = mimetypes.guess_type(local_path.name)[0] or "application/pdf"
    try:
        with local_path.open("rb") as f:
            response = client.post(
                f"{args.ingestion_url.rstrip('/')}/documents/upload",
                headers=headers or None,
                data={
                    "tenant_id": args.tenant_id,
                    "doc_type": document.doc_type,
                    "metadata": json.dumps(metadata),
                },
                files={"file": (local_path.name, f, mime_type)},
            )
        response.raise_for_status()
        payload = response.json()
        return {
            "manifest_document_id": document.document_id,
            "filename": document.filename,
            "status": "uploaded",
            "service_document_id": payload.get("doc_id"),
            "service_status": payload.get("status"),
            "response": payload,
        }
    except httpx.HTTPStatusError as exc:
        return {
            "manifest_document_id": document.document_id,
            "filename": document.filename,
            "status": "failed",
            "error": f"HTTP {exc.response.status_code}: {_response_text(exc.response)}",
        }
    except httpx.RequestError as exc:
        return {
            "manifest_document_id": document.document_id,
            "filename": document.filename,
            "status": "failed",
            "error": f"Service unavailable: {exc}",
        }


def _poll_document_status(
    client: httpx.Client,
    args: argparse.Namespace,
    headers: dict[str, str],
    service_document_id: str,
) -> list[dict[str, Any]]:
    deadline = time.time() + args.poll_timeout_seconds
    polls: list[dict[str, Any]] = []
    while time.time() <= deadline:
        try:
            response = client.get(
                f"{args.status_url.rstrip('/')}/documents/{service_document_id}/status",
                headers=headers or None,
            )
            response.raise_for_status()
            payload = response.json()
            polls.append({
                "timestamp_utc": utc_timestamp(),
                "status": payload.get("status"),
                "response": payload,
            })
            if payload.get("status") in TERMINAL_STATUSES:
                break
        except httpx.HTTPStatusError as exc:
            polls.append({
                "timestamp_utc": utc_timestamp(),
                "status": "status_check_failed",
                "error": f"HTTP {exc.response.status_code}: {_response_text(exc.response)}",
            })
            break
        except httpx.RequestError as exc:
            polls.append({
                "timestamp_utc": utc_timestamp(),
                "status": "status_check_failed",
                "error": f"Service unavailable: {exc}",
            })
            break
        time.sleep(args.poll_interval_seconds)
    return polls



def _load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Retrieval mode requires sentence-transformers to be installed") from exc
    return SentenceTransformer(model_name, device="cpu")


def _load_pinecone_index(args: argparse.Namespace):
    if not args.pinecone_api_key:
        raise RuntimeError("Retrieval mode requires PINECONE_API_KEY or --pinecone-api-key")
    if not args.pinecone_index:
        raise RuntimeError("Retrieval mode requires PINECONE_INDEX or --pinecone-index")
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise RuntimeError("Retrieval mode requires pinecone to be installed") from exc
    return Pinecone(api_key=args.pinecone_api_key).Index(args.pinecone_index)


def _load_ingestion_mapping(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for row in payload.get("ingestion", {}).get("documents", []):
        manifest_id = row.get("manifest_document_id")
        service_id = row.get("service_document_id")
        if manifest_id and service_id:
            mapping[manifest_id] = service_id
    return mapping


def _matches_from_pinecone_response(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        return list(response.get("matches", []))
    matches = getattr(response, "matches", [])
    normalized: list[dict[str, Any]] = []
    for match in matches:
        if isinstance(match, dict):
            normalized.append(match)
        else:
            normalized.append({
                "id": getattr(match, "id", None),
                "score": getattr(match, "score", 0.0),
                "metadata": getattr(match, "metadata", {}) or {},
            })
    return normalized


def _evaluate_retrieval_query(
    query: Any,
    matches: list[dict[str, Any]],
    ingestion_mapping: dict[str, str],
    top_k: int,
    section_labels_by_document: dict[str, list[dict[str, Any]]] | None = None,
    document_identity_lookup: dict[str, str] | None = None,
) -> dict[str, Any]:
    section_labels_by_document = section_labels_by_document or {}
    document_identity_lookup = document_identity_lookup or {
        **{str(manifest_id): str(manifest_id) for manifest_id in ingestion_mapping},
        **{str(service_id): str(manifest_id) for manifest_id, service_id in ingestion_mapping.items()},
    }
    target_document_ids = set(query.target_document_ids)
    mapped_target_ids = {ingestion_mapping.get(document_id, document_id) for document_id in target_document_ids}
    relevant_pages = set(query.relevant_pages)
    relevant_sections = {str(section_id) for section_id in getattr(query, "relevant_sections", [])}
    relevant_chunk_ids = {str(chunk_id) for chunk_id in query.relevant_chunk_ids}
    label_granularity = _label_granularity(query)
    relevant_count = _relevant_count(query)

    all_results = []
    for rank, match in enumerate(matches, start=1):
        metadata = match.get("metadata") or {}
        result = {
            "rank": rank,
            "id": match.get("id"),
            "score": match.get("sec_aware_score", match.get("hybrid_score", match.get("score", 0.0))),
            "vector_score": match.get("vector_score", match.get("score", 0.0)),
            "bm25_score": match.get("bm25_score", 0.0),
            "hybrid_score": match.get("hybrid_score"),
            "sec_aware_score": match.get("sec_aware_score"),
            "sec_metadata_score": match.get("sec_metadata_score"),
            "sec_metadata_reasons": match.get("sec_metadata_reasons"),
            "doc_id": metadata.get("doc_id"),
            "manifest_document_id": metadata.get("manifest_document_id"),
            "filename": metadata.get("filename"),
            "page": metadata.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "indexed_section_id": metadata.get("section_id"),
            "indexed_ticker": metadata.get("ticker"),
            "indexed_accession_number": metadata.get("accession_number"),
            "indexed_filing_year": metadata.get("filing_year"),
            "is_table_of_contents": metadata.get("is_table_of_contents"),
            "text_preview": (metadata.get("text") or "")[:240],
        }
        result["matched_manifest_document_id"] = _manifest_document_id_for_result(result, document_identity_lookup)
        result["section_id"] = _section_id_for_page(
            result["matched_manifest_document_id"],
            result.get("page"),
            section_labels_by_document,
        )
        result["is_relevant"] = _is_relevant_result(
            result,
            target_document_ids,
            mapped_target_ids,
            relevant_pages,
            relevant_sections,
            relevant_chunk_ids,
            label_granularity,
        )
        all_results.append(result)

    metric_results = _dedupe_results_for_metrics(all_results, label_granularity)
    top_results = all_results[:top_k]
    metric_top_results = metric_results[:top_k]
    ranked_relevance = [result["is_relevant"] for result in metric_top_results]
    metrics = {
        "recall@1": recall_at_k(ranked_relevance, relevant_count, 1),
        "recall@3": recall_at_k(ranked_relevance, relevant_count, 3),
        "recall@5": recall_at_k(ranked_relevance, relevant_count, 5),
        "mrr": reciprocal_rank(ranked_relevance),
        "ndcg@5": ndcg_at_k(ranked_relevance, relevant_count, 5),
    }
    return {
        "query_id": query.query_id,
        "query": query.query,
        "category": query.category,
        "target_document_ids": query.target_document_ids,
        "mapped_target_document_ids": sorted(mapped_target_ids),
        "relevant_pages": query.relevant_pages,
        "relevant_sections": sorted(relevant_sections),
        "relevant_chunk_ids": query.relevant_chunk_ids,
        "label_granularity": label_granularity,
        "candidate_pool_contains_relevant": any(result["is_relevant"] for result in all_results),
        "metrics": metrics,
        "metrics_deduplicated_by": label_granularity,
        "metric_result_count": len(metric_results),
        "top_results": top_results,
    }


def _section_labels_by_document(documents: list[Any]) -> dict[str, list[dict[str, Any]]]:
    labels_by_document: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        raw_labels = document.source_metadata.get("sec_section_labels", [])
        if not isinstance(raw_labels, list):
            continue
        normalized = []
        for raw_label in raw_labels:
            if not isinstance(raw_label, dict):
                continue
            section_id = raw_label.get("section_id")
            start_page = _coerce_int(raw_label.get("start_page"))
            if not section_id or start_page is None:
                continue
            end_page = _coerce_int(raw_label.get("end_page")) or start_page
            if end_page < start_page:
                end_page = start_page
            normalized.append({
                "section_id": str(section_id),
                "section_name": raw_label.get("section_name"),
                "start_page": start_page,
                "end_page": end_page,
                "confidence": raw_label.get("confidence"),
            })
        labels_by_document[document.document_id] = normalized
    return labels_by_document


def _document_identity_lookup(documents: list[Any], ingestion_mapping: dict[str, str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for document in documents:
        manifest_id = str(document.document_id)
        for value in [document.document_id, document.filename, Path(document.filename).name]:
            if value:
                lookup[str(value)] = manifest_id
        service_id = ingestion_mapping.get(document.document_id)
        if service_id:
            lookup[str(service_id)] = manifest_id
    return lookup


def _manifest_document_id_for_result(result: dict[str, Any], document_identity_lookup: dict[str, str]) -> str | None:
    for value in [result.get("manifest_document_id"), result.get("doc_id"), result.get("filename")]:
        if value is None:
            continue
        candidate = str(value)
        if candidate in document_identity_lookup:
            return document_identity_lookup[candidate]
        basename = Path(candidate).name
        if basename in document_identity_lookup:
            return document_identity_lookup[basename]
    return None


def _section_id_for_page(
    manifest_document_id: str | None,
    page: Any,
    section_labels_by_document: dict[str, list[dict[str, Any]]],
) -> str | None:
    page_number = _coerce_int(page)
    if not manifest_document_id or page_number is None:
        return None
    for label in section_labels_by_document.get(manifest_document_id, []):
        start_page = _coerce_int(label.get("start_page"))
        end_page = _coerce_int(label.get("end_page")) or start_page
        if start_page is None or end_page is None:
            continue
        if start_page <= page_number <= end_page:
            return str(label.get("section_id"))
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _dedupe_results_for_metrics(results: list[dict[str, Any]], label_granularity: str) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for result in results:
        key = _metric_identity(result, label_granularity)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _metric_identity(result: dict[str, Any], label_granularity: str) -> tuple[Any, ...]:
    doc_identity = next(
        (
            str(value)
            for value in [
                result.get("matched_manifest_document_id"),
                result.get("doc_id"),
                result.get("manifest_document_id"),
                result.get("filename"),
            ]
            if value is not None
        ),
        str(result.get("id")),
    )
    if label_granularity == "document":
        return ("document", doc_identity)
    if label_granularity == "section":
        return ("section", doc_identity, result.get("section_id") or result.get("page"))
    if label_granularity == "page":
        return ("page", doc_identity, result.get("page"))
    chunk_identity = result.get("id") or f"{doc_identity}:{result.get('chunk_id')}"
    return ("chunk", str(chunk_identity))

def _label_granularity(query: Any) -> str:
    if getattr(query, "relevant_chunk_ids", []):
        return "chunk"
    if getattr(query, "relevant_sections", []):
        return "section"
    if getattr(query, "relevant_pages", []):
        return "page"
    return "document"

def _relevant_count(query: Any) -> int:
    relevant_chunk_ids = getattr(query, "relevant_chunk_ids", [])
    relevant_sections = getattr(query, "relevant_sections", [])
    relevant_pages = getattr(query, "relevant_pages", [])
    if relevant_chunk_ids:
        return len(set(relevant_chunk_ids))
    if relevant_sections:
        return max(1, len(set(query.target_document_ids)) * len(set(relevant_sections)))
    if relevant_pages:
        return max(1, len(set(relevant_pages)))
    return max(1, len(set(query.target_document_ids)))

def _is_relevant_result(
    result: dict[str, Any],
    target_document_ids: set[str],
    mapped_target_ids: set[str],
    relevant_pages: set[int],
    relevant_sections: set[str],
    relevant_chunk_ids: set[str],
    label_granularity: str,
) -> bool:
    doc_candidates = {
        str(value)
        for value in [
            result.get("matched_manifest_document_id"),
            result.get("doc_id"),
            result.get("manifest_document_id"),
            result.get("filename"),
        ]
        if value is not None
    }
    doc_match = bool(doc_candidates & (target_document_ids | mapped_target_ids))
    if label_granularity == "document":
        return doc_match
    if label_granularity == "section":
        return doc_match and result.get("section_id") in relevant_sections
    if label_granularity == "page":
        return doc_match and _coerce_int(result.get("page")) in relevant_pages
    chunk_candidates = {str(value) for value in [result.get("id"), result.get("chunk_id")] if value is not None}
    return bool(chunk_candidates & relevant_chunk_ids)

def _average_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in METRIC_KEYS}
    return {
        key: round(mean(row["metrics"][key] for row in rows), 6)
        for key in METRIC_KEYS
    }


def _category_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return {
        category: {
            "query_count": len(category_rows),
            "metrics": _average_metrics(category_rows),
        }
        for category, category_rows in sorted(grouped.items())
    }

def _load_answer_resume_source(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    answer_payload = payload.get("answer")
    if not isinstance(answer_payload, dict) or not isinstance(answer_payload.get("queries"), list):
        raise RuntimeError(f"Answer resume source does not contain answer queries: {path}")
    return answer_payload


def _merge_answer_rows(
    manifest_queries: list[Any],
    previous_rows: list[dict[str, Any]],
    retry_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    previous_by_id = {row.get("query_id"): row for row in previous_rows if row.get("query_id")}
    retry_by_id = {row.get("query_id"): row for row in retry_rows if row.get("query_id")}
    merged: list[dict[str, Any]] = []
    for query in manifest_queries:
        query_id = query.query_id
        if query_id in retry_by_id:
            row = dict(retry_by_id[query_id])
            row["resume_previous_status"] = previous_by_id.get(query_id, {}).get("status")
            merged.append(row)
        elif query_id in previous_by_id:
            row = dict(previous_by_id[query_id])
            row["resume_retained_from_previous_report"] = True
            merged.append(row)
    return merged


def _answer_metric_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    successful_rows = [row for row in rows if row.get("status") == "ok"]
    return {
        "query_count": len(rows),
        "failure_count": sum(1 for row in rows if row.get("status") == "failed"),
        "non_empty_answer_rate": _rate(row.get("answer_non_empty") for row in rows),
        "citation_presence_rate_required": _required_citation_rate(rows),
        "average_expected_hint_overlap": round(mean(row.get("expected_hint_overlap", 0.0) for row in rows), 6) if rows else 0.0,
        "model_counts": dict(Counter(row.get("model_used") for row in rows if row.get("model_used"))),
        "retrieval_strategy_counts": dict(Counter(row.get("retrieval_strategy") for row in rows if row.get("retrieval_strategy"))),
        "estimated_tokens_used": sum(row.get("tokens_used", 0) or 0 for row in rows),
        "average_confidence_score": round(mean(row.get("confidence_score", 0.0) for row in successful_rows if row.get("confidence_score") is not None), 6) if any(row.get("confidence_score") is not None for row in successful_rows) else 0.0,
        "average_latency_ms": round(mean(row.get("latency_ms", 0.0) for row in successful_rows if row.get("latency_ms") is not None), 6) if any(row.get("latency_ms") is not None for row in successful_rows) else 0.0,
        "average_retrieval_count": round(mean(row.get("retrieval_count", 0.0) for row in successful_rows if row.get("retrieval_count") is not None), 6) if any(row.get("retrieval_count") is not None for row in successful_rows) else 0.0,
        "retrieval_candidate_pool": args.retrieval_candidate_pool,
        "sec_aware_rerank": args.sec_aware_rerank,
        "sec_metadata_weight": args.sec_metadata_weight if args.sec_aware_rerank else None,
        "target_doc_filter_enabled": not args.answer_disable_target_doc_filter,
        "answer_delay_seconds": args.answer_delay_seconds,
        "answer_max_retries": args.answer_max_retries,
        "answer_retry_cooldown_seconds": args.answer_retry_cooldown_seconds,
        "failure_errors": dict(Counter(row.get("error") for row in rows if row.get("status") == "failed" and row.get("error"))),
    }


def _compact_answer_metrics(answer_payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "query_count",
        "failure_count",
        "non_empty_answer_rate",
        "citation_presence_rate_required",
        "average_expected_hint_overlap",
        "estimated_tokens_used",
        "model_counts",
        "retrieval_strategy_counts",
        "answer_delay_seconds",
        "answer_max_retries",
        "answer_retry_cooldown_seconds",
        "failure_errors",
    ]
    return {key: answer_payload.get(key) for key in keys if key in answer_payload}


def _build_answer_request_payload(
    args: argparse.Namespace,
    query: Any,
    ingestion_mapping: dict[str, str],
    document_types: dict[str, str],
) -> dict[str, Any]:
    mapped_targets = [ingestion_mapping.get(document_id, document_id) for document_id in query.target_document_ids]
    doc_type = None
    target_types = {document_types.get(document_id) for document_id in query.target_document_ids}
    target_types.discard(None)
    if len(target_types) == 1:
        doc_type = next(iter(target_types))

    payload = {
        "query": query.query,
        "tenant_id": args.tenant_id,
        "top_k": args.answer_top_k,
        "model_choice": args.model_choice,
        "agent": args.agent,
        "include_citations": True,
        "retrieval_candidate_pool": args.retrieval_candidate_pool,
        "sec_aware_rerank": args.sec_aware_rerank,
        "sec_metadata_weight": args.sec_metadata_weight,
    }
    if doc_type in INGESTION_SUPPORTED_DOC_TYPES:
        payload["doc_type"] = doc_type
    if len(mapped_targets) == 1 and not args.answer_disable_target_doc_filter:
        payload["doc_id"] = mapped_targets[0]
    return payload


def _call_answer_query_with_retries(
    client: httpx.Client,
    args: argparse.Namespace,
    headers: dict[str, str],
    query: Any,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    max_attempts = max(1, int(args.answer_max_retries) + 1)
    attempts: list[dict[str, Any]] = []
    row: dict[str, Any] | None = None
    for attempt_index in range(max_attempts):
        row = _call_answer_query(client, args, headers, query, request_payload)
        attempts.append({
            "attempt": attempt_index + 1,
            "status": row.get("status"),
            "error": row.get("error"),
        })
        if row.get("status") == "ok":
            break
        if attempt_index < max_attempts - 1 and args.answer_retry_cooldown_seconds > 0:
            time.sleep(args.answer_retry_cooldown_seconds)
    assert row is not None
    row["attempt_count"] = len(attempts)
    if len(attempts) > 1:
        row["attempts"] = attempts
    return row


def _call_answer_query(
    client: httpx.Client,
    args: argparse.Namespace,
    headers: dict[str, str],
    query: Any,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    base_row = {
        "query_id": query.query_id,
        "query": query.query,
        "category": query.category,
        "citation_required": query.citation_required,
        "expected_answer_hints": query.expected_answer_hints,
        "request": _redacted_answer_request(request_payload),
    }
    try:
        response = client.post(
            f"{args.query_api_url.rstrip('/')}/query",
            headers=headers or None,
            json=request_payload,
        )
        response.raise_for_status()
        payload = response.json()
        answer_text = payload.get("answer") or ""
        citations = payload.get("citations") or []
        metadata = payload.get("metadata") or {}
        return {
            **base_row,
            "status": "ok",
            "answer_non_empty": bool(answer_text.strip()),
            "citation_present": bool(citations),
            "expected_hint_overlap": _expected_hint_overlap(answer_text, query.expected_answer_hints),
            "model_used": payload.get("model_used"),
            "confidence_score": payload.get("confidence_score"),
            "latency_ms": payload.get("latency_ms"),
            "tokens_used": metadata.get("tokens_used", 0),
            "retrieval_count": metadata.get("retrieval_count"),
            "cache_hit": metadata.get("cache_hit"),
            "retrieval_strategy": metadata.get("retrieval_strategy"),
            "retrieval_candidate_pool": metadata.get("retrieval_candidate_pool"),
            "sec_aware_rerank": metadata.get("sec_aware_rerank"),
            "citation_count": len(citations),
        }
    except httpx.HTTPStatusError as exc:
        return {
            **base_row,
            "status": "failed",
            "error": f"HTTP {exc.response.status_code}: {_response_text(exc.response)}",
            "answer_non_empty": False,
            "citation_present": False,
            "expected_hint_overlap": 0.0,
        }
    except httpx.RequestError as exc:
        return {
            **base_row,
            "status": "failed",
            "error": f"Service unavailable: {exc}",
            "answer_non_empty": False,
            "citation_present": False,
            "expected_hint_overlap": 0.0,
        }


def _expected_hint_overlap(answer: str, hints: list[str]) -> float:
    if not hints:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for hint in hints if hint.lower() in answer_lower)
    return hits / len(hints)


def _required_citation_rate(rows: list[dict[str, Any]]) -> float:
    required = [row for row in rows if row.get("citation_required")]
    if not required:
        return 0.0
    return _rate(row.get("citation_present") for row in required)


def _rate(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(1 for value in values if value) / len(values), 6)


def _redacted_answer_request(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in {"api_key", "authorization"}}
def _auth_headers(args: argparse.Namespace) -> dict[str, str]:
    headers: dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key
    if args.bearer_token:
        headers["Authorization"] = f"Bearer {args.bearer_token}"
    return headers


def _response_text(response: httpx.Response) -> str:
    try:
        return json.dumps(response.json())
    except Exception:
        return response.text



def _pinecone_namespace(args: argparse.Namespace) -> str:
    return args.pinecone_namespace or args.tenant_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate document RAG over a curated PDF corpus.")
    parser.add_argument("mode", choices=["validate-only", "preflight", "ingest", "retrieve", "answer"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--tenant-id", default=os.getenv("DOCUMENT_RAG_EVAL_TENANT_ID", "tenant_eval_local"))
    parser.add_argument("--ingestion-url", default=os.getenv("INGESTION_SERVICE_URL", "http://localhost:8001"))
    parser.add_argument("--status-url", default=os.getenv("INGESTION_SERVICE_URL", "http://localhost:8001"))
    parser.add_argument("--api-key", default=os.getenv("DOCUMENT_RAG_EVAL_API_KEY"))
    parser.add_argument("--bearer-token", default=os.getenv("DOCUMENT_RAG_EVAL_BEARER_TOKEN"))
    parser.add_argument("--skip-file-check", action="store_true")
    parser.add_argument("--poll-status", action="store_true")
    parser.add_argument("--poll-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--pinecone-api-key", default=os.getenv("PINECONE_API_KEY"))
    parser.add_argument("--pinecone-index", default=os.getenv("PINECONE_INDEX"))
    parser.add_argument("--pinecone-namespace", default=os.getenv("PINECONE_NAMESPACE"))
    parser.add_argument("--embedding-model", default=os.getenv("DOCUMENT_RAG_EVAL_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"))
    parser.add_argument("--ingestion-run", type=Path, default=None)
    parser.add_argument("--retrieval-candidate-pool", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--sec-aware-rerank", action="store_true")
    parser.add_argument("--sec-metadata-weight", type=float, default=0.5)
    parser.add_argument("--query-api-url", default=os.getenv("INFERENCE_API_URL", "http://localhost:8000"))
    parser.add_argument("--answer-top-k", type=int, default=5)
    parser.add_argument("--answer-disable-target-doc-filter", action="store_true")
    parser.add_argument("--answer-delay-seconds", type=float, default=0.0)
    parser.add_argument("--answer-retry-failed-from", type=Path, default=None)
    parser.add_argument("--answer-max-retries", type=int, default=0)
    parser.add_argument("--answer-retry-cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--model-choice", default="auto")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--preflight-target", choices=["validate-only", "ingest", "retrieve", "answer", "acquisition"], default="validate-only")
    parser.add_argument("--public-source", choices=["cuad_atticus", "sec_edgar"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "validate-only":
            report = validate_only(args)
        elif args.mode == "preflight":
            report = preflight(args)
        elif args.mode == "ingest":
            report = ingest(args)
        elif args.mode == "retrieve":
            report = retrieve(args)
        elif args.mode == "answer":
            report = answer(args)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    json_path = write_json_report(report, args.output_dir, args.run_id)
    markdown_path = json_path.with_suffix(".md")
    write_markdown_report(markdown_path, report)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    if args.write_csv:
        csv_path = json_path.with_suffix(".csv")
        write_csv_report(csv_path, report)
        print(f"Wrote {csv_path}")

    if args.mode == "ingest" and report.get("ingestion", {}).get("failure_count", 0):
        raise SystemExit(2)
    if args.mode == "preflight" and report.get("preflight", {}).get("required_failure_count", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
