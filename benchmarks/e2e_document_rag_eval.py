import argparse
import json
import mimetypes
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.corpus_manifest import LoadedCorpus, format_corpus_summary, load_corpus_manifest


DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "corpora" / "example_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "corpora" / "results"
DEFAULT_PDF_ROOT = REPO_ROOT / "benchmarks" / "corpora" / "local_pdfs"
INGESTION_SUPPORTED_DOC_TYPES = {"legal_contract", "financial_report"}
TERMINAL_STATUSES = {"ocr_complete", "embedding_complete", "indexed", "completed", "failed"}


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
            "tenant_id": args.tenant_id,
        },
        "limitations": [
            "This harness runs against local or explicitly configured services; it is not production evidence.",
            "Raw PDFs are expected to live in an ignored local directory and are not committed by default.",
            "Manifest labels may be incomplete until real indexing produces page or chunk labels.",
            "Ingestion results do not prove retrieval quality, legal correctness, financial correctness, QPS, uptime, or cost savings.",
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


def write_json_report(report: dict[str, Any], output_dir: Path, run_id: str | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"document_rag_eval_{report['mode']}_{suffix}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate document RAG over a curated PDF corpus.")
    parser.add_argument("mode", choices=["validate-only", "ingest"])
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "validate-only":
        report = validate_only(args)
    elif args.mode == "ingest":
        report = ingest(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    path = write_json_report(report, args.output_dir, args.run_id)
    print(f"Wrote {path}")

    if args.mode == "ingest" and report.get("ingestion", {}).get("failure_count", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
