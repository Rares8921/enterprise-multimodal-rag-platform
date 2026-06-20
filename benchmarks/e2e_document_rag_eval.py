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

from benchmarks.corpus_manifest import LoadedCorpus, format_corpus_summary, load_corpus_manifest


DEFAULT_MANIFEST = REPO_ROOT / "benchmarks" / "corpora" / "example_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "corpora" / "results"
DEFAULT_PDF_ROOT = REPO_ROOT / "benchmarks" / "corpora" / "local_pdfs"
HYBRID_MODULE_PATH = REPO_ROOT / "services" / "inference-api" / "utils" / "hybrid_retrieval.py"
INGESTION_SUPPORTED_DOC_TYPES = {"legal_contract", "financial_report"}
TERMINAL_STATUSES = {"ocr_complete", "embedding_complete", "indexed", "completed", "failed"}
METRIC_KEYS = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]

def _load_hybrid_module():
    spec = importlib.util.spec_from_file_location("document_rag_eval_hybrid_utils", HYBRID_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["document_rag_eval_hybrid_utils"] = module
    spec.loader.exec_module(module)
    return module


hybrid_utils = _load_hybrid_module()
hybrid_rerank = hybrid_utils.hybrid_rerank


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
            "tenant_id": args.tenant_id,
        },
        "limitations": [
            "This harness runs against local or explicitly configured services; it is not production evidence.",
            "Raw PDFs are expected to live in an ignored local directory and are not committed by default.",
            "Manifest labels may be incomplete until real indexing produces page or chunk labels.",
            "Ingestion and retrieval results do not prove legal correctness, financial correctness, QPS, uptime, or cost savings.",
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
        reranked = hybrid_rerank(
            query.query,
            matches,
            vector_weight=args.vector_weight,
            bm25_weight=args.bm25_weight,
        )
        rows.append(_evaluate_retrieval_query(query, reranked, ingestion_mapping, args.top_k))

    report["retrieval"] = {
        "mode": "pinecone_real_service",
        "strategy": "pinecone_vector_candidates_plus_bm25_hybrid_rerank",
        "top_k": args.top_k,
        "candidate_pool_size": args.retrieval_candidate_pool,
        "vector_weight": args.vector_weight,
        "bm25_weight": args.bm25_weight,
        "ingestion_run": str(args.ingestion_run) if args.ingestion_run else None,
        "overall": _average_metrics(rows),
        "by_category": _category_metrics(rows),
        "label_granularity_counts": dict(Counter(row["label_granularity"] for row in rows)),
        "candidate_pool_miss_count": sum(1 for row in rows if not row["candidate_pool_contains_relevant"]),
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
        "Document-level labels are weaker than page-level or chunk-level labels; metric interpretation depends on manifest label granularity.",
        "This is local real-service evidence only and must not be described as production retrieval quality.",
    ])
    return report
def write_json_report(report: dict[str, Any], output_dir: Path, run_id: str | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"document_rag_eval_{report['mode']}_{suffix}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path



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
) -> dict[str, Any]:
    target_document_ids = set(query.target_document_ids)
    mapped_target_ids = {ingestion_mapping.get(document_id, document_id) for document_id in target_document_ids}
    relevant_pages = set(query.relevant_pages)
    relevant_chunk_ids = {str(chunk_id) for chunk_id in query.relevant_chunk_ids}
    label_granularity = _label_granularity(query)
    relevant_count = _relevant_count(query)

    all_results = []
    for rank, match in enumerate(matches, start=1):
        metadata = match.get("metadata") or {}
        result = {
            "rank": rank,
            "id": match.get("id"),
            "score": match.get("hybrid_score", match.get("score", 0.0)),
            "vector_score": match.get("vector_score", match.get("score", 0.0)),
            "bm25_score": match.get("bm25_score", 0.0),
            "doc_id": metadata.get("doc_id"),
            "manifest_document_id": metadata.get("manifest_document_id"),
            "filename": metadata.get("filename"),
            "page": metadata.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "text_preview": (metadata.get("text") or "")[:240],
        }
        result["is_relevant"] = _is_relevant_result(
            result,
            target_document_ids,
            mapped_target_ids,
            relevant_pages,
            relevant_chunk_ids,
            label_granularity,
        )
        all_results.append(result)

    top_results = all_results[:top_k]
    ranked_relevance = [result["is_relevant"] for result in top_results]
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
        "relevant_chunk_ids": query.relevant_chunk_ids,
        "label_granularity": label_granularity,
        "candidate_pool_contains_relevant": any(result["is_relevant"] for result in all_results),
        "metrics": metrics,
        "top_results": top_results,
    }
def _label_granularity(query: Any) -> str:
    if query.relevant_chunk_ids:
        return "chunk"
    if query.relevant_pages:
        return "page"
    return "document"


def _relevant_count(query: Any) -> int:
    if query.relevant_chunk_ids:
        return len(set(query.relevant_chunk_ids))
    if query.relevant_pages:
        return max(1, len(set(query.relevant_pages)))
    return max(1, len(set(query.target_document_ids)))


def _is_relevant_result(
    result: dict[str, Any],
    target_document_ids: set[str],
    mapped_target_ids: set[str],
    relevant_pages: set[int],
    relevant_chunk_ids: set[str],
    label_granularity: str,
) -> bool:
    doc_candidates = {
        str(value)
        for value in [result.get("doc_id"), result.get("manifest_document_id"), result.get("filename")]
        if value is not None
    }
    doc_match = bool(doc_candidates & (target_document_ids | mapped_target_ids))
    if label_granularity == "document":
        return doc_match
    if label_granularity == "page":
        return doc_match and result.get("page") in relevant_pages
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
    parser.add_argument("mode", choices=["validate-only", "ingest", "retrieve"])
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "validate-only":
            report = validate_only(args)
        elif args.mode == "ingest":
            report = ingest(args)
        elif args.mode == "retrieve":
            report = retrieve(args)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    path = write_json_report(report, args.output_dir, args.run_id)
    print(f"Wrote {path}")

    if args.mode == "ingest" and report.get("ingestion", {}).get("failure_count", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
