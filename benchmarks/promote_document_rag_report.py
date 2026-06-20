import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


CONTENT_KEYS = {
    "answer",
    "answer_text",
    "raw_answer",
    "raw_response",
    "response",
    "request",
    "text",
    "text_preview",
    "context",
    "contexts",
    "document_text",
    "chunk_text",
}
PATH_KEYS = {"manifest_path", "pdf_root", "file_root", "target_path", "output_pdf_dir", "output_file_dir"}
QUERY_TEXT_KEYS = {"query"}
SECRET_PATTERNS = [re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|secret)")]
ABSOLUTE_WINDOWS_PATH = re.compile(r"[A-Za-z]:\\[^\s`|]+")
ABSOLUTE_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9_])/(?:[^\s`|]+/)+[^\s`|]+")


class ReportPromotionError(ValueError):
    """Raised when a report cannot be promoted safely."""


def promote_report(
    report_path: Path,
    *,
    output_markdown: Path,
    output_json: Path | None = None,
    allow_private_summary: bool = False,
    allow_query_text: bool = False,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    corpus_mode = report.get("corpus", {}).get("mode")
    if corpus_mode == "private_local" and not allow_private_summary:
        raise ReportPromotionError("Refusing to promote private_local report without --allow-private-summary")

    sanitized = _sanitize_report(report, allow_query_text=allow_query_text)
    sanitized["sanitization"] = {
        "source_report": report_path.name,
        "private_summary_allowed": allow_private_summary,
        "query_text_allowed": allow_query_text,
        "removed_content_keys": sorted(CONTENT_KEYS - ({"query"} if allow_query_text else set())),
        "path_redaction": "absolute paths and local corpus roots redacted",
    }
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_markdown_summary(sanitized), encoding="utf-8")
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    return sanitized


def _sanitize_report(report: dict[str, Any], *, allow_query_text: bool) -> dict[str, Any]:
    sanitized = deepcopy(report)
    sanitized = _sanitize_value(sanitized, allow_query_text=allow_query_text)
    corpus = sanitized.get("corpus") if isinstance(sanitized, dict) else None
    if isinstance(corpus, dict):
        for key in list(PATH_KEYS):
            corpus.pop(key, None)
        documents = corpus.get("documents")
        if isinstance(documents, list):
            corpus["documents"] = [
                {
                    key: value
                    for key, value in doc.items()
                    if key in {"document_id", "doc_type", "source_format", "allowed_to_commit", "exists"}
                }
                for doc in documents
                if isinstance(doc, dict)
            ]
    return sanitized


def _sanitize_value(value: Any, *, allow_query_text: bool) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if _is_secret_key(key):
                cleaned[key] = "[redacted-secret-field]"
                continue
            if key in CONTENT_KEYS:
                if key == "answer" and isinstance(item, dict):
                    cleaned[key] = _sanitize_value(item, allow_query_text=allow_query_text)
                else:
                    cleaned[key] = "[removed-content-field]"
                continue
            if key in QUERY_TEXT_KEYS and not allow_query_text:
                cleaned[key] = "[removed-query-text]"
                continue
            if key in PATH_KEYS:
                cleaned[key] = _path_label(item)
                continue
            cleaned[key] = _sanitize_value(item, allow_query_text=allow_query_text)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_value(item, allow_query_text=allow_query_text) for item in value]
    if isinstance(value, str):
        return _redact_paths(value)
    return value


def _is_secret_key(key: str) -> bool:
    return any(pattern.search(key) for pattern in SECRET_PATTERNS)


def _path_label(value: Any) -> str:
    if not isinstance(value, str):
        return "[redacted-local-path]"
    path = Path(value)
    name = path.name or value.rstrip("/\\").split("/")[-1].split("\\")[-1]
    return f"[redacted-local-path: {name}]"


def _redact_paths(value: str) -> str:
    value = ABSOLUTE_WINDOWS_PATH.sub("[redacted-local-path]", value)
    value = ABSOLUTE_POSIX_PATH.sub("[redacted-local-path]", value)
    return value


def _markdown_summary(report: dict[str, Any]) -> str:
    corpus = report.get("corpus", {})
    lines = [
        "# Sanitized Document RAG Evaluation Summary",
        "",
        f"Source mode: `{corpus.get('mode', 'unknown')}`",
        f"Evaluation mode: `{report.get('mode', 'unknown')}`",
        f"Git commit: `{report.get('git_commit', 'unknown')}`",
        f"Timestamp UTC: `{report.get('timestamp_utc', 'unknown')}`",
        f"Corpus: `{corpus.get('corpus_name', 'unknown')}` (`{corpus.get('corpus_id', 'unknown')}`)",
        f"Documents: `{corpus.get('document_count', 'unknown')}`",
        f"Queries: `{corpus.get('query_count', 'unknown')}`",
        "",
        "## Evidence Type",
        "",
        f"- Report mode: `{report.get('mode', 'unknown')}`",
        f"- Corpus mode: `{corpus.get('mode', 'unknown')}`",
        "- This is a sanitized local report summary, not production evidence.",
        "",
    ]
    if "retrieval" in report:
        metrics = report["retrieval"].get("overall", {})
        lines.extend([
            "## Retrieval Metrics",
            "",
            "| Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |",
            "|---:|---:|---:|---:|---:|",
            f"| {_fmt(metrics.get('recall@1'))} | {_fmt(metrics.get('recall@3'))} | {_fmt(metrics.get('recall@5'))} | {_fmt(metrics.get('mrr'))} | {_fmt(metrics.get('ndcg@5'))} |",
            "",
        ])
    if "preflight" in report:
        preflight = report["preflight"]
        lines.extend([
            "## Preflight",
            "",
            f"Status: `{preflight.get('status')}`",
            f"Required failures: `{preflight.get('required_failure_count')}`",
            f"Warnings: `{preflight.get('warning_count')}`",
            "",
        ])
    if "ingestion" in report:
        ingestion = report["ingestion"]
        lines.extend([
            "## Ingestion",
            "",
            f"Successes: `{ingestion.get('success_count')}`",
            f"Failures: `{ingestion.get('failure_count')}`",
            f"Skipped: `{ingestion.get('skipped_count')}`",
            "",
        ])
    if "answer" in report:
        answer = report["answer"]
        lines.extend([
            "## Answer Proxy",
            "",
            f"Non-empty answer rate: `{answer.get('non_empty_answer_rate')}`",
            f"Citation presence rate for required citations: `{answer.get('citation_presence_rate_required')}`",
            f"Average expected-hint overlap: `{answer.get('average_expected_hint_overlap')}`",
            "",
        ])
    lines.extend([
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
    lines.extend([
        "",
        "## Sanitization",
        "",
    ])
    sanitization = report.get("sanitization", {})
    lines.extend(f"- `{key}`: `{value}`" for key, value in sanitization.items())
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "n/a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a local document RAG report into a sanitized public summary.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--allow-private-summary", action="store_true")
    parser.add_argument("--allow-query-text", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sanitized = promote_report(
            args.report,
            output_markdown=args.output_md,
            output_json=args.output_json,
            allow_private_summary=args.allow_private_summary,
            allow_query_text=args.allow_query_text,
        )
    except ReportPromotionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps({"status": "promoted", "mode": sanitized.get("mode"), "corpus_mode": sanitized.get("corpus", {}).get("mode")}, indent=2))


if __name__ == "__main__":
    main()
