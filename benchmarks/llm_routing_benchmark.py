import argparse
import csv
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "services" / "llm-orchestrator"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from complexity_analyzer import QueryComplexityAnalyzer
from utils import ModelRouter


MODEL_EXPENSIVE = "gemini"
MODEL_CHEAP = "mistral"
BENCHMARK_MODE = "mock_synthetic"
DEFAULT_WORKLOAD = REPO_ROOT / "benchmarks" / "data_samples" / "llm_routing_workload.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results"


@dataclass(frozen=True)
class BenchmarkSettings:
    complexity_threshold: float = 0.7


@dataclass(frozen=True)
class QueryResult:
    id: str
    category: str
    strategy: str
    query: str
    doc_type: str
    primary_model: str
    selected_model: str
    routing_reason: str
    fallback_used: bool
    cache_hit: bool
    estimated_input_tokens: int
    estimated_output_tokens: int
    billable_input_tokens: int
    billable_output_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    answer_non_empty: bool
    requires_citation: bool
    citation_present: bool
    keyword_overlap: float
    expected_keywords: list[str]


def estimate_query_tokens(text: str) -> int:
    return max(1, len(text) // 4)


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


def load_workload(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Workload must be a non-empty JSON list")
    return data


def select_for_strategy(router: ModelRouter, strategy: str, item: dict[str, Any]) -> tuple[str, str]:
    if strategy == "always_expensive":
        return MODEL_EXPENSIVE, "baseline_always_expensive"
    if strategy == "always_cheap":
        return MODEL_CHEAP, "baseline_always_cheap"
    if strategy == "heuristic":
        decision = router.route(
            item["query"],
            context_length=int(item["context_tokens"]),
            doc_type=item["doc_type"],
        )
        return decision.model, decision.reason
    raise ValueError(f"Unsupported strategy: {strategy}")


def latency_estimate_ms(model: str, category: str, input_tokens: int, output_tokens: int, cache_hit: bool) -> float:
    if cache_hit:
        return 5.0

    base_by_model = {
        MODEL_EXPENSIVE: 620.0,
        MODEL_CHEAP: 340.0,
    }
    token_slope_by_model = {
        MODEL_EXPENSIVE: 0.34,
        MODEL_CHEAP: 0.22,
    }
    category_multiplier = {
        "simple_factual": 0.75,
        "medium_document_qa": 1.0,
        "complex_legal_financial": 1.25,
        "citation_heavy": 1.2,
        "long_context": 1.55,
        "adversarial_ambiguous": 1.05,
    }.get(category, 1.0)

    return round(
        (base_by_model[model] + (input_tokens + output_tokens) * token_slope_by_model[model])
        * category_multiplier,
        2,
    )


def synthetic_quality_proxy(model: str, item: dict[str, Any]) -> tuple[bool, bool, float]:
    expected = item.get("expected_keywords", [])
    if not expected:
        return True, not item.get("requires_citation", False), 1.0

    category = item["category"]
    if model == MODEL_EXPENSIVE:
        covered = len(expected)
    elif category in {"simple_factual", "medium_document_qa"}:
        covered = len(expected)
    elif category == "citation_heavy":
        covered = max(1, len(expected) - 1)
    elif category == "long_context":
        covered = max(1, len(expected) - 2)
    elif category == "adversarial_ambiguous":
        covered = max(1, len(expected) - 1)
    else:
        covered = max(1, len(expected) - 2)

    keyword_overlap = covered / len(expected)
    citation_present = bool(item.get("requires_citation")) and (
        model == MODEL_EXPENSIVE or category in {"simple_factual", "medium_document_qa"}
    )
    if not item.get("requires_citation"):
        citation_present = False

    return True, citation_present, round(keyword_overlap, 4)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round((pct / 100.0) * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def run_strategy(router: ModelRouter, workload: list[dict[str, Any]], strategy: str) -> list[QueryResult]:
    results: list[QueryResult] = []
    cache: set[tuple[str, str, str, int]] = set()

    for item in workload:
        primary_model, reason = select_for_strategy(router, strategy, item)
        selected_model = primary_model
        fallback_used = False

        failed_models = set(item.get("simulate_failure_for", []))
        if primary_model in failed_models:
            selected_model = MODEL_CHEAP if primary_model == MODEL_EXPENSIVE else MODEL_EXPENSIVE
            fallback_used = True

        estimated_input_tokens = (
            estimate_query_tokens(item["query"])
            + int(item["context_tokens"])
            + 220
        )
        estimated_output_tokens = int(item.get("expected_output_tokens", 256))

        cache_key = (
            strategy,
            selected_model,
            item["query"],
            item["doc_type"],
            int(item["context_tokens"]),
        )
        cache_hit = cache_key in cache
        cache.add(cache_key)

        billable_input_tokens = 0 if cache_hit else estimated_input_tokens
        billable_output_tokens = 0 if cache_hit else estimated_output_tokens
        estimated_cost = router.estimate_cost(selected_model, billable_input_tokens, billable_output_tokens)
        latency_ms = latency_estimate_ms(
            selected_model,
            item["category"],
            estimated_input_tokens,
            estimated_output_tokens,
            cache_hit,
        )
        if fallback_used:
            latency_ms += 250.0

        answer_non_empty, citation_present, keyword_overlap = synthetic_quality_proxy(selected_model, item)

        results.append(QueryResult(
            id=item["id"],
            category=item["category"],
            strategy=strategy,
            query=item["query"],
            doc_type=item["doc_type"],
            primary_model=primary_model,
            selected_model=selected_model,
            routing_reason=reason,
            fallback_used=fallback_used,
            cache_hit=cache_hit,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            billable_input_tokens=billable_input_tokens,
            billable_output_tokens=billable_output_tokens,
            estimated_cost_usd=round(estimated_cost, 8),
            latency_ms=round(latency_ms, 2),
            answer_non_empty=answer_non_empty,
            requires_citation=bool(item.get("requires_citation")),
            citation_present=citation_present,
            keyword_overlap=keyword_overlap,
            expected_keywords=list(item.get("expected_keywords", [])),
        ))

    return results


def summarize(results: list[QueryResult]) -> dict[str, Any]:
    latencies = [r.latency_ms for r in results]
    citation_required = [r for r in results if r.requires_citation]
    return {
        "query_count": len(results),
        "category_counts": dict(Counter(r.category for r in results)),
        "model_counts": dict(Counter(r.selected_model for r in results)),
        "estimated_total_cost_usd": round(sum(r.estimated_cost_usd for r in results), 8),
        "estimated_cost_per_query_usd": round(mean([r.estimated_cost_usd for r in results]), 8),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
        },
        "failure_rate": 0.0,
        "cache_hit_rate": round(sum(1 for r in results if r.cache_hit) / max(1, len(results)), 4),
        "fallback_count": sum(1 for r in results if r.fallback_used),
        "quality_proxy": {
            "answer_non_empty_rate": round(sum(1 for r in results if r.answer_non_empty) / max(1, len(results)), 4),
            "citation_presence_rate": round(
                sum(1 for r in citation_required if r.citation_present) / max(1, len(citation_required)),
                4,
            ),
            "avg_expected_keyword_overlap": round(mean([r.keyword_overlap for r in results]), 4),
        },
    }


def build_report(workload: list[dict[str, Any]], strategies: dict[str, list[QueryResult]], command: str) -> dict[str, Any]:
    return {
        "benchmark_name": "llm_routing_mock_benchmark",
        "mode": BENCHMARK_MODE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "command": command,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "runner": "standard-library synthetic benchmark",
        },
        "workload": {
            "query_count": len(workload),
            "category_counts": dict(Counter(item["category"] for item in workload)),
            "workload_file": str(DEFAULT_WORKLOAD.relative_to(REPO_ROOT)),
        },
        "cost_model": {
            "unit": "USD per 1M tokens",
            "gemini": {"input": 3.50, "output": 10.50},
            "mistral": {"input": 0.70, "output": 2.10},
        },
        "strategies": {
            name: {
                "summary": summarize(results),
                "queries": [asdict(result) for result in results],
            }
            for name, results in strategies.items()
        },
        "limitations": [
            "This benchmark is mock/synthetic and does not call real LLM providers.",
            "Latency values are deterministic estimates, not measured provider latency.",
            "The quality proxy checks non-empty answers, citation-marker presence, and expected keyword overlap only.",
            "The quality proxy is not a semantic evaluation and must not be presented as model accuracy.",
            "Costs are estimates from the static cost table in the router benchmark, not billing records.",
            "The workload is fixed and small; it is intended for reproducibility and smoke comparison, not broad evaluation.",
        ],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_csv(path: Path, strategies: dict[str, list[QueryResult]]) -> None:
    rows = [asdict(result) for results in strategies.values() for result in results]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# LLM Routing Mock Benchmark",
        "",
        f"Mode: `{report['mode']}`",
        f"Timestamp UTC: `{report['timestamp_utc']}`",
        f"Git commit: `{report['git_commit']}`",
        f"Command: `{report['command']}`",
        "",
        "## Summary",
        "",
        "| Strategy | Queries | Cost USD | Cost/Query USD | p50 ms | p95 ms | p99 ms | Cache Hit Rate | Fallbacks | Keyword Overlap | Citation Presence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, payload in report["strategies"].items():
        summary = payload["summary"]
        quality = summary["quality_proxy"]
        latency = summary["latency_ms"]
        lines.append(
            f"| {name} | {summary['query_count']} | "
            f"{summary['estimated_total_cost_usd']:.8f} | "
            f"{summary['estimated_cost_per_query_usd']:.8f} | "
            f"{latency['p50']:.2f} | {latency['p95']:.2f} | {latency['p99']:.2f} | "
            f"{summary['cache_hit_rate']:.4f} | {summary['fallback_count']} | "
            f"{quality['avg_expected_keyword_overlap']:.4f} | {quality['citation_presence_rate']:.4f} |"
        )

    lines.extend([
        "",
        "## Per-Query Routing",
        "",
        "| Strategy | Query ID | Category | Primary | Selected | Reason | Input Tokens | Output Tokens | Cost USD | Latency ms | Cache |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---|",
    ])

    for name, payload in report["strategies"].items():
        for result in payload["queries"]:
            lines.append(
                f"| {name} | {result['id']} | {result['category']} | {result['primary_model']} | "
                f"{result['selected_model']} | {result['routing_reason']} | "
                f"{result['estimated_input_tokens']} | {result['estimated_output_tokens']} | "
                f"{result['estimated_cost_usd']:.8f} | {result['latency_ms']:.2f} | {result['cache_hit']} |"
            )

    lines.extend([
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mock/synthetic LLM routing benchmark.")
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default=None, help="Optional stable suffix for output file names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = load_workload(args.workload)
    router = ModelRouter(BenchmarkSettings(), QueryComplexityAnalyzer())

    strategies = {
        "always_expensive": run_strategy(router, workload, "always_expensive"),
        "always_cheap": run_strategy(router, workload, "always_cheap"),
        "heuristic": run_strategy(router, workload, "heuristic"),
    }

    command = "python " + " ".join(sys.argv)
    report = build_report(workload, strategies, command)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"llm_routing_benchmark_{suffix}.json"
    markdown_path = args.output_dir / f"llm_routing_benchmark_{suffix}.md"
    csv_path = args.output_dir / f"llm_routing_benchmark_{suffix}.csv"

    write_json(json_path, report)
    write_markdown(markdown_path, report)
    write_csv(csv_path, strategies)

    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
