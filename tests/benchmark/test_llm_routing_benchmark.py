import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.llm_routing_benchmark import (
    BenchmarkSettings,
    build_report,
    load_workload,
    run_strategy,
    write_csv,
    write_json,
    write_markdown,
)
from complexity_analyzer import QueryComplexityAnalyzer
from utils import ModelRouter


@pytest.fixture()
def workload():
    return load_workload(Path("benchmarks/data_samples/llm_routing_workload.json"))


@pytest.fixture()
def router():
    return ModelRouter(BenchmarkSettings(), QueryComplexityAnalyzer())


@pytest.mark.benchmark
def test_llm_routing_workload_has_required_categories(workload):
    categories = {item["category"] for item in workload}

    assert len(workload) == 13
    assert {
        "simple_factual",
        "medium_document_qa",
        "complex_legal_financial",
        "citation_heavy",
        "long_context",
        "adversarial_ambiguous",
    }.issubset(categories)
    assert any(item.get("duplicate_of") == "q001" for item in workload)


@pytest.mark.benchmark
def test_llm_routing_strategies_produce_comparable_summaries(workload, router):
    strategies = {
        "always_expensive": run_strategy(router, workload, "always_expensive"),
        "always_cheap": run_strategy(router, workload, "always_cheap"),
        "heuristic": run_strategy(router, workload, "heuristic"),
    }
    report = build_report(workload, strategies, "python benchmarks/llm_routing_benchmark.py --run-id test")

    assert report["mode"] == "mock_synthetic"
    assert set(report["strategies"]) == {"always_expensive", "always_cheap", "heuristic"}

    heuristic_counts = report["strategies"]["heuristic"]["summary"]["model_counts"]
    assert heuristic_counts["gemini"] > 0
    assert heuristic_counts["mistral"] > 0

    cheap_cost = report["strategies"]["always_cheap"]["summary"]["estimated_total_cost_usd"]
    expensive_cost = report["strategies"]["always_expensive"]["summary"]["estimated_total_cost_usd"]
    heuristic_cost = report["strategies"]["heuristic"]["summary"]["estimated_total_cost_usd"]
    assert cheap_cost <= heuristic_cost <= expensive_cost

    for payload in report["strategies"].values():
        summary = payload["summary"]
        assert summary["query_count"] == len(workload)
        assert {"p50", "p95", "p99"}.issubset(summary["latency_ms"])
        assert "quality_proxy" in summary


@pytest.mark.benchmark
def test_llm_routing_benchmark_writes_json_markdown_and_csv(workload, router, temp_dir):
    strategies = {
        "always_expensive": run_strategy(router, workload, "always_expensive"),
        "always_cheap": run_strategy(router, workload, "always_cheap"),
        "heuristic": run_strategy(router, workload, "heuristic"),
    }
    report = build_report(workload, strategies, "python benchmarks/llm_routing_benchmark.py --run-id test")

    json_path = temp_dir / "report.json"
    markdown_path = temp_dir / "report.md"
    csv_path = temp_dir / "report.csv"
    write_json(json_path, report)
    write_markdown(markdown_path, report)
    write_csv(csv_path, strategies)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")

    assert loaded["benchmark_name"] == "llm_routing_mock_benchmark"
    assert "This benchmark is mock/synthetic" in " ".join(loaded["limitations"])
    assert "LLM Routing Mock Benchmark" in markdown
    assert "always_expensive" in markdown
    assert csv_text.startswith("id,category,strategy")
