import argparse
import csv
import importlib.util
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_PATH = REPO_ROOT / "benchmarks" / "data_samples" / "retrieval_documents.json"
QUERIES_PATH = REPO_ROOT / "benchmarks" / "data_samples" / "retrieval_queries.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results"
HYBRID_MODULE_PATH = REPO_ROOT / "services" / "inference-api" / "utils" / "hybrid_retrieval.py"
BENCHMARK_MODE = "synthetic_offline"
VECTOR_SIMULATOR = "semantic_terms_bow_cosine"


def _load_hybrid_module():
    spec = importlib.util.spec_from_file_location("retrieval_benchmark_hybrid_utils", HYBRID_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["retrieval_benchmark_hybrid_utils"] = module
    spec.loader.exec_module(module)
    return module


hybrid_utils = _load_hybrid_module()
bm25_scores = hybrid_utils.bm25_scores
tokenize = hybrid_utils.tokenize


@dataclass(frozen=True)
class Strategy:
    name: str
    vector_weight: float
    bm25_weight: float


STRATEGIES = [
    Strategy("vector_only", 1.0, 0.0),
    Strategy("bm25_only", 0.0, 1.0),
    Strategy("hybrid_70_30", 0.7, 0.3),
    Strategy("hybrid_50_50", 0.5, 0.5),
    Strategy("hybrid_30_70", 0.3, 0.7),
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_strategy(strategy: Strategy) -> None:
    if strategy.vector_weight < 0 or strategy.bm25_weight < 0:
        raise ValueError("Strategy weights must be non-negative")
    if strategy.vector_weight == 0 and strategy.bm25_weight == 0:
        raise ValueError("At least one strategy weight must be positive")


def flatten_chunks(documents_payload: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()

    for document in documents_payload.get("documents", []):
        required_doc_keys = {"doc_id", "filename", "doc_type", "chunks"}
        missing_doc_keys = required_doc_keys - set(document)
        if missing_doc_keys:
            raise ValueError(f"Document missing keys: {sorted(missing_doc_keys)}")

        for chunk in document["chunks"]:
            required_chunk_keys = {"chunk_id", "page", "section", "text", "semantic_terms"}
            missing_chunk_keys = required_chunk_keys - set(chunk)
            if missing_chunk_keys:
                raise ValueError(f"Chunk missing keys: {sorted(missing_chunk_keys)}")
            if chunk["chunk_id"] in seen:
                raise ValueError(f"Duplicate chunk_id: {chunk['chunk_id']}")
            seen.add(chunk["chunk_id"])

            chunks.append({
                **chunk,
                "doc_id": document["doc_id"],
                "filename": document["filename"],
                "doc_type": document["doc_type"],
                "document_title": document.get("title", ""),
            })

    if not chunks:
        raise ValueError("No chunks found in retrieval document fixture")
    return chunks


def validate_queries(queries_payload: dict[str, Any], chunk_ids: set[str]) -> list[dict[str, Any]]:
    queries = queries_payload.get("queries", [])
    if not queries:
        raise ValueError("No queries found in retrieval query fixture")

    seen: set[str] = set()
    for query in queries:
        required_query_keys = {"query_id", "category", "query", "semantic_terms", "relevant_chunk_ids"}
        missing_query_keys = required_query_keys - set(query)
        if missing_query_keys:
            raise ValueError(f"Query missing keys: {sorted(missing_query_keys)}")
        if query["query_id"] in seen:
            raise ValueError(f"Duplicate query_id: {query['query_id']}")
        seen.add(query["query_id"])
        if not query["relevant_chunk_ids"]:
            raise ValueError(f"Query {query['query_id']} has no relevant chunks")
        missing = [chunk_id for chunk_id in query["relevant_chunk_ids"] if chunk_id not in chunk_ids]
        if missing:
            raise ValueError(f"Query {query['query_id']} references missing chunks: {missing}")

    return queries


def semantic_tokens(terms: list[str]) -> list[str]:
    tokens: list[str] = []
    for term in terms:
        tokens.extend(tokenize(term))
    return [token for token in tokens if not token.replace(".", "", 1).isdigit()]


def cosine_score(query_terms: list[str], chunk_terms: list[str]) -> float:
    query_counts = Counter(semantic_tokens(query_terms))
    chunk_counts = Counter(semantic_tokens(chunk_terms))
    if not query_counts or not chunk_counts:
        return 0.0

    shared = set(query_counts) & set(chunk_counts)
    dot = sum(query_counts[token] * chunk_counts[token] for token in shared)
    query_norm = math.sqrt(sum(value * value for value in query_counts.values()))
    chunk_norm = math.sqrt(sum(value * value for value in chunk_counts.values()))
    if query_norm == 0 or chunk_norm == 0:
        return 0.0
    return dot / (query_norm * chunk_norm)


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def rank_query(
    query: dict[str, Any],
    chunks: list[dict[str, Any]],
    strategy: Strategy,
    *,
    candidate_pool_size: int,
) -> dict[str, Any]:
    validate_strategy(strategy)
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")

    vector_scores = [
        cosine_score(query.get("semantic_terms", []), chunk.get("semantic_terms", []))
        for chunk in chunks
    ]
    vector_ranked_indices = sorted(range(len(chunks)), key=lambda idx: vector_scores[idx], reverse=True)
    candidate_indices = vector_ranked_indices[: min(candidate_pool_size, len(chunks))]

    candidate_chunks = [chunks[idx] for idx in candidate_indices]
    candidate_vector_scores = [vector_scores[idx] for idx in candidate_indices]
    candidate_bm25_scores = bm25_scores(query["query"], [chunk["text"] for chunk in candidate_chunks])

    normalized_vector = normalize(candidate_vector_scores)
    normalized_bm25 = normalize(candidate_bm25_scores)
    total_weight = strategy.vector_weight + strategy.bm25_weight

    scored_candidates = []
    for chunk, vector_score, bm25_score, norm_vector, norm_bm25 in zip(
        candidate_chunks,
        candidate_vector_scores,
        candidate_bm25_scores,
        normalized_vector,
        normalized_bm25,
    ):
        combined_score = (
            (strategy.vector_weight / total_weight) * norm_vector
            + (strategy.bm25_weight / total_weight) * norm_bm25
        )
        scored_candidates.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "doc_type": chunk["doc_type"],
            "section": chunk["section"],
            "vector_score": round(vector_score, 6),
            "bm25_score": round(bm25_score, 6),
            "score": round(combined_score, 6),
            "text": chunk["text"],
        })

    ranked = sorted(scored_candidates, key=lambda item: item["score"], reverse=True)
    relevant = set(query["relevant_chunk_ids"])
    relevant_ranks = {
        item["chunk_id"]: rank
        for rank, item in enumerate(ranked, start=1)
        if item["chunk_id"] in relevant
    }

    return {
        "query_id": query["query_id"],
        "category": query["category"],
        "query": query["query"],
        "strategy": strategy.name,
        "relevant_chunk_ids": query["relevant_chunk_ids"],
        "candidate_pool_size": len(candidate_indices),
        "candidate_pool_contains_relevant": all(chunk_id in {chunks[idx]["chunk_id"] for idx in candidate_indices} for chunk_id in relevant),
        "relevant_ranks": relevant_ranks,
        "top_results": [
            {
                **item,
                "is_relevant": item["chunk_id"] in relevant,
            }
            for item in ranked[:5]
        ],
    }


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def metrics_for_result(result: dict[str, Any]) -> dict[str, float]:
    ranked_ids = [item["chunk_id"] for item in result["top_results"]]
    relevant_ids = set(result["relevant_chunk_ids"])
    return {
        "recall@1": recall_at_k(ranked_ids, relevant_ids, 1),
        "recall@3": recall_at_k(ranked_ids, relevant_ids, 3),
        "recall@5": recall_at_k(ranked_ids, relevant_ids, 5),
        "mrr": reciprocal_rank(ranked_ids, relevant_ids),
        "ndcg@5": ndcg_at_k(ranked_ids, relevant_ids, 5),
    }


def average_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]}

    metric_rows = [row["metrics"] for row in rows]
    return {
        key: round(mean(metric[key] for metric in metric_rows), 6)
        for key in metric_rows[0]
    }


def summarize_strategy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    missed_top5 = [
        {
            "query_id": row["query_id"],
            "category": row["category"],
            "relevant_chunk_ids": row["relevant_chunk_ids"],
            "top_chunk_ids": [item["chunk_id"] for item in row["top_results"]],
        }
        for row in rows
        if row["metrics"]["recall@5"] == 0.0
    ]

    return {
        "query_count": len(rows),
        "overall": average_metrics(rows),
        "by_category": {
            category: {
                "query_count": len(category_rows),
                "metrics": average_metrics(category_rows),
            }
            for category, category_rows in sorted(by_category.items())
        },
        "missed_queries_top5": missed_top5,
        "candidate_pool_miss_count": sum(1 for row in rows if not row["candidate_pool_contains_relevant"]),
    }


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


def run_benchmark(
    documents_path: Path,
    queries_path: Path,
    *,
    candidate_pool_size: int,
) -> dict[str, Any]:
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")

    documents_payload = read_json(documents_path)
    queries_payload = read_json(queries_path)
    chunks = flatten_chunks(documents_payload)
    queries = validate_queries(queries_payload, {chunk["chunk_id"] for chunk in chunks})

    strategies_payload: dict[str, Any] = {}
    for strategy in STRATEGIES:
        rows = []
        for query in queries:
            row = rank_query(query, chunks, strategy, candidate_pool_size=candidate_pool_size)
            row["metrics"] = metrics_for_result(row)
            rows.append(row)
        strategies_payload[strategy.name] = {
            "strategy": {
                "vector_weight": strategy.vector_weight,
                "bm25_weight": strategy.bm25_weight,
            },
            "summary": summarize_strategy(rows),
            "queries": rows,
        }

    return {
        "benchmark_name": "synthetic_retrieval_quality_benchmark",
        "mode": BENCHMARK_MODE,
        "vector_simulator": VECTOR_SIMULATOR,
        "candidate_pool": {
            "description": "Top-N chunks by deterministic simulated vector score; BM25 and hybrid strategies rerank this same candidate pool.",
            "candidate_pool_size": candidate_pool_size,
        },
        "dataset": {
            "documents_path": str(documents_path.relative_to(REPO_ROOT)),
            "queries_path": str(queries_path.relative_to(REPO_ROOT)),
            "document_count": len(documents_payload.get("documents", [])),
            "chunk_count": len(chunks),
            "query_count": len(queries),
            "query_categories": dict(Counter(query["category"] for query in queries)),
            "dataset_type": documents_payload.get("dataset_type", "unknown"),
        },
        "strategies": strategies_payload,
    }


def build_report(base_report: dict[str, Any], command: str) -> dict[str, Any]:
    return {
        **base_report,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "command": command,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "runner": "offline deterministic retrieval benchmark",
        },
        "metrics": ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"],
        "limitations": [
            "This benchmark uses synthetic legal/financial-style fixtures, not private or production documents.",
            "It runs fully offline and does not call Pinecone or external embedding services.",
            "Vector scores come from a deterministic semantic_terms bag-of-words cosine simulator.",
            "BM25 and hybrid strategies rerank the same simulated vector candidate pool.",
            "Results compare retrieval mechanics on controlled fixtures only.",
            "The benchmark does not measure production retrieval quality, legal correctness, financial correctness, latency, QPS, or customer data behavior.",
        ],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Synthetic Retrieval Quality Benchmark",
        "",
        f"Mode: `{report['mode']}`",
        f"Vector simulator: `{report['vector_simulator']}`",
        f"Timestamp UTC: `{report['timestamp_utc']}`",
        f"Git commit: `{report['git_commit']}`",
        f"Command: `{report['command']}`",
        f"Dataset: `{report['dataset']['documents_path']}` and `{report['dataset']['queries_path']}`",
        f"Chunks: `{report['dataset']['chunk_count']}`",
        f"Queries: `{report['dataset']['query_count']}`",
        f"Candidate pool size: `{report['candidate_pool']['candidate_pool_size']}`",
        "",
        "## Strategy Metrics",
        "",
        "| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | Candidate Pool Misses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for strategy_name, strategy_payload in report["strategies"].items():
        summary = strategy_payload["summary"]
        metrics = summary["overall"]
        lines.append(
            f"| {strategy_name} | {metrics['recall@1']:.4f} | {metrics['recall@3']:.4f} | "
            f"{metrics['recall@5']:.4f} | {metrics['mrr']:.4f} | {metrics['ndcg@5']:.4f} | "
            f"{summary['candidate_pool_miss_count']} |"
        )

    lines.extend([
        "",
        "## Category Metrics",
        "",
        "| Strategy | Category | Queries | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])

    for strategy_name, strategy_payload in report["strategies"].items():
        for category, category_payload in strategy_payload["summary"]["by_category"].items():
            metrics = category_payload["metrics"]
            lines.append(
                f"| {strategy_name} | {category} | {category_payload['query_count']} | "
                f"{metrics['recall@1']:.4f} | {metrics['recall@3']:.4f} | {metrics['recall@5']:.4f} | "
                f"{metrics['mrr']:.4f} | {metrics['ndcg@5']:.4f} |"
            )

    lines.extend([
        "",
        "## Missed Queries At Top 5",
        "",
    ])
    any_missed = False
    for strategy_name, strategy_payload in report["strategies"].items():
        missed = strategy_payload["summary"]["missed_queries_top5"]
        if not missed:
            continue
        any_missed = True
        lines.append(f"### {strategy_name}")
        lines.append("")
        for row in missed:
            lines.append(
                f"- `{row['query_id']}` ({row['category']}): relevant={row['relevant_chunk_ids']}, top5={row['top_chunk_ids']}"
            )
        lines.append("")
    if not any_missed:
        lines.append("No top-5 misses in this fixture run.")
        lines.append("")

    lines.extend([
        "## Limitations",
        "",
    ])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for strategy_name, strategy_payload in report["strategies"].items():
        for query in strategy_payload["queries"]:
            rows.append({
                "strategy": strategy_name,
                "query_id": query["query_id"],
                "category": query["category"],
                "recall@1": query["metrics"]["recall@1"],
                "recall@3": query["metrics"]["recall@3"],
                "recall@5": query["metrics"]["recall@5"],
                "mrr": query["metrics"]["mrr"],
                "ndcg@5": query["metrics"]["ndcg@5"],
                "top_chunk_ids": " ".join(item["chunk_id"] for item in query["top_results"]),
                "relevant_chunk_ids": " ".join(query["relevant_chunk_ids"]),
            })

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline synthetic retrieval benchmark.")
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_PATH)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--candidate-pool-size", type=int, default=25)
    parser.add_argument("--write-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_report = run_benchmark(
        args.documents,
        args.queries,
        candidate_pool_size=args.candidate_pool_size,
    )
    command = "python " + " ".join(sys.argv)
    report = build_report(base_report, command)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"retrieval_benchmark_{suffix}.json"
    markdown_path = args.output_dir / f"retrieval_benchmark_{suffix}.md"

    write_json(json_path, report)
    write_markdown(markdown_path, report)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")

    if args.write_csv:
        csv_path = args.output_dir / f"retrieval_benchmark_{suffix}.csv"
        write_csv(csv_path, report)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
