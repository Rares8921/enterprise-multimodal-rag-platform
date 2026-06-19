# LLM Routing Benchmark

## Goal

This benchmark evaluates the repository's LLM routing heuristic against two fixed baselines:

- `always_expensive`: route every request to Gemini.
- `always_cheap`: route every request to Mistral.
- `heuristic`: use `services/llm-orchestrator/utils/ModelRouter.py`.

The current benchmark is `mock_synthetic`. It does not call real LLM providers and must not be presented as production latency, production cost, model accuracy, or realized savings.

## Reproduction

Run:

```powershell
python benchmarks\llm_routing_benchmark.py --output-dir benchmarks\results --run-id mock_latest
```

Cross-platform equivalent:

```bash
python benchmarks/llm_routing_benchmark.py --output-dir benchmarks/results --run-id mock_latest
```

Checked-in report files:

- `benchmarks/results/llm_routing_benchmark_mock_latest.json`
- `benchmarks/results/llm_routing_benchmark_mock_latest.md`

The runner also writes a CSV file, but `*.csv` is ignored by the repository's current `.gitignore`.

## Workload

The fixed workload is `benchmarks/data_samples/llm_routing_workload.json`.

It contains 13 queries:

- 3 simple factual queries, including one duplicate cache-probe query.
- 2 medium document QA queries.
- 2 complex legal or financial reasoning queries.
- 2 citation-heavy queries.
- 2 long-context queries.
- 2 adversarial or ambiguous queries.

Each item records query text, document type, estimated context tokens, expected output tokens, expected keywords, and whether citations are expected.

## Cost Model

Costs are static estimates in USD per 1M tokens:

| Model | Input | Output |
|---|---:|---:|
| Gemini | 3.50 | 10.50 |
| Mistral | 0.70 | 2.10 |

These numbers are benchmark assumptions, not billing records.

## Latency

Latency values are deterministic synthetic estimates from `benchmarks/llm_routing_benchmark.py`. They use model, token count, query category, and cache-hit status. They are useful for reproducible comparison of benchmark mechanics, not for claiming real provider latency.

## Current Results

Report commit: `f0d1fd0`

| Strategy | Estimated Total Cost | p50 ms | p95 ms | p99 ms | Cache Hit Rate | Fallbacks | Keyword Overlap | Citation Presence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_expensive | 0.40670000 | 1306.12 | 20326.14 | 26680.18 | 0.0769 | 0 | 1.0000 | 1.0000 |
| always_cheap | 0.08134000 | 783.96 | 13057.39 | 17168.82 | 0.0769 | 0 | 0.7756 | 0.4545 |
| heuristic | 0.35658280 | 783.96 | 20326.14 | 26680.18 | 0.0769 | 0 | 0.9141 | 0.8182 |

Interpretation: in this mock workload, the heuristic routes simple and medium requests mostly to Mistral and complex or long-context requests to Gemini. The lower cost versus `always_expensive` is an estimate from the static cost table. The quality proxy is only a harness check.

## Quality Proxy

The benchmark records:

- answer is non-empty
- citation marker is present when the workload expects citations
- overlap with manually listed expected keywords

This is not semantic evaluation. It cannot support claims about answer correctness, legal correctness, financial correctness, or real model accuracy.

## What Can Be Claimed

- The repository has a reproducible mock benchmark for LLM routing strategies.
- The benchmark records selected model, routing reason, estimated tokens, estimated cost, latency estimate, cache hit rate, fallback count, and quality proxy fields per query.
- The benchmark compares two fixed baselines with the heuristic router.

## What Cannot Be Claimed Yet

- Real provider latency.
- Real provider cost or billing reduction.
- Production cost savings.
- Production failure rate or uptime.
- Semantic answer quality.
- Legal or financial accuracy.
- Real user traffic or QPS.
