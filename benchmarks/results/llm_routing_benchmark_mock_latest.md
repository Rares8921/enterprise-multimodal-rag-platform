# LLM Routing Mock Benchmark

Mode: `mock_synthetic`
Timestamp UTC: `2026-06-19T23:43:40.487429+00:00`
Git commit: `f0d1fd0`
Command: `python benchmarks\llm_routing_benchmark.py --output-dir benchmarks\results --run-id mock_latest`

## Summary

| Strategy | Queries | Cost USD | Cost/Query USD | p50 ms | p95 ms | p99 ms | Cache Hit Rate | Fallbacks | Keyword Overlap | Citation Presence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always_expensive | 13 | 0.40670000 | 0.03128462 | 1306.12 | 20326.14 | 26680.18 | 0.0769 | 0 | 1.0000 | 1.0000 |
| always_cheap | 13 | 0.08134000 | 0.00625692 | 783.96 | 13057.39 | 17168.82 | 0.0769 | 0 | 0.7756 | 0.4545 |
| heuristic | 13 | 0.35658280 | 0.02742945 | 783.96 | 20326.14 | 26680.18 | 0.0769 | 0 | 0.9141 | 0.8182 |

## Per-Query Routing

| Strategy | Query ID | Category | Primary | Selected | Reason | Input Tokens | Output Tokens | Cost USD | Latency ms | Cache |
|---|---|---|---|---|---|---:|---:|---:|---:|---|
| always_expensive | q001 | simple_factual | gemini | gemini | baseline_always_expensive | 506 | 64 | 0.00244300 | 610.35 | False |
| always_expensive | q002 | simple_factual | gemini | gemini | baseline_always_expensive | 650 | 96 | 0.00328300 | 655.23 | False |
| always_expensive | q003 | medium_document_qa | gemini | gemini | baseline_always_expensive | 1838 | 180 | 0.00832300 | 1306.12 | False |
| always_expensive | q004 | medium_document_qa | gemini | gemini | baseline_always_expensive | 2137 | 220 | 0.00978950 | 1421.38 | False |
| always_expensive | q005 | complex_legal_financial | gemini | gemini | baseline_always_expensive | 3847 | 360 | 0.01724450 | 2562.98 | False |
| always_expensive | q006 | complex_legal_financial | gemini | gemini | baseline_always_expensive | 4445 | 420 | 0.01996750 | 2842.63 | False |
| always_expensive | q007 | citation_heavy | gemini | gemini | baseline_always_expensive | 2844 | 300 | 0.01310400 | 2026.75 | False |
| always_expensive | q008 | citation_heavy | gemini | gemini | baseline_always_expensive | 3342 | 340 | 0.01526700 | 2246.26 | False |
| always_expensive | q009 | long_context | gemini | gemini | baseline_always_expensive | 36246 | 500 | 0.13211100 | 20326.14 | False |
| always_expensive | q010 | long_context | gemini | gemini | baseline_always_expensive | 48243 | 560 | 0.17473050 | 26680.18 | False |
| always_expensive | q011 | adversarial_ambiguous | gemini | gemini | baseline_always_expensive | 1143 | 160 | 0.00568050 | 1116.17 | False |
| always_expensive | q012 | adversarial_ambiguous | gemini | gemini | baseline_always_expensive | 939 | 140 | 0.00475650 | 1036.20 | False |
| always_expensive | q013 | simple_factual | gemini | gemini | baseline_always_expensive | 506 | 64 | 0.00000000 | 5.00 | True |
| always_cheap | q001 | simple_factual | mistral | mistral | baseline_always_cheap | 506 | 64 | 0.00048860 | 349.05 | False |
| always_cheap | q002 | simple_factual | mistral | mistral | baseline_always_cheap | 650 | 96 | 0.00065660 | 378.09 | False |
| always_cheap | q003 | medium_document_qa | mistral | mistral | baseline_always_cheap | 1838 | 180 | 0.00166460 | 783.96 | False |
| always_cheap | q004 | medium_document_qa | mistral | mistral | baseline_always_cheap | 2137 | 220 | 0.00195790 | 858.54 | False |
| always_cheap | q005 | complex_legal_financial | mistral | mistral | baseline_always_cheap | 3847 | 360 | 0.00344890 | 1581.92 | False |
| always_cheap | q006 | complex_legal_financial | mistral | mistral | baseline_always_cheap | 4445 | 420 | 0.00399350 | 1762.88 | False |
| always_cheap | q007 | citation_heavy | mistral | mistral | baseline_always_cheap | 2844 | 300 | 0.00262080 | 1238.02 | False |
| always_cheap | q008 | citation_heavy | mistral | mistral | baseline_always_cheap | 3342 | 340 | 0.00305340 | 1380.05 | False |
| always_cheap | q009 | long_context | mistral | mistral | baseline_always_cheap | 36246 | 500 | 0.02642220 | 13057.39 | False |
| always_cheap | q010 | long_context | mistral | mistral | baseline_always_cheap | 48243 | 560 | 0.03494610 | 17168.82 | False |
| always_cheap | q011 | adversarial_ambiguous | mistral | mistral | baseline_always_cheap | 1143 | 160 | 0.00113610 | 657.99 | False |
| always_cheap | q012 | adversarial_ambiguous | mistral | mistral | baseline_always_cheap | 939 | 140 | 0.00095130 | 606.25 | False |
| always_cheap | q013 | simple_factual | mistral | mistral | baseline_always_cheap | 506 | 64 | 0.00000000 | 5.00 | True |
| heuristic | q001 | simple_factual | mistral | mistral | cost_efficient | 506 | 64 | 0.00048860 | 349.05 | False |
| heuristic | q002 | simple_factual | mistral | mistral | cost_efficient | 650 | 96 | 0.00065660 | 378.09 | False |
| heuristic | q003 | medium_document_qa | mistral | mistral | cost_efficient | 1838 | 180 | 0.00166460 | 783.96 | False |
| heuristic | q004 | medium_document_qa | mistral | mistral | cost_efficient | 2137 | 220 | 0.00195790 | 858.54 | False |
| heuristic | q005 | complex_legal_financial | gemini | gemini | high_complexity | 3847 | 360 | 0.01724450 | 2562.98 | False |
| heuristic | q006 | complex_legal_financial | gemini | gemini | high_complexity | 4445 | 420 | 0.01996750 | 2842.63 | False |
| heuristic | q007 | citation_heavy | mistral | mistral | cost_efficient | 2844 | 300 | 0.00262080 | 1238.02 | False |
| heuristic | q008 | citation_heavy | mistral | mistral | cost_efficient | 3342 | 340 | 0.00305340 | 1380.05 | False |
| heuristic | q009 | long_context | gemini | gemini | context_size | 36246 | 500 | 0.13211100 | 20326.14 | False |
| heuristic | q010 | long_context | gemini | gemini | context_size | 48243 | 560 | 0.17473050 | 26680.18 | False |
| heuristic | q011 | adversarial_ambiguous | mistral | mistral | cost_efficient | 1143 | 160 | 0.00113610 | 657.99 | False |
| heuristic | q012 | adversarial_ambiguous | mistral | mistral | cost_efficient | 939 | 140 | 0.00095130 | 606.25 | False |
| heuristic | q013 | simple_factual | mistral | mistral | cost_efficient | 506 | 64 | 0.00000000 | 5.00 | True |

## Limitations

- This benchmark is mock/synthetic and does not call real LLM providers.
- Latency values are deterministic estimates, not measured provider latency.
- The quality proxy checks non-empty answers, citation-marker presence, and expected keyword overlap only.
- The quality proxy is not a semantic evaluation and must not be presented as model accuracy.
- Costs are estimates from the static cost table in the router benchmark, not billing records.
- The workload is fixed and small; it is intended for reproducibility and smoke comparison, not broad evaluation.
