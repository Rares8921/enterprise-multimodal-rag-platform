PYTHON ?= python
BENCHMARK_OUTPUT_DIR ?= benchmarks/results
BENCHMARK_RUN_ID ?= mock_latest

.PHONY: test test-llm-routing test-hybrid-retrieval test-benchmark benchmark-llm-routing benchmark-llm-routing-smoke

test: test-llm-routing test-hybrid-retrieval test-benchmark

test-llm-routing:
	$(PYTHON) -m pytest tests/unit/test_llm_routing.py -q

test-hybrid-retrieval:
	$(PYTHON) -m pytest tests/unit/test_hybrid_retrieval.py -q

test-benchmark:
	$(PYTHON) -m pytest tests/benchmark/test_llm_routing_benchmark.py -q

benchmark-llm-routing:
	$(PYTHON) benchmarks/llm_routing_benchmark.py --output-dir $(BENCHMARK_OUTPUT_DIR) --run-id $(BENCHMARK_RUN_ID)

benchmark-llm-routing-smoke:
	$(PYTHON) benchmarks/llm_routing_benchmark.py --output-dir $(BENCHMARK_OUTPUT_DIR) --run-id smoke
