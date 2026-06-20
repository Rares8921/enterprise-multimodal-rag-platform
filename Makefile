PYTHON ?= python
BENCHMARK_OUTPUT_DIR ?= benchmarks/results
BENCHMARK_RUN_ID ?= mock_latest
CORPUS_MANIFEST ?= benchmarks/corpora/synthetic_smoke_manifest.json
CORPUS_PDF_ROOT ?= benchmarks/corpora/local_pdfs
CORPUS_OUTPUT_DIR ?= benchmarks/corpora/results
CORPUS_RUN_ID ?= local
CORPUS_PREFLIGHT_TARGET ?= validate-only
CORPUS_SAMPLE_SIZE ?= 6
CUAD_METADATA ?= benchmarks/corpora/local_pdfs/cuad_metadata.json
SEC_FILINGS_JSON ?= benchmarks/corpora/local_pdfs/sec_filings.json
PINECONE_NAMESPACE ?= $(DOCUMENT_RAG_EVAL_TENANT_ID)
REPORT_JSON ?= benchmarks/corpora/results/document_rag_eval_validate-only_local.json
PROMOTED_REPORT_MD ?= benchmarks/corpora/results/sanitized_document_rag_summary.md

.PHONY: test test-llm-routing test-hybrid-retrieval test-benchmark test-corpus-workflow benchmark-llm-routing benchmark-llm-routing-smoke corpus-validate corpus-preflight corpus-generate-synthetic corpus-acquire-cuad corpus-acquire-sec corpus-ingest corpus-retrieve corpus-answer corpus-promote-report

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

test-corpus-workflow:
	$(PYTHON) -m pytest tests/benchmark/test_public_corpus_workflow.py -q

corpus-validate:
	$(PYTHON) benchmarks/e2e_document_rag_eval.py validate-only --manifest $(CORPUS_MANIFEST) --pdf-root $(CORPUS_PDF_ROOT) --output-dir $(CORPUS_OUTPUT_DIR) --run-id $(CORPUS_RUN_ID)

corpus-preflight:
	$(PYTHON) benchmarks/e2e_document_rag_eval.py preflight --preflight-target $(CORPUS_PREFLIGHT_TARGET) --manifest $(CORPUS_MANIFEST) --pdf-root $(CORPUS_PDF_ROOT) --output-dir $(CORPUS_OUTPUT_DIR) --run-id $(CORPUS_RUN_ID)

corpus-generate-synthetic:
	$(PYTHON) benchmarks/generate_synthetic_pdf_corpus.py --output-pdf-dir $(CORPUS_PDF_ROOT)/synthetic_smoke --manifest-out benchmarks/corpora/synthetic_smoke_manifest.json --overwrite --seed 7 --num-docs $(CORPUS_SAMPLE_SIZE)

corpus-acquire-cuad:
	$(PYTHON) benchmarks/acquire_public_corpus.py cuad --metadata-json $(CUAD_METADATA) --output-pdf-dir $(CORPUS_PDF_ROOT)/cuad --manifest-out benchmarks/corpora/cuad_manifest.generated.json --sample-size $(CORPUS_SAMPLE_SIZE)

corpus-acquire-sec:
	$(PYTHON) benchmarks/acquire_public_corpus.py sec-edgar --filings-json $(SEC_FILINGS_JSON) --output-file-dir $(CORPUS_PDF_ROOT)/sec_edgar --manifest-out benchmarks/corpora/sec_edgar_manifest.generated.json --sample-size $(CORPUS_SAMPLE_SIZE)

corpus-ingest:
	$(PYTHON) benchmarks/e2e_document_rag_eval.py ingest --manifest $(CORPUS_MANIFEST) --pdf-root $(CORPUS_PDF_ROOT) --output-dir $(CORPUS_OUTPUT_DIR) --run-id $(CORPUS_RUN_ID) --poll-status

corpus-retrieve:
	$(PYTHON) benchmarks/e2e_document_rag_eval.py retrieve --manifest $(CORPUS_MANIFEST) --pdf-root $(CORPUS_PDF_ROOT) --output-dir $(CORPUS_OUTPUT_DIR) --run-id $(CORPUS_RUN_ID) --pinecone-namespace $(PINECONE_NAMESPACE)

corpus-answer:
	$(PYTHON) benchmarks/e2e_document_rag_eval.py answer --manifest $(CORPUS_MANIFEST) --pdf-root $(CORPUS_PDF_ROOT) --output-dir $(CORPUS_OUTPUT_DIR) --run-id $(CORPUS_RUN_ID)

corpus-promote-report:
	$(PYTHON) benchmarks/promote_document_rag_report.py $(REPORT_JSON) --output-md $(PROMOTED_REPORT_MD)
