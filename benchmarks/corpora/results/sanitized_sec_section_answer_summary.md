# Sanitized Document RAG Evaluation Summary

Source mode: `public`
Evaluation mode: `answer`
Git commit: `8e5462b`
Timestamp UTC: `2026-06-22T18:15:00.898474+00:00`
Corpus: `SEC EDGAR Public Filing Sample Rendered PDF - Section Retrieval Evaluation` (`sec_edgar_public_sample_rendered_pdf_section_eval`)
Documents: `8`
Queries: `29`

## Evidence Type

- Report mode: `answer`
- Corpus mode: `public`
- This is a sanitized local report summary, not production evidence.

## Answer Proxy

Failures: `18`
Non-empty answer rate: `0.37931`
Citation presence rate for required citations: `0.344828`
Average expected-hint overlap: `0.344828`
Estimated tokens used: `28098`
Average latency ms: `3024.91162`
Average retrieved context count: `5`
Model counts: `{"gemini": 11}`
Retrieval strategy counts: `{"pinecone_vector_candidates_plus_bm25_sec_aware_rerank": 11}`
Answer delay seconds: `15.0`
Failure categories: `{"HTTP 503: {\"detail\": \"LLM service circuit breaker open\"}": 5, "HTTP 503: {\"detail\": \"LLM service unavailable\"}": 11, "HTTP 504: {\"detail\": \"Request timeout\"}": 2}`

## Limitations

- This harness runs against local or explicitly configured services; it is not production evidence.
- Raw PDFs are expected to live in an ignored local directory and are not committed by default.
- Manifest labels may be incomplete until real indexing produces page or chunk labels.
- Ingestion and retrieval results do not prove legal correctness, financial correctness, QPS, uptime, or cost savings.
- Answer mode may call real LLM providers through the query service and can incur cost depending on local configuration.
- Answer evaluation is a lightweight proxy: non-empty answer, citation presence, and expected-hint overlap only.
- Expected-hint overlap is not semantic correctness and must not be described as legal or financial accuracy.
- Use --answer-delay-seconds for live providers with request-per-minute limits; delayed runs trade runtime for fewer provider quota failures.

## Unsupported Claims

- production usage
- customer data evaluation
- real users
- production legal or financial correctness
- production retrieval quality
- uptime, QPS, SLA, or incident recovery
- real cost savings
- provider accuracy

## Sanitization

- `source_report`: `document_rag_eval_answer_sec_section_answer_v4_rate_limited.json`
- `private_summary_allowed`: `False`
- `query_text_allowed`: `False`
- `removed_content_keys`: `['answer', 'answer_text', 'chunk_text', 'context', 'contexts', 'document_text', 'raw_answer', 'raw_response', 'request', 'response', 'text', 'text_preview']`
- `path_redaction`: `absolute paths and local corpus roots redacted`
