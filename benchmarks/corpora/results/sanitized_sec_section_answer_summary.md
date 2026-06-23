# Sanitized Document RAG Evaluation Summary

Source mode: `public`
Evaluation mode: `answer`
Git commit: `9b00c5e`
Timestamp UTC: `2026-06-23T01:50:54.042760+00:00`
Corpus: `SEC EDGAR Public Filing Sample Rendered PDF - Section Retrieval Evaluation` (`sec_edgar_public_sample_rendered_pdf_section_eval`)
Documents: `8`
Queries: `29`

## Evidence Type

- Report mode: `answer`
- Corpus mode: `public`
- This is a sanitized local report summary, not production evidence.

## Answer Proxy

Failures: `16`
Non-empty answer rate: `0.448276`
Citation presence rate for required citations: `0.413793`
Average expected-hint overlap: `0.413793`
Estimated tokens used: `33363`
Average latency ms: `4109.796029`
Average retrieved context count: `5`
Model counts: `{"gemini": 13}`
Retrieval strategy counts: `{"pinecone_vector_candidates_plus_bm25_sec_aware_rerank": 13}`
Answer delay seconds: `30.0`
Max retries per failed query: `1`
Retry cooldown seconds: `45.0`
Failure categories: `{"HTTP 503: {\"detail\": \"LLM service unavailable\"}": 16}`

## Answer Resume

Source report: `document_rag_eval_answer_sec_section_answer_v4_rate_limited.json`
Retried queries: `18`

| Segment | Queries | Failures | Non-empty rate | Citation rate | Hint overlap | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| source | 29 | 18 | 0.37931 | 0.344828 | 0.344828 | 28098 |
| retry-only | 18 | 16 | 0.111111 | 0.111111 | 0.111111 | 5265 |
| combined | 29 | 16 | 0.448276 | 0.413793 | 0.413793 | 33363 |

## Limitations

- This harness runs against local or explicitly configured services; it is not production evidence.
- Raw PDFs are expected to live in an ignored local directory and are not committed by default.
- Manifest labels may be incomplete until real indexing produces page or chunk labels.
- Ingestion and retrieval results do not prove legal correctness, financial correctness, QPS, uptime, or cost savings.
- Answer mode may call real LLM providers through the query service and can incur cost depending on local configuration.
- Answer evaluation is a lightweight proxy: non-empty answer, citation presence, and expected-hint overlap only.
- Expected-hint overlap is not semantic correctness and must not be described as legal or financial accuracy.
- Use --answer-delay-seconds for live providers with request-per-minute limits; delayed runs trade runtime for fewer provider quota failures.
- Use --answer-retry-failed-from only to transparently retry failed prior rows; combined reports keep the full manifest denominator and preserve remaining failures.

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

- `source_report`: `document_rag_eval_answer_sec_section_answer_v5_combined.json`
- `private_summary_allowed`: `False`
- `query_text_allowed`: `False`
- `removed_content_keys`: `['answer', 'answer_text', 'chunk_text', 'context', 'contexts', 'document_text', 'raw_answer', 'raw_response', 'request', 'response', 'text', 'text_preview']`
- `path_redaction`: `absolute paths and local corpus roots redacted`
