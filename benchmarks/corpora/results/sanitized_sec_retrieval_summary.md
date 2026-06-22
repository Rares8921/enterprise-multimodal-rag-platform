# Sanitized Document RAG Evaluation Summary

Source mode: `public`
Evaluation mode: `retrieve`
Git commit: `607225c`
Timestamp UTC: `2026-06-22T02:51:53.431430+00:00`
Corpus: `SEC EDGAR Public Filing Sample Rendered PDF` (`sec_edgar_public_sample_rendered_pdf`)
Documents: `8`
Queries: `16`

## Evidence Type

- Report mode: `retrieve`
- Corpus mode: `public`
- This is a sanitized local report summary, not production evidence.

## Retrieval Metrics

| Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---:|---:|---:|---:|---:|
| 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.8155 |

## Limitations

- This harness runs against local or explicitly configured services; it is not production evidence.
- Raw PDFs are expected to live in an ignored local directory and are not committed by default.
- Manifest labels may be incomplete until real indexing produces page or chunk labels.
- Ingestion and retrieval results do not prove legal correctness, financial correctness, QPS, uptime, or cost savings.
- Retrieval mode requires a live Pinecone index populated by the ingestion/OCR/layout/embedding pipeline.
- Document-level labels are weaker than page-level or chunk-level labels; metric interpretation depends on manifest label granularity.
- Retrieval metrics deduplicate repeated Pinecone chunks by the active label granularity before scoring.
- This is local real-service evidence only and must not be described as production retrieval quality.

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

- `source_report`: `document_rag_eval_retrieve_sec_retrieve.json`
- `private_summary_allowed`: `False`
- `query_text_allowed`: `False`
- `removed_content_keys`: `['answer', 'answer_text', 'chunk_text', 'context', 'contexts', 'document_text', 'raw_answer', 'raw_response', 'request', 'response', 'text', 'text_preview']`
- `path_redaction`: `absolute paths and local corpus roots redacted`
