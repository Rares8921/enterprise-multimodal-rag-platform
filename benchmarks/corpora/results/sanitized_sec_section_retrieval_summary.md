# Sanitized Document RAG Evaluation Summary

Source mode: `public`
Evaluation mode: `retrieve`
Git commit: `a6bdc08`
Timestamp UTC: `2026-06-22T11:00:27.965207+00:00`
Corpus: `SEC EDGAR Public Filing Sample Rendered PDF - Section Retrieval Evaluation` (`sec_edgar_public_sample_rendered_pdf_section_eval`)
Documents: `8`
Queries: `29`

## Evidence Type

- Report mode: `retrieve`
- Corpus mode: `public`
- This is a sanitized local report summary, not production evidence.

## Retrieval Metrics

Strategy: `pinecone_vector_candidates_plus_bm25_hybrid_rerank`
Candidate pool size: `25`
Top K: `5`
Label granularity counts: `{"section": 29}`
Candidate pool misses: `13`

| Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---:|---:|---:|---:|---:|
| 0.1034 | 0.2759 | 0.3448 | 0.1879 | 0.2269 |

## Limitations

- This harness runs against local or explicitly configured services; it is not production evidence.
- Raw PDFs are expected to live in an ignored local directory and are not committed by default.
- Manifest labels may be incomplete until real indexing produces page or chunk labels.
- Ingestion and retrieval results do not prove legal correctness, financial correctness, QPS, uptime, or cost savings.
- Retrieval mode requires a live Pinecone index populated by the ingestion/OCR/layout/embedding pipeline.
- Document-level labels are weaker than section-level, page-level, or chunk-level labels; metric interpretation depends on manifest label granularity.
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

- `source_report`: `document_rag_eval_retrieve_sec_section_retrieve.json`
- `private_summary_allowed`: `False`
- `query_text_allowed`: `False`
- `removed_content_keys`: `['answer', 'answer_text', 'chunk_text', 'context', 'contexts', 'document_text', 'raw_answer', 'raw_response', 'request', 'response', 'text', 'text_preview']`
- `path_redaction`: `absolute paths and local corpus roots redacted`
