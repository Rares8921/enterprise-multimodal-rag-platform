# Sanitized Document RAG Evaluation Summary

Source mode: `public`
Evaluation mode: `retrieve`
Git commit: `7c52a7d`
Timestamp UTC: `2026-06-22T12:09:59.805694+00:00`
Corpus: `SEC EDGAR Public Filing Sample Rendered PDF - Section Retrieval Evaluation` (`sec_edgar_public_sample_rendered_pdf_section_eval`)
Documents: `8`
Queries: `29`

## Evidence Type

- Report mode: `retrieve`
- Corpus mode: `public`
- This is a sanitized local report summary, not production evidence.

## Retrieval Metrics

Strategy: `pinecone_vector_candidates_plus_bm25_sec_aware_rerank`
Candidate pool size: `100`
Top K: `5`
Label granularity counts: `{"section": 29}`
Candidate pool misses: `6`

| Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---:|---:|---:|---:|---:|
| 0.7586 | 0.7931 | 0.7931 | 0.7759 | 0.7804 |

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

- `source_report`: `document_rag_eval_retrieve_sec_section_retrieve_v2_pool100.json`
- `private_summary_allowed`: `False`
- `query_text_allowed`: `False`
- `removed_content_keys`: `['answer', 'answer_text', 'chunk_text', 'context', 'contexts', 'document_text', 'raw_answer', 'raw_response', 'request', 'response', 'text', 'text_preview']`
- `path_redaction`: `absolute paths and local corpus roots redacted`


## Baseline Comparison

All rows use the same 8 public SEC 10-K filings, 29 section-level queries, and `sentence-transformers/all-MiniLM-L6-v2` query embeddings. Metrics are section-level and local/Pinecone-backed, not production retrieval quality.

| Run | Namespace | Candidate Pool | Reranking | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | Candidate Misses |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| Previous section baseline | `tenant_eval_local` | 25 | Vector candidates + BM25 | 0.1034 | 0.2759 | 0.3448 | 0.1879 | 0.2269 | 13 |
| V2 old namespace baseline | `tenant_eval_local` | 25 | Vector candidates + BM25 | 0.1034 | 0.2759 | 0.3448 | 0.1879 | 0.2269 | 13 |
| V2 enriched metadata only | `tenant_eval_sec_sections_v2` | 25 | Vector candidates + BM25 | 0.1034 | 0.2759 | 0.3448 | 0.1879 | 0.2269 | 13 |
| V2 SEC-aware rerank | `tenant_eval_sec_sections_v2` | 25 | SEC metadata-aware rerank | 0.5172 | 0.5517 | 0.5517 | 0.5287 | 0.5345 | 13 |
| V2 SEC-aware rerank | `tenant_eval_sec_sections_v2` | 50 | SEC metadata-aware rerank | 0.6897 | 0.7241 | 0.7241 | 0.7069 | 0.7114 | 8 |
| V2 SEC-aware rerank | `tenant_eval_sec_sections_v2` | 100 | SEC metadata-aware rerank | 0.7586 | 0.7931 | 0.7931 | 0.7759 | 0.7804 | 6 |

Per-query comparison against the previous section baseline: 13 queries improved at Recall@5, 0 regressed at Recall@5, and 6 remaining Recall@5 misses were still candidate-pool misses.
