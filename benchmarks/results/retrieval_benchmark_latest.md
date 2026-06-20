# Synthetic Retrieval Quality Benchmark

Mode: `synthetic_offline`
Vector simulator: `semantic_terms_bow_cosine`
Timestamp UTC: `2026-06-20T00:33:18.487851+00:00`
Git commit: `61fedd9`
Command: `python benchmarks\retrieval_benchmark.py --output-dir benchmarks\results --run-id latest`
Dataset: `benchmarks\data_samples\retrieval_documents.json` and `benchmarks\data_samples\retrieval_queries.json`
Chunks: `40`
Queries: `15`
Candidate pool size: `25`

## Strategy Metrics

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | Candidate Pool Misses |
|---|---:|---:|---:|---:|---:|---:|
| vector_only | 0.9000 | 1.0000 | 1.0000 | 0.9667 | 0.9754 | 0 |
| bm25_only | 0.8667 | 0.9667 | 1.0000 | 0.9222 | 0.9468 | 0 |
| hybrid_70_30 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| hybrid_50_50 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| hybrid_30_70 | 0.9667 | 1.0000 | 1.0000 | 1.0000 | 0.9946 | 0 |

## Category Metrics

| Strategy | Category | Queries | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| vector_only | ambiguous_query | 1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vector_only | citation_oriented_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vector_only | distractor_heavy_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vector_only | exact_lexical_match | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vector_only | legal_clause_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vector_only | lexical_bm25_should_help | 2 | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.8155 |
| vector_only | numeric_financial_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vector_only | paraphrase_semantic_match | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vector_only | vector_semantic_should_help | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bm25_only | ambiguous_query | 1 | 0.0000 | 0.5000 | 1.0000 | 0.3333 | 0.5706 |
| bm25_only | citation_oriented_query | 2 | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.8155 |
| bm25_only | distractor_heavy_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bm25_only | exact_lexical_match | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bm25_only | legal_clause_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bm25_only | lexical_bm25_should_help | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bm25_only | numeric_financial_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bm25_only | paraphrase_semantic_match | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bm25_only | vector_semantic_should_help | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_70_30 | ambiguous_query | 1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_70_30 | citation_oriented_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_70_30 | distractor_heavy_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_70_30 | exact_lexical_match | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_70_30 | legal_clause_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_70_30 | lexical_bm25_should_help | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_70_30 | numeric_financial_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_70_30 | paraphrase_semantic_match | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_70_30 | vector_semantic_should_help | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | ambiguous_query | 1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | citation_oriented_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | distractor_heavy_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | exact_lexical_match | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | legal_clause_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | lexical_bm25_should_help | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | numeric_financial_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | paraphrase_semantic_match | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_50_50 | vector_semantic_should_help | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_30_70 | ambiguous_query | 1 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.9197 |
| hybrid_30_70 | citation_oriented_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_30_70 | distractor_heavy_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_30_70 | exact_lexical_match | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_30_70 | legal_clause_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_30_70 | lexical_bm25_should_help | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_30_70 | numeric_financial_query | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_30_70 | paraphrase_semantic_match | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_30_70 | vector_semantic_should_help | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Missed Queries At Top 5

No top-5 misses in this fixture run.

## Limitations

- This benchmark uses synthetic legal/financial-style fixtures, not private or production documents.
- It runs fully offline and does not call Pinecone or external embedding services.
- Vector scores come from a deterministic semantic_terms bag-of-words cosine simulator.
- BM25 and hybrid strategies rerank the same simulated vector candidate pool.
- Results compare retrieval mechanics on controlled fixtures only.
- The benchmark does not measure production retrieval quality, legal correctness, financial correctness, latency, QPS, or customer data behavior.
