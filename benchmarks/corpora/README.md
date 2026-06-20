# Curated PDF Corpora

This directory is for document RAG case-study corpora that may be public, synthetic, or private/local.

Raw PDFs are not committed by default. Place local PDFs under:

```text
benchmarks/corpora/local_pdfs/
```

Generated local reports are not committed by default. Write them under:

```text
benchmarks/corpora/results/
```

Commit only manifests, schemas, harness code, tests, and deliberately selected small reports. Do not commit private PDFs, arbitrary client documents, credentials, service exports, caches, or large generated artifacts.


## Public Source Registry

`public_sources.json` lists supported public corpus sources and their usage boundaries. It is metadata only; it does not mean any CUAD contracts or SEC filings have been downloaded, ingested, indexed, or evaluated.

Current source entries are:

- CUAD / Atticus Project for legal contract evaluation workflows.
- SEC EDGAR company filings for financial-report evaluation workflows.

Raw files acquired from these sources still belong under ignored local storage such as `benchmarks/corpora/local_pdfs/cuad/` or `benchmarks/corpora/local_pdfs/sec_edgar/`. Preserve attribution and source notes in generated manifests and reports.

## Manifest

Use `example_manifest.json` as the starting point. The manifest records:

- corpus identity and mode: `public`, `synthetic`, or `private_local`
- document IDs, filenames, document types, source notes, page counts, and commit permission flags
- labeled queries with target document IDs, optional relevant pages/chunks, expected answer hints, and citation requirements

The manifest can describe private/local PDFs without committing those PDFs. For private corpora, set:

```json
"mode": "private_local"
```

and keep `allowed_to_commit` set to `false` for every sensitive document.

## Expected Workflow

1. Copy PDFs into `benchmarks/corpora/local_pdfs/`.
2. Copy `example_manifest.json` to a new manifest file.
3. Update filenames and labels to match the local corpus.
4. Run manifest validation before any service calls.
5. Run ingestion, retrieval, and optional answer evaluation only against explicitly configured local services.

The harness and reports must be described as local/real-service evaluation, not production usage or production retrieval quality.

