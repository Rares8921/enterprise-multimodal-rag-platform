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
- document IDs, filenames, document types, source notes, optional source metadata, source format, page counts, and commit permission flags
- labeled queries with target document IDs, optional relevant pages/chunks, expected answer hints, and citation requirements

The manifest can describe private/local PDFs or public HTML/PDF source files without committing those raw files. For private corpora, set:

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


## Acquisition And Smoke Workflows

Generate synthetic public-safe PDFs for smoke testing:

```powershell
python benchmarks\generate_synthetic_pdf_corpus.py --output-pdf-dir benchmarks\corpora\local_pdfs\synthetic_smoke --manifest-out benchmarks\corpora\synthetic_smoke_manifest.json --overwrite --seed 7 --num-docs 6
```

Prepare a CUAD manifest from local metadata. This does not download files unless `--download` is passed and metadata rows include explicit PDF URLs:

```powershell
python benchmarks\acquire_public_corpus.py cuad --metadata-json benchmarks\corpora\local_pdfs\cuad_metadata.json --output-pdf-dir benchmarks\corpora\local_pdfs\cuad --manifest-out benchmarks\corpora\cuad_manifest.generated.json --sample-size 10
```

Prepare a SEC EDGAR manifest from local filing metadata without network access:

```powershell
python benchmarks\acquire_public_corpus.py sec-edgar --filings-json benchmarks\corpora\local_pdfs\sec_filings.json --output-file-dir benchmarks\corpora\local_pdfs\sec_edgar --manifest-out benchmarks\corpora\sec_edgar_manifest.generated.json --sample-size 6
```

Fetch SEC metadata only with a real contact User-Agent and conservative request pacing:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
python benchmarks\acquire_public_corpus.py sec-edgar --fetch-metadata --ticker AAPL --form-type 10-K --sample-size 1 --manifest-out benchmarks\corpora\sec_edgar_manifest.generated.json
```

Run preflight before service calls:

```powershell
python benchmarks\e2e_document_rag_eval.py preflight --preflight-target ingest --manifest benchmarks\corpora\synthetic_smoke_manifest.json --pdf-root benchmarks\corpora\local_pdfs
```

Validate a generated manifest:

```powershell
python benchmarks\e2e_document_rag_eval.py validate-only --manifest benchmarks\corpora\synthetic_smoke_manifest.json --pdf-root benchmarks\corpora\local_pdfs
```

Promote only reviewed local reports:

```powershell
python benchmarks\promote_document_rag_report.py benchmarks\corpora\results\document_rag_eval_retrieve_local.json --output-md benchmarks\corpora\results\sanitized_document_rag_summary.md
```

## Claim Boundaries

After acquisition code exists, you may claim only that the repository can prepare manifest-compatible CUAD/SEC corpora. After synthetic generation, you may claim public-safe PDF smoke fixtures. After a real public-corpus run, claim only the exact metrics in the generated, reviewed report.

Do not claim production retrieval quality, legal correctness, financial correctness, customer data evaluation, provider accuracy, uptime, QPS, SLA, or cost savings from this workflow.
