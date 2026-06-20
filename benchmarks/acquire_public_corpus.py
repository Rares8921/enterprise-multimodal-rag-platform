import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.corpus_sources.cuad import DEFAULT_CUAD_LOCAL_DIR, DEFAULT_CUAD_MANIFEST, prepare_cuad_corpus
from benchmarks.corpus_sources.sec_edgar import DEFAULT_SEC_COMPANY_LIST, DEFAULT_SEC_LOCAL_DIR, DEFAULT_SEC_MANIFEST, prepare_sec_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire or prepare public corpora for document RAG evaluation.")
    subparsers = parser.add_subparsers(dest="source", required=True)

    cuad = subparsers.add_parser("cuad", help="Prepare a small CUAD / Atticus Project legal corpus manifest.")
    cuad.add_argument("--metadata-json", type=Path, required=True, help="Local CUAD-style metadata JSON. This file is not committed by default.")
    cuad.add_argument("--output-pdf-dir", type=Path, default=DEFAULT_CUAD_LOCAL_DIR)
    cuad.add_argument("--manifest-out", type=Path, default=DEFAULT_CUAD_MANIFEST)
    cuad.add_argument("--sample-size", type=int, default=10)
    cuad.add_argument("--download", action="store_true", help="Download documents only when metadata rows include explicit source_url values.")
    cuad.add_argument("--copy-local-files", action="store_true", help="Copy local_source_path PDFs from metadata into ignored corpus storage.")
    cuad.add_argument("--overwrite", action="store_true")
    cuad.add_argument("--timeout-seconds", type=float, default=60.0)

    sec = subparsers.add_parser("sec-edgar", help="Prepare a small SEC EDGAR financial filing corpus manifest.")
    sec.add_argument("--filings-json", type=Path, default=None, help="Local filing metadata JSON for no-network mode.")
    sec.add_argument("--company-list", type=Path, default=DEFAULT_SEC_COMPANY_LIST)
    sec.add_argument("--ticker", action="append", dest="tickers", help="Ticker to include when fetching metadata; can be repeated.")
    sec.add_argument("--form-type", default="10-K")
    sec.add_argument("--filings-per-company", type=int, default=1)
    sec.add_argument("--sample-size", type=int, default=6)
    sec.add_argument("--output-file-dir", type=Path, default=DEFAULT_SEC_LOCAL_DIR)
    sec.add_argument("--manifest-out", type=Path, default=DEFAULT_SEC_MANIFEST)
    sec.add_argument("--sec-user-agent", default=None, help="Required for SEC network requests; may also be set via SEC_USER_AGENT.")
    sec.add_argument("--fetch-metadata", action="store_true", help="Fetch recent filing metadata from SEC submissions APIs.")
    sec.add_argument("--download", action="store_true", help="Download primary filing documents into ignored local storage.")
    sec.add_argument("--request-delay-seconds", type=float, default=0.25)
    sec.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "cuad":
        report = prepare_cuad_corpus(
            metadata_json=args.metadata_json,
            output_pdf_dir=args.output_pdf_dir,
            manifest_out=args.manifest_out,
            sample_size=args.sample_size,
            download=args.download,
            copy_local_files=args.copy_local_files,
            overwrite=args.overwrite,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.source == "sec-edgar":
        report = prepare_sec_corpus(
            filings_json=args.filings_json,
            company_list=args.company_list,
            tickers=args.tickers,
            form_type=args.form_type,
            filings_per_company=args.filings_per_company,
            sample_size=args.sample_size,
            output_file_dir=args.output_file_dir,
            manifest_out=args.manifest_out,
            user_agent=args.sec_user_agent,
            fetch_metadata=args.fetch_metadata,
            download=args.download,
            request_delay_seconds=args.request_delay_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        raise ValueError(f"Unsupported source: {args.source}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
