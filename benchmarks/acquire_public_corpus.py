import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.corpus_sources.cuad import DEFAULT_CUAD_LOCAL_DIR, DEFAULT_CUAD_MANIFEST, prepare_cuad_corpus


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
    else:
        raise ValueError(f"Unsupported source: {args.source}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
