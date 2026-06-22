import argparse
import json
import re
import sys
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.corpus_manifest import load_corpus_manifest, validate_manifest_payload
from benchmarks.generate_synthetic_pdf_corpus import _escape_pdf_text, _write_pdf_objects


DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "benchmarks" / "corpora" / "sec_edgar_manifest.generated.json"
DEFAULT_PDF_ROOT = REPO_ROOT / "benchmarks" / "corpora" / "local_pdfs"
DEFAULT_OUTPUT_PDF_DIR = DEFAULT_PDF_ROOT / "sec_edgar_rendered"
DEFAULT_RENDERED_MANIFEST = REPO_ROOT / "benchmarks" / "corpora" / "sec_edgar_rendered_manifest.generated.json"


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag.lower() in {"br", "p", "div", "tr", "table", "section", "article", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag.lower() in {"p", "div", "tr", "table", "section", "article", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        joined = unescape(" ".join(self._parts))
        joined = re.sub(r"[ \t\r\f\v]+", " ", joined)
        joined = re.sub(r"\n\s+", "\n", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()


def render_sec_html_manifest_to_pdf(
    *,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    pdf_root: Path = DEFAULT_PDF_ROOT,
    output_pdf_dir: Path = DEFAULT_OUTPUT_PDF_DIR,
    manifest_out: Path = DEFAULT_RENDERED_MANIFEST,
    overwrite: bool = False,
    lines_per_page: int = 50,
) -> dict[str, Any]:
    if lines_per_page < 10:
        raise ValueError("--lines-per-page must be at least 10")

    loaded = load_corpus_manifest(source_manifest, pdf_root=pdf_root, require_files=True)
    output_pdf_dir.mkdir(parents=True, exist_ok=True)
    relative_prefix = output_pdf_dir.relative_to(pdf_root).as_posix()

    rendered_documents = []
    actions = []
    for document in loaded.manifest.documents:
        if document.source_format != "html":
            raise ValueError(f"{document.document_id} has source_format={document.source_format!r}; only SEC HTML manifests are supported")
        html_path = loaded.document_paths[document.document_id]
        rendered_filename = f"{document.document_id}.pdf"
        rendered_path = output_pdf_dir / rendered_filename
        if rendered_path.exists() and not overwrite:
            page_count = _existing_page_count(rendered_path)
            status = "exists"
        else:
            text = _extract_visible_text(html_path)
            if not text:
                raise ValueError(f"No visible text extracted from {html_path}")
            pages = _paginate_text(text, lines_per_page=lines_per_page)
            _write_text_pdf(rendered_path, _document_title(document), pages)
            page_count = len(pages)
            status = "rendered"

        source_metadata = {
            **document.source_metadata,
            "original_source_format": document.source_metadata.get("source_format", document.source_format),
            "rendered_format": "pdf",
            "rendered_from_filename": document.filename,
            "rendering_method": "html_text_to_pdf",
            "rendering_limitations": "Generated from visible HTML text for ingestion; not a browser-faithful SEC filing layout rendering.",
        }
        source_metadata["source_format"] = "rendered_pdf"
        rendered_documents.append({
            "document_id": document.document_id,
            "filename": f"{relative_prefix}/{rendered_filename}",
            "doc_type": document.doc_type,
            "source_type": document.source_type,
            "source_note": (
                f"{document.source_note}; rendered_format=pdf; "
                "rendering_method=html_text_to_pdf; layout fidelity is not claimed"
            ),
            "source_format": "rendered_pdf",
            "source_metadata": source_metadata,
            "page_count": page_count,
            "allowed_to_commit": False,
        })
        actions.append({
            "document_id": document.document_id,
            "status": status,
            "source_path": str(html_path),
            "rendered_path": str(rendered_path),
            "page_count": page_count,
        })

    manifest = {
        "corpus_id": f"{loaded.manifest.corpus_id}_rendered_pdf",
        "corpus_name": f"{loaded.manifest.corpus_name} Rendered PDF",
        "mode": loaded.manifest.mode,
        "source_metadata": {
            **loaded.manifest.source_metadata,
            "source_format": "rendered_pdf",
            "original_source_format": "html",
            "rendering_method": "html_text_to_pdf",
            "rendering_limitations": "Rendered PDFs preserve extracted visible text for ingestion; they are not evidence of visual-layout fidelity.",
        },
        "documents": rendered_documents,
        "queries": [query_to_manifest_dict(query) for query in loaded.manifest.queries],
    }
    validate_manifest_payload(manifest)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "source_manifest": str(source_manifest),
        "manifest_out": str(manifest_out),
        "output_pdf_dir": str(output_pdf_dir),
        "document_count": len(rendered_documents),
        "query_count": len(manifest["queries"]),
        "actions": actions,
        "limitations": [
            "Rendered PDFs are generated from extracted visible HTML text for ingestion compatibility.",
            "The renderer does not claim browser-faithful SEC filing layout, legal correctness, financial correctness, or production retrieval quality.",
            "Labels remain document-level unless the source manifest provides page or chunk labels.",
        ],
    }


def query_to_manifest_dict(query: Any) -> dict[str, Any]:
    return {
        "query_id": query.query_id,
        "query": query.query,
        "category": query.category,
        "target_document_ids": list(query.target_document_ids),
        "relevant_pages": list(query.relevant_pages),
        "relevant_chunk_ids": list(query.relevant_chunk_ids),
        "expected_answer_hints": list(query.expected_answer_hints),
        "citation_required": query.citation_required,
    }


def _extract_visible_text(path: Path) -> str:
    parser = _VisibleTextParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    return parser.text()


def _paginate_text(text: str, *, lines_per_page: int) -> list[list[str]]:
    wrapped_lines: list[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            if wrapped_lines and wrapped_lines[-1]:
                wrapped_lines.append("")
            continue
        wrapped_lines.extend(_wrap_text(paragraph, 92))
    if not wrapped_lines:
        return [["No visible text extracted."]]
    return [wrapped_lines[index:index + lines_per_page] for index in range(0, len(wrapped_lines), lines_per_page)]


def _write_text_pdf(path: Path, title: str, pages: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    objects: list[bytes] = []
    page_ids = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    next_id = 4
    for page_number, lines in enumerate(pages, start=1):
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_ids.append(page_id)
        content = _page_content(title, page_number, len(pages), lines)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii"))
        objects.append(f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"\nendstream")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    _write_pdf_objects(path, objects)


def _page_content(title: str, page_number: int, total_pages: int, lines: list[str]) -> bytes:
    display_lines = [_safe_pdf_line(title), f"Page {page_number} of {total_pages}", "", *[_safe_pdf_line(line) for line in lines]]
    commands = ["BT", "/F1 10 Tf", "54 750 Td", "12 TL"]
    for index, line in enumerate(display_lines[:58]):
        if index:
            commands.append("T*")
        commands.append(f"({_escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("ascii", errors="ignore")


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(item) for item in current) + len(current) + len(word) > width:
            if current:
                lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _document_title(document: Any) -> str:
    metadata = document.source_metadata
    ticker = metadata.get("ticker", "SEC")
    form_type = metadata.get("form_type", "filing")
    filing_date = metadata.get("filing_date", "unknown date")
    accession = metadata.get("accession_number", document.document_id)
    return f"{ticker} {form_type} {filing_date} {accession}"


def _safe_pdf_line(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode("ascii")


def _existing_page_count(path: Path) -> int | None:
    data = path.read_bytes()
    matches = re.findall(rb"/Type\s*/Page\b", data)
    return len(matches) or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render downloaded SEC EDGAR HTML filings to local PDFs and create a rendered manifest.")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument("--output-pdf-dir", type=Path, default=DEFAULT_OUTPUT_PDF_DIR)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_RENDERED_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lines-per-page", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = render_sec_html_manifest_to_pdf(
        source_manifest=args.source_manifest,
        pdf_root=args.pdf_root,
        output_pdf_dir=args.output_pdf_dir,
        manifest_out=args.manifest_out,
        overwrite=args.overwrite,
        lines_per_page=args.lines_per_page,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
