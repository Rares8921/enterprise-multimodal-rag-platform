import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.corpus_manifest import validate_manifest_payload


DEFAULT_OUTPUT_PDF_DIR = REPO_ROOT / "benchmarks" / "corpora" / "local_pdfs" / "synthetic_smoke"
DEFAULT_MANIFEST_OUT = REPO_ROOT / "benchmarks" / "corpora" / "synthetic_smoke_manifest.json"


BASE_DOCUMENTS = [
    {
        "document_id": "synthetic_legal_msa",
        "filename": "synthetic_master_services_agreement.pdf",
        "doc_type": "legal_contract",
        "title": "Synthetic Master Services Agreement",
        "pages": [
            "Synthetic Master Services Agreement. The parties agree that services begin on January 1 and invoices are due within thirty days.",
            "Termination for convenience requires thirty days written notice. Confidential information must be protected for three years.",
        ],
        "query": "What notice is required for termination for convenience?",
        "category": "legal_clause",
        "relevant_pages": [2],
        "expected_answer_hints": ["thirty days", "written notice", "termination"],
    },
    {
        "document_id": "synthetic_legal_license",
        "filename": "synthetic_license_agreement.pdf",
        "doc_type": "legal_contract",
        "title": "Synthetic License Agreement",
        "pages": [
            "Synthetic License Agreement. The license is non-exclusive and limited to internal business use by the customer.",
            "The governing law is New York. Audit rights may be exercised once per calendar year with ten business days notice.",
        ],
        "query": "Which governing law applies to the license agreement?",
        "category": "legal_clause",
        "relevant_pages": [2],
        "expected_answer_hints": ["New York", "governing law"],
    },
    {
        "document_id": "synthetic_legal_supply",
        "filename": "synthetic_supply_agreement.pdf",
        "doc_type": "legal_contract",
        "title": "Synthetic Supply Agreement",
        "pages": [
            "Synthetic Supply Agreement. Supplier will deliver components under quarterly purchase orders and maintain insurance coverage.",
            "A most favored customer clause applies to standard component pricing. Late delivery credits begin after five business days.",
        ],
        "query": "What pricing protection appears in the supply agreement?",
        "category": "legal_clause",
        "relevant_pages": [2],
        "expected_answer_hints": ["most favored customer", "pricing"],
    },
    {
        "document_id": "synthetic_financial_annual",
        "filename": "synthetic_annual_report.pdf",
        "doc_type": "financial_report",
        "title": "Synthetic Annual Report",
        "pages": [
            "Synthetic Annual Report. Revenue increased from 100 million to 118 million because subscription sales expanded.",
            "Operating margin was 14 percent. Management highlighted disciplined hiring and stable infrastructure costs.",
        ],
        "query": "What revenue trend is described in the annual report?",
        "category": "numeric_financial",
        "relevant_pages": [1],
        "expected_answer_hints": ["100 million", "118 million", "revenue"],
    },
    {
        "document_id": "synthetic_financial_quarterly",
        "filename": "synthetic_quarterly_report.pdf",
        "doc_type": "financial_report",
        "title": "Synthetic Quarterly Report",
        "pages": [
            "Synthetic Quarterly Report. Cash and equivalents ended the quarter at 42 million. Debt remained unchanged.",
            "Risk factors include customer concentration and exposure to foreign currency movements in European markets.",
        ],
        "query": "What risk factors are described in the quarterly report?",
        "category": "financial_risk",
        "relevant_pages": [2],
        "expected_answer_hints": ["customer concentration", "foreign currency"],
    },
    {
        "document_id": "synthetic_financial_notes",
        "filename": "synthetic_financial_notes.pdf",
        "doc_type": "financial_report",
        "title": "Synthetic Financial Notes",
        "pages": [
            "Synthetic Financial Notes. Deferred revenue was 24 million and primarily related to annual software contracts.",
            "The company recognized stock based compensation expense of 6 million during the period.",
        ],
        "query": "What amount of deferred revenue is reported in the notes?",
        "category": "numeric_financial",
        "relevant_pages": [1],
        "expected_answer_hints": ["24 million", "deferred revenue"],
    },
    {
        "document_id": "synthetic_legal_data_processing",
        "filename": "synthetic_data_processing_addendum.pdf",
        "doc_type": "legal_contract",
        "title": "Synthetic Data Processing Addendum",
        "pages": [
            "Synthetic Data Processing Addendum. Processor must apply reasonable technical and organizational safeguards.",
            "Subprocessors require prior notice. Data deletion must occur within sixty days after service termination.",
        ],
        "query": "When must data deletion occur after service termination?",
        "category": "legal_clause",
        "relevant_pages": [2],
        "expected_answer_hints": ["sixty days", "data deletion"],
    },
    {
        "document_id": "synthetic_financial_cashflow",
        "filename": "synthetic_cashflow_summary.pdf",
        "doc_type": "financial_report",
        "title": "Synthetic Cash Flow Summary",
        "pages": [
            "Synthetic Cash Flow Summary. Operating cash flow was 31 million and capital expenditures were 9 million.",
            "Management attributed cash generation to collections discipline and lower inventory purchases.",
        ],
        "query": "What operating cash flow is reported?",
        "category": "numeric_financial",
        "relevant_pages": [1],
        "expected_answer_hints": ["31 million", "operating cash flow"],
    },
]


def generate_synthetic_pdf_corpus(
    *,
    output_pdf_dir: Path = DEFAULT_OUTPUT_PDF_DIR,
    manifest_out: Path = DEFAULT_MANIFEST_OUT,
    overwrite: bool = False,
    seed: int = 7,
    num_docs: int = 6,
) -> dict[str, Any]:
    if num_docs <= 0 or num_docs > 10:
        raise ValueError("--num-docs must be between 1 and 10")
    output_pdf_dir.mkdir(parents=True, exist_ok=True)
    docs = _select_documents(num_docs, seed)
    manifest_documents = []
    manifest_queries = []
    relative_prefix = output_pdf_dir.name

    for doc in docs:
        filename = doc["filename"]
        path = output_pdf_dir / filename
        if path.exists() and not overwrite:
            status = "exists"
        else:
            _write_simple_pdf(path, doc["title"], doc["pages"])
            status = "written"
        manifest_documents.append({
            "document_id": doc["document_id"],
            "filename": f"{relative_prefix}/{filename}",
            "doc_type": doc["doc_type"],
            "source_type": "synthetic_smoke_pdf",
            "source_note": "Deterministic public-safe synthetic PDF generated by benchmarks/generate_synthetic_pdf_corpus.py; not real legal or financial data.",
            "source_format": "pdf",
            "source_metadata": {"source_id": "synthetic_smoke", "source_format": "pdf", "generation_seed": seed},
            "page_count": len(doc["pages"]),
            "allowed_to_commit": False,
        })
        manifest_queries.append({
            "query_id": f"{doc['document_id']}_q1",
            "query": doc["query"],
            "category": doc["category"],
            "target_document_ids": [doc["document_id"]],
            "relevant_pages": doc["relevant_pages"],
            "relevant_chunk_ids": [],
            "expected_answer_hints": doc["expected_answer_hints"],
            "citation_required": True,
        })
        doc["write_status"] = status

    manifest = {
        "corpus_id": f"synthetic_smoke_pdf_seed_{seed}",
        "corpus_name": "Synthetic Smoke PDF Corpus",
        "mode": "synthetic",
        "source_metadata": {
            "source_id": "synthetic_smoke",
            "source_name": "Generated synthetic PDF smoke corpus",
            "license_or_usage_note": "Generated test data only; no private, customer, legal, or financial source documents.",
        },
        "documents": manifest_documents,
        "queries": manifest_queries,
    }
    validate_manifest_payload(manifest)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest_out": str(manifest_out),
        "output_pdf_dir": str(output_pdf_dir),
        "num_docs": len(docs),
        "seed": seed,
        "documents": [{"document_id": doc["document_id"], "filename": doc["filename"], "status": doc["write_status"]} for doc in docs],
        "limitations": [
            "Synthetic PDFs are public-safe smoke fixtures, not real legal or financial documents.",
            "Results from this corpus do not prove legal correctness, financial correctness, production retrieval quality, or provider accuracy.",
        ],
    }


def _select_documents(num_docs: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    docs = [dict(item) for item in BASE_DOCUMENTS]
    rng.shuffle(docs)
    return docs[:num_docs]


def _write_simple_pdf(path: Path, title: str, pages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    objects: list[bytes] = []
    page_ids = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    next_id = 4
    for page_number, text in enumerate(pages, start=1):
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_ids.append(page_id)
        content = _page_content(title, page_number, text)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii"))
        objects.append(f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"\nendstream")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    _write_pdf_objects(path, objects)


def _page_content(title: str, page_number: int, text: str) -> bytes:
    lines = [title, f"Page {page_number}", *(_wrap_text(text, 78))]
    commands = ["BT", "/F1 12 Tf", "72 740 Td", "14 TL"]
    for index, line in enumerate(lines):
        if index:
            commands.append("T*")
        commands.append(f"({_escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("ascii", errors="ignore")


def _write_pdf_objects(path: Path, objects: list[bytes]) -> None:
    chunks = [b"%PDF-1.4\n"]
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii"))
        chunks.append(obj)
        chunks.append(b"\nendobj\n")
    xref_offset = sum(len(chunk) for chunk in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    chunks.append(b"0000000000 65535 f \n")
    for offset in offsets:
        chunks.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    chunks.append(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    path.write_bytes(b"".join(chunks))


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(item) for item in current) + len(current) + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _escape_pdf_text(value: str) -> str:
    return re.sub(r"([\\()])", r"\\\1", value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deterministic synthetic PDF corpus for document RAG smoke testing.")
    parser.add_argument("--output-pdf-dir", type=Path, default=DEFAULT_OUTPUT_PDF_DIR)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-docs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate_synthetic_pdf_corpus(
        output_pdf_dir=args.output_pdf_dir,
        manifest_out=args.manifest_out,
        overwrite=args.overwrite,
        seed=args.seed,
        num_docs=args.num_docs,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
