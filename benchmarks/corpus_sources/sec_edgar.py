import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from benchmarks.corpus_manifest import validate_manifest_payload
from benchmarks.public_corpus_sources import PublicCorpusSource, load_public_source_registry


DEFAULT_SEC_COMPANY_LIST = Path("benchmarks/corpora/sec_edgar_sample_companies.json")
DEFAULT_SEC_LOCAL_DIR = Path("benchmarks/corpora/local_pdfs/sec_edgar")
DEFAULT_SEC_MANIFEST = Path("benchmarks/corpora/sec_edgar_manifest.generated.json")
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_REQUEST_DELAY_SECONDS = 0.25


@dataclass(frozen=True)
class SecCompany:
    ticker: str
    company_name: str
    cik: str


@dataclass(frozen=True)
class SecFilingSpec:
    ticker: str
    company_name: str
    cik: str
    accession_number: str
    form_type: str
    filing_date: str
    primary_document: str
    source_url: str
    source_format: str


def load_sec_companies(path: Path = DEFAULT_SEC_COMPANY_LIST) -> list[SecCompany]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_companies = payload.get("companies") if isinstance(payload, dict) else payload
    if not isinstance(raw_companies, list) or not raw_companies:
        raise ValueError("SEC company list must be a non-empty companies list")
    return [_parse_company(raw, index) for index, raw in enumerate(raw_companies)]


def load_sec_filing_specs(path: Path) -> list[SecFilingSpec]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_filings = payload.get("filings") if isinstance(payload, dict) else payload
    if not isinstance(raw_filings, list) or not raw_filings:
        raise ValueError("SEC filings metadata must be a non-empty filings list")
    return [_parse_filing(raw, index) for index, raw in enumerate(raw_filings)]


def fetch_recent_sec_filings(
    *,
    companies: list[SecCompany],
    form_type: str,
    filings_per_company: int,
    user_agent: str,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    timeout_seconds: float = 60.0,
) -> list[SecFilingSpec]:
    _require_sec_user_agent(user_agent)
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
    specs: list[SecFilingSpec] = []
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True, headers=headers) as client:
        for company in companies:
            response = client.get(SEC_SUBMISSIONS_URL.format(cik=company.cik.zfill(10)))
            response.raise_for_status()
            specs.extend(_recent_filings_from_submission(company, response.json(), form_type, filings_per_company))
            time.sleep(max(0.0, request_delay_seconds))
    return specs


def build_sec_manifest(
    specs: list[SecFilingSpec],
    *,
    source: PublicCorpusSource | None = None,
    corpus_id: str = "sec_edgar_public_sample",
    corpus_name: str = "SEC EDGAR Public Filing Sample",
) -> dict[str, Any]:
    source = source or load_public_source_registry().by_id("sec_edgar")
    documents = []
    queries = []
    for spec in specs:
        document_id = _document_id(spec)
        filename = f"sec_edgar/{_local_filename(spec)}"
        documents.append({
            "document_id": document_id,
            "filename": filename,
            "doc_type": "financial_report",
            "source_type": "public_sec_edgar",
            "source_note": f"SEC EDGAR {spec.form_type} filing for {spec.ticker}; accession={spec.accession_number}; filing_date={spec.filing_date}; source_format={spec.source_format}",
            "source_metadata": {
                **source.to_manifest_source_metadata(),
                "ticker": spec.ticker,
                "company_name": spec.company_name,
                "cik": spec.cik,
                "accession_number": spec.accession_number,
                "form_type": spec.form_type,
                "filing_date": spec.filing_date,
                "primary_document": spec.primary_document,
                "source_url": spec.source_url,
                "source_format": spec.source_format,
            },
            "page_count": None,
            "allowed_to_commit": False,
        })
        queries.extend(_query_templates(document_id, spec))
    return {
        "corpus_id": corpus_id,
        "corpus_name": corpus_name,
        "mode": "public",
        "source_metadata": source.to_manifest_source_metadata(),
        "documents": documents,
        "queries": queries,
    }


def prepare_sec_corpus(
    *,
    filings_json: Path | None = None,
    company_list: Path = DEFAULT_SEC_COMPANY_LIST,
    tickers: list[str] | None = None,
    form_type: str = "10-K",
    filings_per_company: int = 1,
    sample_size: int = 6,
    output_file_dir: Path = DEFAULT_SEC_LOCAL_DIR,
    manifest_out: Path = DEFAULT_SEC_MANIFEST,
    user_agent: str | None = None,
    fetch_metadata: bool = False,
    download: bool = False,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    registry = load_public_source_registry()
    source = registry.by_id("sec_edgar")
    user_agent = user_agent or os.getenv("SEC_USER_AGENT")

    if filings_json:
        specs = load_sec_filing_specs(filings_json)
    else:
        if not fetch_metadata:
            raise ValueError("SEC acquisition requires --filings-json for no-network mode or --fetch-metadata for SEC requests")
        _require_sec_user_agent(user_agent)
        companies = _select_companies(load_sec_companies(company_list), tickers, sample_size)
        specs = fetch_recent_sec_filings(
            companies=companies,
            form_type=form_type,
            filings_per_company=filings_per_company,
            user_agent=user_agent or "",
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
        )

    specs = specs[:sample_size]
    output_file_dir.mkdir(parents=True, exist_ok=True)
    actions = []
    if download:
        _require_sec_user_agent(user_agent)
        headers = {"User-Agent": user_agent or "", "Accept-Encoding": "gzip, deflate"}
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True, headers=headers) as client:
            for spec in specs:
                target_path = output_file_dir / _local_filename(spec)
                response = client.get(spec.source_url)
                response.raise_for_status()
                target_path.write_bytes(response.content)
                actions.append({"document_id": _document_id(spec), "status": "downloaded", "target_path": str(target_path), "source_format": spec.source_format})
                time.sleep(max(0.0, request_delay_seconds))
    else:
        actions = [
            {"document_id": _document_id(spec), "status": "manifest_only", "target_path": str(output_file_dir / _local_filename(spec)), "source_format": spec.source_format}
            for spec in specs
        ]

    manifest = build_sec_manifest(specs, source=source)
    validate_manifest_payload(manifest)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "source_id": source.source_id,
        "manifest_out": str(manifest_out),
        "output_file_dir": str(output_file_dir),
        "sample_size": sample_size,
        "document_count": len(specs),
        "query_count": len(manifest["queries"]),
        "download": download,
        "fetch_metadata": fetch_metadata,
        "request_delay_seconds": request_delay_seconds,
        "actions": actions,
        "limitations": source.limitations,
    }


def _parse_company(raw: dict[str, Any], index: int) -> SecCompany:
    if not isinstance(raw, dict):
        raise ValueError(f"SEC company row {index} must be an object")
    ticker = _required(raw, "ticker", f"companies[{index}]").upper()
    cik = re.sub(r"\D+", "", _required(raw, "cik", f"companies[{index}]"))
    return SecCompany(ticker=ticker, company_name=_required(raw, "company_name", f"companies[{index}]"), cik=cik.zfill(10))


def _parse_filing(raw: dict[str, Any], index: int) -> SecFilingSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"SEC filing row {index} must be an object")
    ticker = _required(raw, "ticker", f"filings[{index}]").upper()
    company_name = _required(raw, "company_name", f"filings[{index}]")
    cik = re.sub(r"\D+", "", _required(raw, "cik", f"filings[{index}]")).zfill(10)
    accession = _required(raw, "accession_number", f"filings[{index}]")
    primary_document = _required(raw, "primary_document", f"filings[{index}]")
    source_url = raw.get("source_url") or _archive_url(cik, accession, primary_document)
    return SecFilingSpec(
        ticker=ticker,
        company_name=company_name,
        cik=cik,
        accession_number=accession,
        form_type=_required(raw, "form_type", f"filings[{index}]"),
        filing_date=_required(raw, "filing_date", f"filings[{index}]"),
        primary_document=primary_document,
        source_url=str(source_url),
        source_format=str(raw.get("source_format") or _source_format(primary_document)).lower(),
    )


def _recent_filings_from_submission(company: SecCompany, payload: dict[str, Any], form_type: str, limit: int) -> list[SecFilingSpec]:
    recent = payload.get("filings", {}).get("recent", {}) if isinstance(payload, dict) else {}
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    specs = []
    for form, accession, filing_date, primary_doc in zip(forms, accessions, filing_dates, primary_docs):
        if form != form_type:
            continue
        specs.append(SecFilingSpec(
            ticker=company.ticker,
            company_name=company.company_name,
            cik=company.cik,
            accession_number=accession,
            form_type=form,
            filing_date=filing_date,
            primary_document=primary_doc,
            source_url=_archive_url(company.cik, accession, primary_doc),
            source_format=_source_format(primary_doc),
        ))
        if len(specs) >= limit:
            break
    return specs


def _select_companies(companies: list[SecCompany], tickers: list[str] | None, sample_size: int) -> list[SecCompany]:
    if tickers:
        wanted = {ticker.upper() for ticker in tickers}
        selected = [company for company in companies if company.ticker in wanted]
        missing = sorted(wanted - {company.ticker for company in selected})
        if missing:
            raise ValueError(f"Unknown SEC ticker(s): {missing}")
        return selected
    return companies[:sample_size]


def _query_templates(document_id: str, spec: SecFilingSpec) -> list[dict[str, Any]]:
    return [
        {
            "query_id": f"{document_id}_risk_factors",
            "query": f"What risk factors are discussed in {spec.ticker}'s {spec.form_type} filing?",
            "category": "sec_risk_factors_template",
            "target_document_ids": [document_id],
            "relevant_pages": [],
            "relevant_chunk_ids": [],
            "expected_answer_hints": ["risk factors"],
            "citation_required": True,
        },
        {
            "query_id": f"{document_id}_revenue",
            "query": f"What does {spec.ticker}'s {spec.form_type} filing say about revenue or results of operations?",
            "category": "sec_financial_performance_template",
            "target_document_ids": [document_id],
            "relevant_pages": [],
            "relevant_chunk_ids": [],
            "expected_answer_hints": ["revenue", "operations"],
            "citation_required": True,
        },
    ]


def _require_sec_user_agent(user_agent: str | None) -> None:
    if not user_agent or not user_agent.strip():
        raise ValueError("SEC network acquisition requires SEC_USER_AGENT or --sec-user-agent with a real contact string")


def _archive_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_no_zeros = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    return f"{SEC_ARCHIVES_BASE}/{cik_no_zeros}/{accession_no_dashes}/{primary_document}"


def _document_id(spec: SecFilingSpec) -> str:
    return _slug(f"sec_{spec.ticker}_{spec.form_type}_{spec.accession_number}")


def _local_filename(spec: SecFilingSpec) -> str:
    suffix = Path(spec.primary_document).suffix.lower() or (".pdf" if spec.source_format == "pdf" else ".html")
    return f"{_document_id(spec)}{suffix}"


def _source_format(primary_document: str) -> str:
    suffix = Path(primary_document).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".htm", ".html", ".txt", ".xml"}:
        return "html"
    return "html"


def _required(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
