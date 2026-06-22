import json

from benchmarks.sec_metadata_enrichment import delete_namespace_if_exists, enrich_metadata, is_table_of_contents_chunk, section_for_page


def _lookup():
    record = {
        "document_id": "sec_aapl_2025",
        "service_document_id": "service-a",
        "filename": "sec_edgar_rendered/sec_aapl_2025.pdf",
        "filename_basename": "sec_aapl_2025.pdf",
        "doc_type": "financial_report",
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "filing_date": "2025-10-31",
        "filing_year": "2025",
        "form_type": "10-K",
        "accession_number": "0000320193-25-000079",
        "section_labels": [
            {"section_id": "item_1_business", "section_name": "Item 1. Business", "start_page": 8, "end_page": 11, "confidence": "high"},
            {"section_id": "item_1a_risk_factors", "section_name": "Item 1A. Risk Factors", "start_page": 12, "end_page": 31, "confidence": "high"},
        ],
    }
    return {
        "sec_aapl_2025": record,
        "service-a": record,
        "sec_edgar_rendered/sec_aapl_2025.pdf": record,
        "sec_aapl_2025.pdf": record,
    }


def test_section_for_page_maps_known_range():
    labels = _lookup()["sec_aapl_2025"]["section_labels"]

    assert section_for_page(labels, 10)["section_id"] == "item_1_business"
    assert section_for_page(labels, 20)["section_id"] == "item_1a_risk_factors"
    assert section_for_page(labels, 99) is None


def test_enrich_metadata_adds_sec_fields_from_service_document_id():
    metadata = {
        "doc_id": "service-a",
        "filename": "sec_aapl_2025.pdf",
        "page": 12,
        "chunk_id": 4,
        "text": "Item 1A. Risk Factors business risk discussion",
    }

    enriched = enrich_metadata(metadata, _lookup())

    assert enriched["manifest_document_id"] == "sec_aapl_2025"
    assert enriched["document_id"] == "sec_aapl_2025"
    assert enriched["service_document_id"] == "service-a"
    assert enriched["ticker"] == "AAPL"
    assert enriched["filing_year"] == "2025"
    assert enriched["accession_number"] == "0000320193-25-000079"
    assert enriched["page_number"] == 12
    assert enriched["section_id"] == "item_1a_risk_factors"
    assert enriched["section_name"] == "Item 1A. Risk Factors"
    assert enriched["section_confidence"] == "high"
    assert enriched["is_table_of_contents"] is False


def test_enrich_metadata_marks_unknown_section_without_guessing():
    metadata = {
        "doc_id": "service-a",
        "filename": "sec_aapl_2025.pdf",
        "page": 7,
        "text": "Item 1. Item 1A. Item 1B. Item 2. Item 3. Item 7. Item 8.",
    }

    enriched = enrich_metadata(metadata, _lookup())

    assert enriched["section_id"] == "unknown"
    assert enriched["section_name"] == "unknown"
    assert enriched["section_confidence"] == "unknown"
    assert enriched["is_table_of_contents"] is True


def test_enrich_metadata_can_match_filename_when_service_id_is_missing():
    metadata = {
        "filename": "sec_aapl_2025.pdf",
        "page": "9",
        "text": "Business overview",
    }

    enriched = enrich_metadata(metadata, _lookup())

    assert enriched["document_id"] == "sec_aapl_2025"
    assert enriched["page_number"] == 9
    assert enriched["section_id"] == "item_1_business"


def test_table_of_contents_detection_requires_unknown_section_for_dense_items():
    dense_items = "Item 1. Item 1A. Item 1B. Item 2. Item 3. Item 7."

    assert is_table_of_contents_chunk("Table of Contents " + dense_items, None) is True
    assert is_table_of_contents_chunk(dense_items, None) is True
    assert is_table_of_contents_chunk(dense_items, {"section_id": "item_1_business"}) is False


def test_enriched_metadata_is_json_serializable():
    enriched = enrich_metadata({"doc_id": "service-a", "page": 12, "text": "risk"}, _lookup())

    json.dumps(enriched)

def test_delete_namespace_if_exists_tolerates_missing_namespace():
    class MissingNamespaceIndex:
        def delete(self, *, delete_all, namespace):
            raise Exception("Namespace not found: tenant_eval_sec_sections_v2")

    assert delete_namespace_if_exists(MissingNamespaceIndex(), "tenant_eval_sec_sections_v2") is False


def test_delete_namespace_if_exists_reports_success():
    class ExistingNamespaceIndex:
        def __init__(self):
            self.calls = []

        def delete(self, *, delete_all, namespace):
            self.calls.append((delete_all, namespace))

    index = ExistingNamespaceIndex()

    assert delete_namespace_if_exists(index, "tenant_eval_sec_sections_v2") is True
    assert index.calls == [(True, "tenant_eval_sec_sections_v2")]
