from benchmarks.sec_section_labeler import extract_section_labels_from_pages


def test_sec_section_labeler_skips_table_of_contents_hits():
    pages = [
        "Table of Contents Item 1. Business 4 Item 1A. Risk Factors 9 Item 7. Managements Discussion and Analysis 20 Item 8. Financial Statements 30",
        "Item 1. Business Item 1A. Risk Factors Item 1B. Item 2. Item 3. Item 7. Managements Discussion and Analysis Item 8. Financial Statements",
        "Item 1. Business The registrant designs and sells products.",
        "Item 1A. Risk Factors The registrant faces market and operating risks.",
        "Additional risk factor text.",
        "Item 7. Managements Discussion and Analysis of Financial Condition and Results of Operations",
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
        "Item 8. Financial Statements and Supplementary Data",
        "Item 9A. Controls and Procedures",
    ]

    labels = extract_section_labels_from_pages(pages)
    starts = {label["section_id"]: label["start_page"] for label in labels}

    assert starts["item_1_business"] == 3
    assert starts["item_1a_risk_factors"] == 4
    assert starts["item_7_mda"] == 6
    assert starts["item_7a_market_risk"] == 7
    assert starts["item_8_financial_statements"] == 8
    assert starts["item_9a_controls"] == 9
    assert all("Table of Contents" not in str(label) for label in labels)


def test_sec_section_labeler_handles_same_page_adjacent_sections():
    pages = [
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk Item 8. Financial Statements and Supplementary Data",
    ]

    labels = extract_section_labels_from_pages(pages)
    ranges = {label["section_id"]: (label["start_page"], label["end_page"]) for label in labels}

    assert ranges["item_7a_market_risk"] == (1, 1)
    assert ranges["item_8_financial_statements"] == (1, 1)

def test_sec_section_labeler_collapses_range_when_intervening_headings_are_missing():
    pages = [
        "Item 1A. Risk Factors The registrant faces market and operating risks.",
        "Risk factor continuation.",
        "Item 9A. Controls and Procedures",
    ]

    labels = extract_section_labels_from_pages(pages)
    ranges = {label["section_id"]: (label["start_page"], label["end_page"], label["confidence"]) for label in labels}

    assert ranges["item_1a_risk_factors"] == (1, 1, "medium")
    assert ranges["item_9a_controls"] == (3, 3, "high")
