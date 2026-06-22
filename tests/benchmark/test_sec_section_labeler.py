from benchmarks.sec_section_labeler import extract_section_labels_from_pages


def test_sec_section_labeler_skips_table_of_contents_hits():
    pages = [
        "Table of Contents Item 1. Business 4 Item 1A. Risk Factors 9 Item 7. Managements Discussion and Analysis 20 Item 8. Financial Statements 30",
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

    assert starts["item_1_business"] == 2
    assert starts["item_1a_risk_factors"] == 3
    assert starts["item_7_mda"] == 5
    assert starts["item_7a_market_risk"] == 6
    assert starts["item_8_financial_statements"] == 7
    assert starts["item_9a_controls"] == 8
    assert all("Table of Contents" not in str(label) for label in labels)


def test_sec_section_labeler_handles_same_page_adjacent_sections():
    pages = [
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk Item 8. Financial Statements and Supplementary Data",
    ]

    labels = extract_section_labels_from_pages(pages)
    ranges = {label["section_id"]: (label["start_page"], label["end_page"]) for label in labels}

    assert ranges["item_7a_market_risk"] == (1, 1)
    assert ranges["item_8_financial_statements"] == (1, 1)