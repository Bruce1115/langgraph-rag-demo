from pathlib import Path

from src.ingestion.pdf_loader import load_pdf


def test_load_pdf():
    docs = load_pdf(
        Path("data/source/swe_at_google.2.pdf"),
        source_id="software-engineering-at-google",
    )

    assert docs

    first = docs[0]

    assert first.page_content
    assert first.metadata["source_id"] == "software-engineering-at-google"
    assert first.metadata["source_type"] == "pdf"
    assert first.metadata["page_index"] == 0
    assert first.metadata["page_number"] == 1
    assert len(first.metadata["document_version"]) == 64