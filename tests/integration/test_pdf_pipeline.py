from pathlib import Path

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.splitter import ChunkingConfig, split_documents

PDF_PATH = Path("data/source/swe_at_google.2.pdf")


def test_pdf_ingestion_pipeline():
    documents = load_pdf(
        PDF_PATH,
        source_id="software-engineering-at-google",
    )

    chunks = split_documents(
        documents,
        ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
        ),
    )

    assert len(documents) > 500
    assert len(chunks) > len(documents)

    first = chunks[0]

    assert first.page_content
    assert first.metadata["source_id"] == "software-engineering-at-google"
    assert "document_version" in first.metadata
    assert "page_number" in first.metadata
    assert "chunk_index" in first.metadata
    assert "chunk_id" in first.metadata