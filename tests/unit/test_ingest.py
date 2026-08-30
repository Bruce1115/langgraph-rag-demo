from langchain_core.documents import Document

from src.ingestion.splitter import ChunkingConfig, split_documents


def test_splitter_preserves_document_type_metadata():
    document = Document(
        page_content="A " * 1000,
        metadata={
            "source_id": "book-1",
            "document_version": "v1",
            "page_index": 4,
            "page_number": 5,
            "document_type": "toc",
        },
    )

    chunks = split_documents(
        [document],
        ChunkingConfig(),
    )

    assert chunks

    for chunk in chunks:
        assert chunk.metadata["document_type"] == "toc"
        assert chunk.metadata["page_number"] == 5
        assert chunk.metadata["source_id"] == "book-1"