from langchain_core.documents import Document

from src.ingestion.splitter import ChunkingConfig, split_documents


def test_split_documents_preserves_metadata():
    document = Document(
        page_content=(
            "Software engineering is programming integrated over time. "
            "This sentence is repeated to create enough content. "
        )
        * 100,
        metadata={
            "source_id": "test-book",
            "document_version": "abc123",
            "page_index": 0,
            "page_number": 1,
        },
    )

    chunks = split_documents(
        [document],
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
        ),
    )

    assert len(chunks) > 1

    first = chunks[0]

    assert first.metadata["source_id"] == "test-book"
    assert first.metadata["page_number"] == 1
    assert first.metadata["chunk_index"] == 0
    assert len(first.metadata["chunk_id"]) == 64

def test_chunk_ids_are_deterministic():
    document = Document(
        page_content="A long technical paragraph. " * 100,
        metadata={
            "source_id": "test-book",
            "document_version": "abc123",
            "page_index": 0,
            "page_number": 1,
        },
    )

    config = ChunkingConfig(
        chunk_size=100,
        chunk_overlap=20,
    )

    first_run = split_documents([document], config)
    second_run = split_documents([document], config)

    assert [c.metadata["chunk_id"] for c in first_run] == [
        c.metadata["chunk_id"] for c in second_run
    ]