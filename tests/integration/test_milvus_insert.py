from src.config import EmbeddingConfig, MilvusConfig
from src.infrastructure.milvus import MilvusStore


def test_insert_record() -> None:
    embedding_config = EmbeddingConfig()

    store = MilvusStore(
        MilvusConfig(
            collection_name="rag_chunks_test",
        )
    )

    store.ensure_collection(
        vector_dimension=embedding_config.dimensions
    )

    record = {
        "chunk_id": "test-001",
        "source_id": "test-source",
        "source_path": "test.pdf",
        "document_version": "abc123",
        "chunking_version": "v1",
        "page_number": 1,
        "chunk_index": 0,
        "text": "Software engineering is programming integrated over time.",
        "vector": [0.0] * embedding_config.dimensions,
    }

    store.insert([record])

    results = store.get("test-001")

    assert len(results) == 1
    assert results[0]["chunk_id"] == "test-001"
    assert results[0]["page_number"] == 1
    assert results[0]["text"] == record["text"]