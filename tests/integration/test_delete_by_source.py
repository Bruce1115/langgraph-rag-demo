from src.config import EmbeddingConfig, MilvusConfig
from src.infrastructure.milvus import MilvusStore


def test_delete_by_source() -> None:
    embedding_config = EmbeddingConfig()

    store = MilvusStore(
        MilvusConfig(
            collection_name="rag_chunks_test",
        )
    )

    # 保证每次测试都是干净环境
    store.drop_collection()

    store.ensure_collection(
        vector_dimension=embedding_config.dimensions
    )

    record = {
        "chunk_id": "delete-test-001",
        "source_id": "delete-test-source",
        "source_path": "test.pdf",
        "document_version": "abc123",
        "chunking_version": "v1",
        "page_number": 1,
        "chunk_index": 0,
        "text": "Test document",
        "vector": [0.0] * embedding_config.dimensions,
    }

    # 1. Insert
    insert_result = store.insert([record])

    print("insert result:", insert_result)

    assert insert_result["insert_count"] == 1

    # 2. Flush，确保测试里立即可读
    store.flush()

    # 3. Read back
    results = store.get("delete-test-001")

    print("get result:", results)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "delete-test-001"
    assert results[0]["source_id"] == "delete-test-source"
    assert results[0]["text"] == "Test document"

    # 4. Delete by source
    deleted = store.delete_by_source(
        "delete-test-source"
    )

    print("delete count:", deleted)

    assert deleted == 1

    store.flush()

    # 5. Confirm deleted
    results_after_delete = store.get(
        "delete-test-001"
    )

    print("after delete:", results_after_delete)

    assert results_after_delete == []