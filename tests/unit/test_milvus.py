import pytest
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException


def test_milvus_service_available():
    try:
        client = MilvusClient(
            uri="http://localhost:19530",
            timeout=5,
        )
    except MilvusException as exc:
        pytest.skip(reason=f"Milvus is not available on localhost:19530: {exc}")

    collections = client.list_collections()

    assert isinstance(collections, list)