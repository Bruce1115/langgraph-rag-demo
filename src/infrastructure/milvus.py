from typing import Any

from pymilvus import DataType, MilvusClient

from src.config import MilvusConfig


class MilvusStore:
    def __init__(self, config: MilvusConfig) -> None:
        self._config = config

        self._client = MilvusClient(
            uri=config.uri,
        )

    def ensure_collection(self, vector_dimension: int) -> None:
        if self._client.has_collection(
            collection_name=self._config.collection_name
        ):
            return

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
        )

        schema.add_field(
            field_name="source_id",
            datatype=DataType.VARCHAR,
            max_length=256,
        )

        schema.add_field(
            field_name="source_path",
            datatype=DataType.VARCHAR,
            max_length=1024,
        )

        schema.add_field(
            field_name="document_version",
            datatype=DataType.VARCHAR,
            max_length=64,
        )

        schema.add_field(
            field_name="chunking_version",
            datatype=DataType.VARCHAR,
            max_length=32,
        )

        schema.add_field(
            field_name="document_type",
            datatype=DataType.VARCHAR,
            max_length=32,
        )

        schema.add_field(
            field_name="page_number",
            datatype=DataType.INT64,
        )

        schema.add_field(
            field_name="chunk_index",
            datatype=DataType.INT64,
        )

        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=16384,
        )

        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=vector_dimension,
        )

        index_params = self._client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type=self._config.index_type,
            metric_type=self._config.metric_type,
        )

        self._client.create_collection(
            collection_name=self._config.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert(self, records: list[dict]) -> dict:
        if not records:
            return {"insert_count": 0}

        return self._client.insert(
            collection_name=self._config.collection_name,
            data=records,
        )


    def get(self, chunk_id: str) -> list[dict]:
        return self._client.get(
            collection_name=self._config.collection_name,
            ids=[chunk_id],
            output_fields=[
                "chunk_id",
                "source_id",
                "source_path",
                "document_version",
                "chunking_version",
                "document_type",
                "page_number",
                "chunk_index",
                "text",
            ],
        )


    def flush(self) -> None:
        self._client.flush(
            collection_name=self._config.collection_name,
        )


    def delete_by_source(self, source_id: str) -> int:
        result = self._client.delete(
            collection_name=self._config.collection_name,
            filter=f'source_id == "{source_id}"',
        )

        return result["delete_count"]


    def drop_collection(self) -> None:
        if self._client.has_collection(
            collection_name=self._config.collection_name
        ):
            self._client.drop_collection(
                collection_name=self._config.collection_name
            )

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        document_type: str | None = None,
    ) -> list[dict]:
        filter_expr = ""

        if document_type is not None:
            filter_expr = (
                f'document_type == "{document_type}"'
            )

        results = self._client.search(
            collection_name=self._config.collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=limit,
            filter=filter_expr,
            output_fields=[
                "chunk_id",
                "source_id",
                "document_type",
                "page_number",
                "chunk_index",
                "text",
            ],
            search_params={
                "metric_type": self._config.metric_type,
            },
        )

        return results[0]