from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "bailian"
    model: str = "text-embedding-v4"
    dimensions: int = 1024
    batch_size: int = 10


@dataclass(frozen=True)
class MilvusConfig:
    uri: str = "http://localhost:19530"
    collection_name: str = "rag_chunks_v1"

    metric_type: str = "COSINE"
    index_type: str = "FLAT"