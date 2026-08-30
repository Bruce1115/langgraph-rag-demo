from langchain_core.documents import Document

from src.infrastructure.embeddings import Embedder
from src.infrastructure.milvus import MilvusStore


def index_documents(
    chunks: list[Document],
    embedder: Embedder,
    store: MilvusStore,
) -> int:
    if not chunks:
        return 0

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = embedder.embed_documents(texts)

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: "
            f"{len(chunks)} chunks, "
            f"{len(vectors)} vectors"
        )

    records = []

    for chunk, vector in zip(
        chunks,
        vectors,
        strict=True,
    ):
        metadata = chunk.metadata

        records.append(
            {
                "chunk_id": metadata["chunk_id"],
                "source_id": metadata["source_id"],
                "source_path": metadata["source_path"],
                "document_version": metadata["document_version"],
                "chunking_version": metadata["chunking_version"],
                "document_type": metadata["document_type"],
                "page_number": metadata["page_number"],
                "chunk_index": metadata["chunk_index"],
                "text": chunk.page_content,
                "vector": vector,
            }
        )

    result = store.insert(records)

    if result["insert_count"] != len(records):
        raise RuntimeError(
            f"Insert count mismatch: "
            f"expected {len(records)}, "
            f"inserted {result['insert_count']}"
        )

    return result["insert_count"]