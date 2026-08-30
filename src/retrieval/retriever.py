from dataclasses import dataclass
from typing import Protocol

from src.infrastructure.embeddings import Embedder


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_id: str
    page_number: int
    chunk_index: int
    text: str
    score: float

class VectorStore(Protocol):
    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        document_type: str | None = None,
    ) -> list[dict]:
        ...

class HybridVectorStore(Protocol):
    def search_hybrid(
        self,
        query_vector: list[float],
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
        document_type: str | None = None,
    ) -> list[dict]:
        ...

class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        ...


class MilvusRetriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        document_type: str | None = "content",
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._document_type = document_type

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        query_vector = self._embedder.embed_query(query)

        results = self._store.search(
            query_vector,
            limit=limit,
            document_type=self._document_type,
        )

        chunks: list[RetrievedChunk] = []

        for result in results:
            entity = result["entity"]

            chunks.append(
                RetrievedChunk(
                    chunk_id=entity["chunk_id"],
                    source_id=entity["source_id"],
                    page_number=entity["page_number"],
                    chunk_index=entity["chunk_index"],
                    text=entity["text"],
                    score=result["distance"],
                )
            )

        return chunks

class MilvusHybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        store: HybridVectorStore,
        *,
        candidate_limit: int = 20,
        document_type: str | None = "content",
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._candidate_limit = candidate_limit
        self._document_type = document_type

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        query_vector = self._embedder.embed_query(query)

        results = self._store.search_hybrid(
            query_vector=query_vector,
            query=query,
            limit=limit,
            candidate_limit=self._candidate_limit,
            document_type=self._document_type,
        )

        chunks: list[RetrievedChunk] = []

        for result in results:
            entity = result["entity"]

            chunks.append(
                RetrievedChunk(
                    chunk_id=entity["chunk_id"],
                    source_id=entity["source_id"],
                    page_number=entity["page_number"],
                    chunk_index=entity["chunk_index"],
                    text=entity["text"],
                    score=result["distance"],
                )
            )

        return chunks