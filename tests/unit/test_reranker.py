from unittest.mock import MagicMock

from src.retrieval.reranker import (
    LLMReranker,
    RerankingRetriever,
    RerankOrder,
)
from src.retrieval.retriever import RetrievedChunk


def make_chunk(
    chunk_id: str,
    page: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id="book",
        page_number=page,
        chunk_index=0,
        text=f"content from page {page}",
        score=0.5,
    )


def test_llm_reranker_reorders_chunks():
    model = MagicMock()
    structured_model = MagicMock()

    model.with_structured_output.return_value = (
        structured_model
    )

    structured_model.invoke.return_value = RerankOrder(
        ranked_ids=[
            "candidate_2",
            "candidate_0",
            "candidate_1",
        ],
    )

    reranker = LLMReranker(model)

    chunks = [
        make_chunk("a", 10),
        make_chunk("b", 20),
        make_chunk("c", 30),
    ]

    results = reranker.rerank(
        "test question",
        chunks,
        limit=2,
    )

    assert [
        chunk.chunk_id
        for chunk in results
    ] == [
        "c",
        "a",
    ]

class FakeRetriever:
    def __init__(self) -> None:
        self.received_query: str | None = None
        self.received_limit: int | None = None

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        self.received_query = query
        self.received_limit = limit

        return [
            make_chunk("a", 10),
            make_chunk("b", 20),
            make_chunk("c", 30),
        ]


class FakeReranker:
    def __init__(self) -> None:
        self.received_query: str | None = None
        self.received_limit: int | None = None

    def rerank(
        self,
        query: str,
        chunks,
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        self.received_query = query
        self.received_limit = limit

        return list(reversed(chunks))[:limit]


def test_reranking_retriever():
    base_retriever = FakeRetriever()
    reranker = FakeReranker()

    retriever = RerankingRetriever(
        base_retriever=base_retriever,
        reranker=reranker,
        candidate_limit=10,
    )

    results = retriever.retrieve(
        "test question",
        limit=2,
    )

    assert base_retriever.received_query == "test question"
    assert base_retriever.received_limit == 10

    assert reranker.received_query == "test question"
    assert reranker.received_limit == 2

    assert [
        chunk.chunk_id
        for chunk in results
    ] == [
        "c",
        "b",
    ]