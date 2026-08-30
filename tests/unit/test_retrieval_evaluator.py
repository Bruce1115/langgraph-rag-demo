import pytest

from src.evaluation.dataset import RetrievalEvalCase
from src.evaluation.evaluator import (
    evaluate_case,
    evaluate_case_with_retriever,
)
from src.retrieval.retriever import RetrievedChunk


def test_evaluate_case():
    case = RetrievalEvalCase(
        case_id="hyrums-law-definition",
        question="What is Hyrum's Law?",
        relevant_pages=(36, 38, 474, 560),
    )

    result = evaluate_case(
        case,
        ranked_pages=[36, 474, 38, 5, 560],
    )

    assert result.case_id == "hyrums-law-definition"
    assert result.question == "What is Hyrum's Law?"

    assert result.metrics.hit_at_1 == 1.0
    assert result.metrics.hit_at_3 == 1.0
    assert result.metrics.hit_at_5 == 1.0
    assert result.metrics.mrr == 1.0
    assert result.metrics.precision_at_5 == pytest.approx(0.8)


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                source_id="book-1",
                page_number=36,
                chunk_index=0,
                text="Hyrum's Law...",
                score=0.58,
            ),
            RetrievedChunk(
                chunk_id="chunk-2",
                source_id="book-1",
                page_number=36,
                chunk_index=1,
                text="More about Hyrum's Law...",
                score=0.55,
            ),
            RetrievedChunk(
                chunk_id="chunk-3",
                source_id="book-1",
                page_number=474,
                chunk_index=0,
                text="Hyrum's Law...",
                score=0.50,
            ),
        ]


def test_evaluate_case_with_retriever():
    case = RetrievalEvalCase(
        case_id="hyrums-law-definition",
        question="What is Hyrum's Law?",
        relevant_pages=(36, 474),
    )

    result = evaluate_case_with_retriever(
        case,
        FakeRetriever(),
        limit=5,
    )

    assert result.metrics.hit_at_1 == 1.0
    assert result.metrics.hit_at_3 == 1.0
    assert result.metrics.mrr == 1.0
    assert result.metrics.precision_at_5 == 1.0

    assert len(result.chunks) == 3
    assert result.chunks[0].page_number == 36
    assert result.chunks[0].chunk_id == "chunk-1"