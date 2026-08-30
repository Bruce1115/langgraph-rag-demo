from collections.abc import Sequence
from dataclasses import dataclass

from src.evaluation.dataset import RetrievalEvalCase
from src.evaluation.retrieval import RetrievalMetrics, evaluate_retrieval
from src.retrieval.retriever import RetrievedChunk, Retriever


@dataclass(frozen=True)
class RetrievalEvalResult:
    case_id: str
    question: str
    relevant_pages: tuple[int, ...]
    ranked_pages: tuple[int, ...]
    chunks: tuple[RetrievedChunk, ...]
    metrics: RetrievalMetrics


def evaluate_case(
    case: RetrievalEvalCase,
    ranked_pages: Sequence[int],
    *,
    chunks: Sequence[RetrievedChunk] = (),
) -> RetrievalEvalResult:
    return RetrievalEvalResult(
        case_id=case.case_id,
        question=case.question,
        relevant_pages=case.relevant_pages,
        ranked_pages=tuple(ranked_pages),
        chunks=tuple(chunks),
        metrics=evaluate_retrieval(
            ranked_pages,
            case.relevant_pages,
        ),
    )


def evaluate_case_with_retriever(
    case: RetrievalEvalCase,
    retriever: Retriever,
    *,
    limit: int = 5,
) -> RetrievalEvalResult:
    chunks = retriever.retrieve(
        case.question,
        limit=limit,
    )

    ranked_pages = [
        chunk.page_number
        for chunk in chunks
    ]

    return evaluate_case(
        case,
        ranked_pages,
        chunks=chunks,
    )