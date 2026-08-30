from collections.abc import Collection, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    precision_at_5: float


def hit_at_k(
    ranked_pages: Sequence[int],
    relevant_pages: Collection[int],
    k: int,
) -> float:
    top_k = ranked_pages[:k]
    return float(any(page in relevant_pages for page in top_k))


def reciprocal_rank(
    ranked_pages: Sequence[int],
    relevant_pages: Collection[int],
) -> float:
    for rank, page in enumerate(ranked_pages, start=1):
        if page in relevant_pages:
            return 1.0 / rank

    return 0.0


def precision_at_k(
    ranked_pages: Sequence[int],
    relevant_pages: Collection[int],
    k: int,
) -> float:
    top_k = ranked_pages[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        page in relevant_pages
        for page in top_k
    )

    return relevant_count / len(top_k)


def evaluate_retrieval(
    ranked_pages: Sequence[int],
    relevant_pages: Collection[int],
) -> RetrievalMetrics:
    return RetrievalMetrics(
        hit_at_1=hit_at_k(ranked_pages, relevant_pages, 1),
        hit_at_3=hit_at_k(ranked_pages, relevant_pages, 3),
        hit_at_5=hit_at_k(ranked_pages, relevant_pages, 5),
        mrr=reciprocal_rank(ranked_pages, relevant_pages),
        precision_at_5=precision_at_k(
            ranked_pages,
            relevant_pages,
            5,
        ),
    )