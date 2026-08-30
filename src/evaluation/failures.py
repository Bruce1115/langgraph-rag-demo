from collections.abc import Sequence

from src.evaluation.evaluator import RetrievalEvalResult


def top1_failures(
    results: Sequence[RetrievalEvalResult],
) -> list[RetrievalEvalResult]:
    return [
        result
        for result in results
        if result.metrics.hit_at_1 == 0.0
    ]