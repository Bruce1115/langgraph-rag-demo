import pytest

from src.evaluation.retrieval import evaluate_retrieval


def test_evaluate_retrieval():
    ranked_pages = [
        10,
        36,
        99,
        474,
        5,
    ]

    relevant_pages = {
        36,
        474,
    }

    metrics = evaluate_retrieval(
        ranked_pages,
        relevant_pages,
    )

    assert metrics.hit_at_1 == 0.0
    assert metrics.hit_at_3 == 1.0
    assert metrics.hit_at_5 == 1.0
    assert metrics.mrr == pytest.approx(0.5)
    assert metrics.precision_at_5 == pytest.approx(0.4)