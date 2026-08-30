from src.evaluation.evaluator import RetrievalEvalResult
from src.evaluation.failures import top1_failures
from src.evaluation.retrieval import RetrievalMetrics


def make_result(
    case_id: str,
    *,
    hit_at_1: float,
) -> RetrievalEvalResult:
    return RetrievalEvalResult(
        case_id=case_id,
        question="test question",
        relevant_pages=(1,),
        ranked_pages=(1, 2, 3),
        chunks=(),
        metrics=RetrievalMetrics(
            hit_at_1=hit_at_1,
            hit_at_3=1.0,
            hit_at_5=1.0,
            mrr=1.0,
            precision_at_5=0.4,
        ),
    )


def test_top1_failures():
    results = [
        make_result("case-1", hit_at_1=1.0),
        make_result("case-2", hit_at_1=0.0),
        make_result("case-3", hit_at_1=1.0),
        make_result("case-4", hit_at_1=0.0),
    ]

    failures = top1_failures(results)

    assert [
        result.case_id
        for result in failures
    ] == [
        "case-2",
        "case-4",
    ]