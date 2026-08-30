from pathlib import Path

from src.evaluation.dataset import load_retrieval_eval_cases


def test_load_retrieval_eval_cases(tmp_path: Path):
    path = tmp_path / "retrieval_cases.jsonl"

    path.write_text(
        (
            '{"case_id":"case-1",'
            '"question":"What is Hyrum\'s Law?",'
            '"relevant_pages":[36,38]}\n'
        ),
        encoding="utf-8",
    )

    cases = load_retrieval_eval_cases(path)

    assert len(cases) == 1

    case = cases[0]

    assert case.case_id == "case-1"
    assert case.question == "What is Hyrum's Law?"
    assert case.relevant_pages == (36, 38)