import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalEvalCase:
    case_id: str
    question: str
    relevant_pages: tuple[int, ...]


def load_retrieval_eval_cases(
    path: Path,
) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            cases.append(
                RetrievalEvalCase(
                    case_id=data["case_id"],
                    question=data["question"],
                    relevant_pages=tuple(data["relevant_pages"]),
                )
            )

    return cases