import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT_DIR
    / "evals"
    / "generation_cases.jsonl"
)

DATASET_NAME = "swe-at-google-generation-eval"


def load_cases() -> list[dict]:
    cases = []

    with DATASET_PATH.open(
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            cases.append(
                json.loads(line)
            )

    return cases


def main() -> None:
    load_dotenv()

    client = Client()

    if client.has_dataset(
        dataset_name=DATASET_NAME,
    ):
        dataset = client.read_dataset(
            dataset_name=DATASET_NAME,
        )
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Generation evaluation dataset for "
                "Software Engineering at Google RAG."
            ),
        )

    existing_examples = list(
        client.list_examples(
            dataset_id=dataset.id,
        )
    )

    existing_by_case_id = {}

    for example in existing_examples:
        metadata = example.metadata or {}
        case_id = metadata.get("case_id")

        if case_id:
            existing_by_case_id[case_id] = example

    creates = []
    updates = []

    for case in load_cases():
        case_id = case["case_id"]

        inputs = {
            "question": case["question"],
        }

        outputs = {
            "reference_answer": (
                case["reference_answer"]
            ),
        }

        metadata = {
            "case_id": case_id,
            "relevant_pages": (
                case["relevant_pages"]
            ),
        }

        existing = existing_by_case_id.get(
            case_id
        )

        if existing is None:
            creates.append(
                {
                    "inputs": inputs,
                    "outputs": outputs,
                    "metadata": metadata,
                }
            )
        else:
            updates.append(
                {
                    "id": existing.id,
                    "inputs": inputs,
                    "outputs": outputs,
                    "metadata": metadata,
                }
            )

    if updates:
        client.update_examples(
            dataset_id=dataset.id,
            updates=updates,
        )

    if creates:
        client.create_examples(
            dataset_id=dataset.id,
            examples=creates,
        )

    print(f"Dataset: {DATASET_NAME}")
    print(f"Updated: {len(updates)}")
    print(f"Created: {len(creates)}")
    print(
        f"Total local cases: "
        f"{len(updates) + len(creates)}"
    )


if __name__ == "__main__":
    main()