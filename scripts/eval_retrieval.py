from pathlib import Path
from statistics import fmean

from dotenv import load_dotenv

from src.config import EmbeddingConfig, MilvusConfig
from src.evaluation.dataset import load_retrieval_eval_cases
from src.evaluation.evaluator import (
    RetrievalEvalResult,
    evaluate_case_with_retriever,
)
from src.evaluation.failures import top1_failures
from src.infrastructure.embeddings import BailianEmbedder
from src.infrastructure.milvus import MilvusStore
from src.retrieval.retriever import MilvusRetriever

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "evals" / "retrieval_cases.jsonl"


def print_summary(results: list[RetrievalEvalResult]) -> None:
    print("\n=== Retrieval Eval Summary ===")
    print(f"Cases:       {len(results)}")
    print(
        f"Hit@1:       "
        f"{fmean(r.metrics.hit_at_1 for r in results):.3f}"
    )
    print(
        f"Hit@3:       "
        f"{fmean(r.metrics.hit_at_3 for r in results):.3f}"
    )
    print(
        f"Hit@5:       "
        f"{fmean(r.metrics.hit_at_5 for r in results):.3f}"
    )
    print(
        f"MRR:         "
        f"{fmean(r.metrics.mrr for r in results):.3f}"
    )
    print(
        f"Precision@5: "
        f"{fmean(r.metrics.precision_at_5 for r in results):.3f}"
    )

def print_top1_failures(
    results: list[RetrievalEvalResult],
) -> None:
    failures = top1_failures(results)

    print("\n=== Top-1 Failures ===")
    print(f"Failures: {len(failures)}")

    for result in failures:
        print(f"\n[{result.case_id}]")
        print(f"Question: {result.question}")

        if not result.chunks:
            continue

        top_chunk = result.chunks[0]

        print(
            "Top 1:     "
            f"page={top_chunk.page_number} "
            f"chunk={top_chunk.chunk_index} "
            f"score={top_chunk.score:.4f}"
        )

        for rank, chunk in enumerate(
            result.chunks,
            start=1,
        ):
            if chunk.page_number in result.relevant_pages:
                print(
                    "First hit: "
                    f"rank={rank} "
                    f"page={chunk.page_number} "
                    f"chunk={chunk.chunk_index} "
                    f"score={chunk.score:.4f}"
                )
                break

def main() -> None:
    load_dotenv()

    cases = load_retrieval_eval_cases(DATASET_PATH)

    if not cases:
        raise RuntimeError("Retrieval eval dataset is empty.")

    embedder = BailianEmbedder(
        EmbeddingConfig(),
    )

    store = MilvusStore(
        MilvusConfig(),
    )

    retriever = MilvusRetriever(
        embedder=embedder,
        store=store,
    )

    results: list[RetrievalEvalResult] = []

    for case in cases:
        result = evaluate_case_with_retriever(
            case,
            retriever,
            limit=5,
        )
        results.append(result)

        print(f"\n[{result.case_id}]")
        print(f"Question: {result.question}")
        print(f"Pages:    {result.ranked_pages}")

        print("\nRetrieved passages:")

        for rank, chunk in enumerate(result.chunks, start=1):
            preview = chunk.text.replace("\n", " ")[:300]

            print(
                f"{rank}. "
                f"page={chunk.page_number} "
                f"chunk={chunk.chunk_index} "
                f"score={chunk.score:.4f}"
            )
            print(f"   {preview}")

        print(f"Hit@1:    {result.metrics.hit_at_1:.0f}")
        print(f"Hit@3:    {result.metrics.hit_at_3:.0f}")
        print(f"Hit@5:    {result.metrics.hit_at_5:.0f}")
        print(f"MRR:      {result.metrics.mrr:.3f}")
        print(
            f"P@5:      "
            f"{result.metrics.precision_at_5:.3f}"
        )

    print_summary(results)
    print_top1_failures(results)


if __name__ == "__main__":
    main()