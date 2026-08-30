import argparse

from dotenv import load_dotenv

from src.config import EmbeddingConfig, MilvusConfig
from src.infrastructure.embeddings import BailianEmbedder
from src.infrastructure.milvus import MilvusStore
from src.retrieval.retriever import MilvusRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect retrieval results."
    )

    parser.add_argument(
        "query",
        help="Search query",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Maximum text characters to display per result",
    )

    args = parser.parse_args()

    load_dotenv()

    embedder = BailianEmbedder(
        EmbeddingConfig()
    )

    store = MilvusStore(
        MilvusConfig()
    )

    retriever = MilvusRetriever(
        embedder=embedder,
        store=store,
    )

    results = retriever.retrieve(
        args.query,
        limit=args.limit,
    )

    print()
    print(f"Query: {args.query}")
    print(f"Top K: {args.limit}")
    print("=" * 100)

    for rank, chunk in enumerate(results, start=1):
        print(
            f"\nRank {rank}"
            f" | score={chunk.score:.4f}"
            f" | page={chunk.page_number}"
            f" | chunk={chunk.chunk_index}"
        )

        print(f"chunk_id: {chunk.chunk_id}")

        print("-" * 100)

        text = chunk.text

        if len(text) > args.max_chars:
            text = text[: args.max_chars] + "\n...[truncated]"

        print(text)

        print("=" * 100)


if __name__ == "__main__":
    main()