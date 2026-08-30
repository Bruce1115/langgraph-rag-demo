import statistics
from pathlib import Path

import tiktoken

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.splitter import ChunkingConfig, split_documents


def main() -> None:
    documents = load_pdf(
        Path("data/source/swe_at_google.2.pdf"),
        source_id="software-engineering-at-google",
    )

    chunks = split_documents(
        documents,
        ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
        ),
    )

    encoder = tiktoken.get_encoding("cl100k_base")

    token_counts = [
        len(encoder.encode(chunk.page_content))
        for chunk in chunks
    ]

    empty_pages = [
        doc.metadata["page_number"]
        for doc in documents
        if not doc.page_content.strip()
    ]

    print(f"Pages:        {len(documents)}")
    print(f"Chunks:       {len(chunks)}")
    print(f"Avg/page:     {len(chunks) / len(documents):.2f}")
    print(f"Avg tokens:   {sum(token_counts) / len(token_counts):.1f}")
    print(f"Min tokens:   {min(token_counts)}")
    print(f"Max tokens:   {max(token_counts)}")
    print(f"Empty pages:  {len(empty_pages)}")

    if empty_pages:
        print(f"Empty page numbers: {empty_pages}")

    sorted_counts = sorted(token_counts)

    print()
    print("Token distribution:")
    print(f"P10:          {sorted_counts[int(len(sorted_counts) * 0.10)]}")
    print(f"P25:          {sorted_counts[int(len(sorted_counts) * 0.25)]}")
    print(f"P50:          {statistics.median(sorted_counts):.0f}")
    print(f"P75:          {sorted_counts[int(len(sorted_counts) * 0.75)]}")
    print(f"P90:          {sorted_counts[int(len(sorted_counts) * 0.90)]}")
    print(f"P95:          {sorted_counts[int(len(sorted_counts) * 0.95)]}")

    print()
    print(f"< 50 tokens:  {sum(n < 50 for n in token_counts)}")
    print(f"< 100 tokens: {sum(n < 100 for n in token_counts)}")
    print(f"> 700 tokens: {sum(n > 700 for n in token_counts)}")

    print("\nTiny chunks:")

    for chunk, count in zip(chunks, token_counts):
        if count < 50:
            print(
                f"page={chunk.metadata['page_number']:>3} "
                f"tokens={count:>3} "
                f"text={chunk.page_content[:80]!r}"
            )

if __name__ == "__main__":
    main()