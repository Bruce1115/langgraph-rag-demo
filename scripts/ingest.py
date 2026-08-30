from pathlib import Path

from dotenv import load_dotenv

from src.config import EmbeddingConfig, MilvusConfig
from src.infrastructure.embeddings import BailianEmbedder
from src.infrastructure.milvus import MilvusStore
from src.ingestion.document_type import annotate_document_types
from src.ingestion.indexer import index_documents
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.splitter import ChunkingConfig, split_documents

PDF_PATH = Path("data/source/swe_at_google.2.pdf")
SOURCE_ID = "software-engineering-at-google"


def main() -> None:
    load_dotenv()

    # 1. Load PDF
    documents = load_pdf(
        PDF_PATH,
        source_id=SOURCE_ID,
    )

    print(f"Loaded pages: {len(documents)}")

    # 2. Annotate document structure
    documents = annotate_document_types(
        documents
    )

    # 2. Split into chunks
    chunking_config = ChunkingConfig(
        chunk_size=700,
        chunk_overlap=100,
        encoding_name="cl100k_base",
        version="v1",
    )

    chunks = split_documents(
        documents,
        chunking_config,
    )

    print(f"Generated chunks: {len(chunks)}")

    # 3. Initialize embedding model
    embedding_config = EmbeddingConfig()

    embedder = BailianEmbedder(
        embedding_config,
    )

    # 4. Initialize Milvus
    store = MilvusStore(
        MilvusConfig()
    )

    store.ensure_collection(
        vector_dimension=embedder.dimension
    )

    # 5. Replace old index for this source
    deleted = store.delete_by_source(
        SOURCE_ID
    )

    print(f"Deleted old chunks: {deleted}")

    # 6. Embed + insert
    inserted = index_documents(
        chunks=chunks,
        embedder=embedder,
        store=store,
    )

    # ingestion job 结束时 flush 一次
    store.flush()

    print(f"Inserted chunks: {inserted}")
    print("Ingestion completed.")


if __name__ == "__main__":
    main()