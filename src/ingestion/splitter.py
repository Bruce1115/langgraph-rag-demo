from dataclasses import dataclass
from hashlib import sha256

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 700
    chunk_overlap: int = 100
    encoding_name: str = "cl100k_base"
    version: str = "v1"


def split_documents(
    documents: list[Document],
    config: ChunkingConfig,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.encoding_name,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    chunks: list[Document] = []

    for document in documents:
        page_chunks = splitter.split_documents([document])

        for chunk_index, chunk in enumerate(page_chunks):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = _build_chunk_id(
                chunk,
                chunk_index,
                config.version,
            )
            chunk.metadata["chunking_version"] = config.version

            chunks.append(chunk)

    return chunks


def _build_chunk_id(
    chunk: Document,
    chunk_index: int,
    chunking_version: str,
) -> str:
    metadata = chunk.metadata

    identity = "|".join(
        [
            metadata["source_id"],
            metadata["document_version"],
            chunking_version,
            str(metadata["page_index"]),
            str(chunk_index),
            chunk.page_content,
        ]
    )

    return sha256(identity.encode("utf-8")).hexdigest()