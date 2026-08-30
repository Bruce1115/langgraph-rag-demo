from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def calculate_file_hash(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_pdf(
    path: Path,
    *,
    source_id: str,
) -> list[Document]:
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    document_version = calculate_file_hash(path)
    reader = PdfReader(path)

    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source_id": source_id,
                    "source_path": str(path),
                    "source_type": "pdf",
                    "document_version": document_version,
                    "page_index": page_index,
                    "page_number": page_index + 1,
                },
            )
        )

    return documents