import re
from collections.abc import Sequence
from typing import Literal

from langchain_core.documents import Document

DocumentType = Literal[
    "content",
    "toc",
    "index",
]


def classify_document_type(
    text: str,
) -> DocumentType:
    normalized = " ".join(text.split())

    if "Table of Contents" in normalized:
        return "toc"

    if re.search(
        r"(?:Index\s*\|\s*\d+|\d+\s*\|\s*Index)",
        normalized,
    ):
        return "index"

    return "content"


def annotate_document_types(
    documents: Sequence[Document],
) -> list[Document]:
    return [
        Document(
            page_content=document.page_content,
            metadata={
                **document.metadata,
                "document_type": classify_document_type(
                    document.page_content
                ),
            },
        )
        for document in documents
    ]