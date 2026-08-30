from langchain_core.documents import Document

from src.ingestion.document_type import (
    annotate_document_types,
    classify_document_type,
)


def test_classifies_table_of_contents():
    text = (
        "Code Review Benefits 170 "
        "Table of Contents | vii "
        "10. Documentation 185"
    )

    assert classify_document_type(text) == "toc"


def test_classifies_index():
    text = (
        "regular expressions 368 "
        "Index | 565 "
        "release branches 339"
    )

    assert classify_document_type(text) == "index"


def test_does_not_mistake_content_for_index():
    text = (
        "The search index is stored on flash. "
        "Reverse index construction must scale."
    )

    assert classify_document_type(text) == "content"


def test_classifies_normal_content():
    text = (
        "Hyrum's Law states that with enough users, "
        "all observable behaviors will be depended on."
    )

    assert classify_document_type(text) == "content"

def test_annotates_document_type_metadata():
    documents = [
        Document(
            page_content="Table of Contents | vii",
            metadata={
                "page_number": 5,
                "source_id": "book-1",
            },
        ),
        Document(
            page_content="Normal chapter content.",
            metadata={
                "page_number": 36,
                "source_id": "book-1",
            },
        ),
    ]

    annotated = annotate_document_types(documents)

    assert annotated[0].metadata["document_type"] == "toc"
    assert annotated[1].metadata["document_type"] == "content"

    assert annotated[0].metadata["page_number"] == 5
    assert annotated[0].metadata["source_id"] == "book-1"