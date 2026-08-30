from langchain_core.messages import ToolMessage

from src.retrieval.retriever import RetrievedChunk
from src.retrieval.tool import create_retrieval_tool


class FakeRetriever:
    def __init__(self) -> None:
        self.received_query: str | None = None
        self.received_limit: int | None = None

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        self.received_query = query
        self.received_limit = limit

        return [
            RetrievedChunk(
                chunk_id="chunk-001",
                source_id="book-001",
                page_number=36,
                chunk_index=0,
                text="Hyrum's Law says observable behavior will be depended on.",
                score=0.88,
            )
        ]


def test_retrieval_tool_returns_content_and_artifact() -> None:
    retriever = FakeRetriever()

    retrieval_tool = create_retrieval_tool(
        retriever,
        top_k=3,
    )

    result = retrieval_tool.invoke(
        {
            "name": "retrieve_book",
            "args": {
                "query": "What is Hyrum's Law?",
            },
            "id": "call-001",
            "type": "tool_call",
        }
    )

    assert retriever.received_query == "What is Hyrum's Law?"
    assert retriever.received_limit == 3

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-001"

    assert "[Page 36, chunk 0]" in result.content
    assert "Hyrum's Law" in result.content

    assert len(result.artifact) == 1
    assert result.artifact[0]["chunk_id"] == "chunk-001"
    assert result.artifact[0]["page_number"] == 36
    assert result.artifact[0]["score"] == 0.88


class EmptyFakeRetriever:
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        return []

def test_retrieval_tool_handles_empty_result() -> None:
    retriever = EmptyFakeRetriever()

    retrieval_tool = create_retrieval_tool(
        retriever,
        top_k=3,
    )

    result = retrieval_tool.invoke(
        {
            "name": "retrieve_book",
            "args": {
                "query": "Something not in the book",
            },
            "id": "call-002",
            "type": "tool_call",
        }
    )

    assert isinstance(result, ToolMessage)
    assert result.content == "No relevant passages found."
    assert result.artifact == []