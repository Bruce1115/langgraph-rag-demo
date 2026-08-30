from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from src.retrieval.retriever import RetrievedChunk
from src.retrieval.tool import create_retrieval_tool


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        assert query == "What is Hyrum's Law?"
        assert limit == 5

        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                source_id="book-1",
                page_number=36,
                chunk_index=0,
                text="Hyrum's Law states that all observable behaviors "
                "of your system will be depended on by somebody.",
                score=0.588,
            )
        ]


def test_tool_node_executes_retrieval_tool():
    retriever = FakeRetriever()

    retrieval_tool = create_retrieval_tool(
        retriever,
        top_k=5,
    )

    tool_node = ToolNode([retrieval_tool])

    state: MessagesState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_book",
                        "args": {
                            "query": "What is Hyrum's Law?"
                        },
                        "id": "call-001",
                    }
                ],
            )
        ]
    }

    result = tool_node.invoke(
        state,
        runtime=Runtime(),
    )

    tool_message = result["messages"][0]

    assert isinstance(tool_message, ToolMessage)
    assert tool_message.name == "retrieve_book"
    assert tool_message.tool_call_id == "call-001"

    assert "Page 36" in tool_message.content
    assert "Hyrum's Law" in tool_message.content

    assert len(tool_message.artifact) == 1
    assert tool_message.artifact[0]["chunk_id"] == "chunk-1"
    assert tool_message.artifact[0]["page_number"] == 36