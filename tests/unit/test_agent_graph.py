from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.graph import create_graph
from src.agent.nodes import GradeDocuments
from src.retrieval.retriever import RetrievedChunk
from src.retrieval.tool import create_retrieval_tool


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
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


def test_graph_direct_answer():
    model = MagicMock()
    bound_model = MagicMock()

    model.bind_tools.return_value = bound_model

    bound_model.invoke.return_value = AIMessage(
        content="Hello! How can I help?"
    )

    retrieval_tool = create_retrieval_tool(
        FakeRetriever()
    )

    graph = create_graph(
        model,
        retrieval_tool,
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Hello")
            ]
        }
    )

    assert len(result["messages"]) == 2

    assert isinstance(
        result["messages"][1],
        AIMessage,
    )

    assert result["messages"][1].content == (
        "Hello! How can I help?"
    )


def test_graph_generates_answer_for_relevant_retrieval():
    model = MagicMock()
    bound_model = MagicMock()

    model.bind_tools.return_value = bound_model
    bound_model.invoke.return_value = AIMessage(
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


    grader = MagicMock()
    model.with_structured_output.return_value = grader
    grader.invoke.return_value = GradeDocuments(
        binary_score="yes"
    )   

    model.invoke.return_value = AIMessage(
        content=(
            "Hyrum's Law says that all observable "
            "behaviors of a system will eventually "
            "be depended on by someone."
        )
    )

    retrieval_tool = create_retrieval_tool(
        FakeRetriever()
    )

    graph = create_graph(
        model,
        retrieval_tool,
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is Hyrum's Law?"
                )
            ]
        }
    )

    assert len(result["messages"]) == 4

    assert isinstance(result["messages"][1], AIMessage)
    assert isinstance(result["messages"][2], ToolMessage)
    assert isinstance(result["messages"][3], AIMessage)

    assert "observable" in result["messages"][3].content

    grader.invoke.assert_called_once()
    model.invoke.assert_called_once()

def test_graph_rewrites_irrelevant_query():
    model = MagicMock()
    bound_model = MagicMock()
    grader = MagicMock()

    model.bind_tools.return_value = bound_model
    model.with_structured_output.return_value = grader

    bound_model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_book",
                    "args": {
                        "query": "What does Hyrum say?"
                    },
                    "id": "call-001",
                }
            ],
        ),
        AIMessage(
            content="I can answer directly now."
        ),
    ]

    grader.invoke.return_value = GradeDocuments(
        binary_score="no"
    )

    model.invoke.return_value = AIMessage(
        content=(
            "What is Hyrum's Law and how does it relate "
            "to observable system behavior?"
        )
    )

    retrieval_tool = create_retrieval_tool(
        FakeRetriever()
    )

    graph = create_graph(
        model,
        retrieval_tool,
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What does Hyrum say?"
                )
            ]
        }
    )

    assert len(result["messages"]) == 5

    assert isinstance(result["messages"][1], AIMessage)
    assert isinstance(result["messages"][2], ToolMessage)
    assert isinstance(result["messages"][3], HumanMessage)
    assert isinstance(result["messages"][4], AIMessage)

    assert result["messages"][3].content == (
        "What is Hyrum's Law and how does it relate "
        "to observable system behavior?"
    )

    assert bound_model.invoke.call_count == 2
    model.invoke.assert_called_once()