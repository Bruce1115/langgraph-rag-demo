from unittest.mock import MagicMock

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import MessagesState

from src.agent.nodes import (
    GradeDocuments,
    create_generate_answer,
    create_generate_query_or_respond,
    create_grade_documents,
    create_rewrite_question,
)


def test_generate_query_or_respond_direct_answer() -> None:
    model = MagicMock()
    retrieval_tool = MagicMock()

    bound_model = MagicMock()
    model.bind_tools.return_value = bound_model

    expected_response = AIMessage(
        content="Hello! How can I help?"
    )

    bound_model.invoke.return_value = expected_response

    node = create_generate_query_or_respond(
        model=model,
        retrieval_tool=retrieval_tool,
    )

    state: MessagesState = {
        "messages": [
            HumanMessage(content="Hello")
        ]
    }

    result = node(state)

    model.bind_tools.assert_called_once_with(
        [retrieval_tool]
    )

    bound_model.invoke.assert_called_once_with(
        state["messages"]
    )

    assert result == {
        "messages": [expected_response]
    }


def test_generate_query_or_respond_tool_call() -> None:
    model = MagicMock()
    retrieval_tool = MagicMock()

    bound_model = MagicMock()
    model.bind_tools.return_value = bound_model

    expected_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "retrieve_book",
                "args": {
                    "query": "What is Hyrum's Law?"
                },
                "id": "call-001",
                "type": "tool_call",
            }
        ],
    )

    bound_model.invoke.return_value = expected_response

    node = create_generate_query_or_respond(
        model=model,
        retrieval_tool=retrieval_tool,
    )

    state: MessagesState = {
        "messages": [
            HumanMessage(
                content="What is Hyrum's Law?"
            )
        ]
    }

    result = node(state)

    assert result["messages"][0] == expected_response

    tool_calls = result["messages"][0].tool_calls

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "retrieve_book"
    assert tool_calls[0]["args"]["query"] == "What is Hyrum's Law?"

def test_grade_documents_relevant():
    model = MagicMock()
    grader = MagicMock()

    model.with_structured_output.return_value = grader

    grader.invoke.return_value = GradeDocuments(
        binary_score="yes"
    )

    grade_documents = create_grade_documents(model)

    state: MessagesState = {
        "messages": [
            HumanMessage(
                content="What is Hyrum's Law?"
            ),
            ToolMessage(
                content="Hyrum's Law describes how observable "
                "system behavior becomes depended upon.",
                tool_call_id="call-001",
            ),
        ]
    }

    result = grade_documents(state)

    assert result == "generate_answer"


def test_grade_documents_irrelevant():
    model = MagicMock()
    grader = MagicMock()

    model.with_structured_output.return_value = grader

    grader.invoke.return_value = GradeDocuments(
        binary_score="no"
    )

    grade_documents = create_grade_documents(model)

    state: MessagesState = {
        "messages": [
            HumanMessage(
                content="What is Hyrum's Law?"
            ),
            ToolMessage(
                content="This passage discusses office furniture.",
                tool_call_id="call-001",
            ),
        ]
    }

    result = grade_documents(state)

    assert result == "rewrite_question"

def test_rewrite_question():
    model = MagicMock()

    model.invoke.return_value = AIMessage(
        content=(
            "What is Hyrum's Law and how does it relate "
            "to observable system behavior?"
        )
    )

    rewrite_question = create_rewrite_question(model)

    state: MessagesState = {
        "messages": [
            HumanMessage(
                content="What does Hyrum say?"
            ),
            ToolMessage(
                content="Irrelevant retrieval result.",
                tool_call_id="call-001",
            ),
        ]
    }

    result = rewrite_question(state)

    assert len(result["messages"]) == 1

    rewritten_message = result["messages"][0]

    assert isinstance(
        rewritten_message,
        HumanMessage,
    )

    assert rewritten_message.content == (
        "What is Hyrum's Law and how does it relate "
        "to observable system behavior?"
    )

    model.invoke.assert_called_once()

def test_generate_answer():
    model = MagicMock()

    ai_message = AIMessage(
        content=(
            "Hyrum's Law says that all observable "
            "behaviors of a system will eventually "
            "be depended on by someone."
        )
    )

    model.invoke.return_value = ai_message

    generate_answer = create_generate_answer(model)

    state: MessagesState = {
        "messages": [
            HumanMessage(
                content="What is Hyrum's Law?"
            ),
            ToolMessage(
                content=(
                    "Hyrum's Law states that all "
                    "observable behaviors of your system "
                    "will be depended on by somebody."
                ),
                tool_call_id="call-001",
            ),
        ]
    }

    result = generate_answer(state)

    assert result == {
        "messages": [ai_message]
    }

    model.invoke.assert_called_once()