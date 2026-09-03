from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.nodes import (
    create_generate_answer,
    create_generate_query_or_respond,
    create_grade_documents,
    create_rewrite_question,
)


def create_graph(
    model: BaseChatModel,
    retrieval_tool: BaseTool,
    checkpointer: BaseCheckpointSaver | None = None,
):
    generate_query_or_respond = create_generate_query_or_respond(
        model,
        retrieval_tool,
    )

    grade_documents = create_grade_documents(model)
    rewrite_question = create_rewrite_question(model)
    generate_answer = create_generate_answer(model)

    builder = StateGraph(MessagesState)  # ty: ignore[invalid-argument-type]

    builder.add_node(
        "generate_query_or_respond",
        generate_query_or_respond,
    )

    builder.add_node(
        "retrieve",
        ToolNode([retrieval_tool]),
    )
    
    builder.add_node(
        "rewrite_question",
        rewrite_question,
    )

    builder.add_node(
        "generate_answer",
        generate_answer,
    )

    builder.add_edge(
        START,
        "generate_query_or_respond",
    )

    builder.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        },
    )

    builder.add_edge(
        "rewrite_question",
        "generate_query_or_respond",
    )

    builder.add_edge(
        "generate_answer",
        END,
    )

    return builder.compile(checkpointer=checkpointer)