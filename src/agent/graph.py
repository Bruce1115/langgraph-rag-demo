from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.nodes import (
    cancel_tool_call,
    create_generate_answer,
    create_generate_query_or_respond,
    create_grade_documents,
    create_rewrite_question,
    create_summarize_history,
    review_tool_call,
    should_summarize,
)
from src.agent.state import AgentState


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
    summarize_history = create_summarize_history(model)

    builder = StateGraph(AgentState)  

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

    builder.add_node(
        "review_tool_call",
        review_tool_call,
    )

    builder.add_node(
        "cancel_tool_call",
        cancel_tool_call,
    )

    builder.add_node(
        "summarize_history",
        summarize_history,
    )

    builder.add_conditional_edges(
        START,
        should_summarize,
        {
            "summarize_history": "summarize_history",
            "generate_query_or_respond": "generate_query_or_respond",
        },
    )

    builder.add_edge(
        "summarize_history",
        "generate_query_or_respond",
    )

    builder.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "review_tool_call",
            END: END,
        },
    )

    builder.add_edge(
        "cancel_tool_call",
        END,
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