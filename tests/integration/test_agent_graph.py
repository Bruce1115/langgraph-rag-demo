import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

from src.agent.graph import create_graph
from src.infrastructure.llm import create_bailian_chat_model
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
                text=(
                    "Hyrum's Law states that with a sufficient "
                    "number of users of an API, all observable "
                    "behaviors of your system will be depended "
                    "on by somebody."
                ),
                score=0.9,
            )
        ]


def test_real_llm_runs_agent_graph():
    load_dotenv()

    model = create_bailian_chat_model(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_BASE_URL"],
        model=os.getenv(
            "DASHSCOPE_CHAT_MODEL",
            "qwen-plus",
        ),
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
                    content=(
                        "Use the retrieve_book tool to answer: "
                        "What is Hyrum's Law?"
                    )
                )
            ]
        },
        config={
            "recursion_limit": 10,
        },
    )

    assert any(
        isinstance(message, ToolMessage)
        for message in result["messages"]
    )

    assert len(result["messages"]) == 4
    assert result["messages"][-1].content