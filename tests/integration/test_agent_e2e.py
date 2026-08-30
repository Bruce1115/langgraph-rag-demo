import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

from src.agent.graph import create_graph
from src.config import EmbeddingConfig, MilvusConfig
from src.infrastructure.embeddings import BailianEmbedder
from src.infrastructure.llm import create_bailian_chat_model
from src.infrastructure.milvus import MilvusStore
from src.retrieval.retriever import MilvusRetriever
from src.retrieval.tool import create_retrieval_tool


def test_agent_end_to_end():
    load_dotenv()

    model = create_bailian_chat_model(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_BASE_URL"],
        model=os.getenv(
            "DASHSCOPE_CHAT_MODEL",
            "qwen-plus",
        ),
    )

    embedder = BailianEmbedder(
        EmbeddingConfig()
    )

    store = MilvusStore(
        MilvusConfig()
    )

    retriever = MilvusRetriever(
        embedder=embedder,
        store=store,
    )

    retrieval_tool = create_retrieval_tool(
        retriever,
        top_k=5,
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

    final_message = result["messages"][-1]

    assert final_message.content