import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about Software Engineering at Google."
    )

    parser.add_argument(
        "question",
        help="Question to ask the RAG agent.",
    )

    args = parser.parse_args()

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
                HumanMessage(content=args.question)
            ]
        },
        config={
            "recursion_limit": 10,
        },
    )

    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            print("\nRetrieved passages:")

            for rank, chunk in enumerate(
                message.artifact,
                start=1,
            ):
                print(
                    f"\n{rank}. "
                    f"page={chunk['page_number']} "
                    f"chunk={chunk['chunk_index']} "
                    f"score={chunk['score']:.4f}"
                )

                text = chunk["text"]

                if len(text) > 500:
                    text = text[:500] + "..."

                print(text)

    print("\nAnswer:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()