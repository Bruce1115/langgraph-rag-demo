import argparse
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver

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

    embedding_config = EmbeddingConfig()
    milvus_config = MilvusConfig()

    top_k = 5
    document_type = "content"

    embedder = BailianEmbedder(
        embedding_config
    )

    store = MilvusStore(
        milvus_config
    )

    retriever = MilvusRetriever(
        embedder=embedder,  # ty: ignore[invalid-argument-type]
        store=store,  # ty: ignore[invalid-argument-type]
        document_type=document_type,
    )

    retrieval_tool = create_retrieval_tool(
        retriever,
        top_k=top_k,
    )

    checkpoint_db_uri = os.environ[
        "LANGGRAPH_CHECKPOINT_DB_URI"
    ]

    with PostgresSaver.from_conn_string(
        checkpoint_db_uri
    ) as checkpointer:
        checkpointer.setup()

        graph = create_graph(
            model,
            retrieval_tool,
            checkpointer=checkpointer,
        )

        config = {
            "configurable": {
                "thread_id": "postgres-persistence-test-1",
            },
            "recursion_limit": 10,
            "tags": [
                "rag",
                "cli",
            ],
            "metadata": {
                "retriever": "dense",
                "top_k": top_k,
                "document_type": document_type,
                "embedding_model": embedding_config.model,
                "embedding_dimensions": embedding_config.dimensions,
                "vector_metric": milvus_config.metric_type,
            },
        }

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=args.question)
                ],
                "current_question": args.question,
            },
            config=config,
        )

 
        print("\nAnswer:")
        print(result["messages"][-1].content)

        print("\nMessage count:")
        print(len(result["messages"]))

        print("\nCurrent question:")
        print(result["current_question"])

if __name__ == "__main__":
    main()