import argparse
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

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

    checkpointer = InMemorySaver()

    graph = create_graph(
        model,
        retrieval_tool,
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {
            "thread_id": "demo-1",
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
            ]
        },
        config=config,
    )

    # for message in result["messages"]:
    #     if isinstance(message, ToolMessage):
    #         print("\nRetrieved passages:")

    #         for rank, chunk in enumerate(
    #             message.artifact,
    #             start=1,
    #         ):
    #             print(
    #                 f"\n{rank}. "
    #                 f"page={chunk['page_number']} "
    #                 f"chunk={chunk['chunk_index']} "
    #                 f"score={chunk['score']:.4f}"
    #             )

    #             text = chunk["text"]

    #             if len(text) > 500:
    #                 text = text[:500] + "..."

    #             print(text)

    print("\nAnswer:")
    print(result["messages"][-1].content)

    snapshot = graph.get_state(config)

    print("\nCheckpoint state:")
    print(snapshot)


    history = list(graph.get_state_history(config))

    print("\nCheckpoint history:")

    for index, snapshot in enumerate(history, start=1):
        print(
            f"{index}. "
            f"step={snapshot.metadata.get('step')} "
            f"next={snapshot.next} "
            f"checkpoint_id="
            f"{snapshot.config['configurable']['checkpoint_id']}"
        )


    result2 = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Why is it important?"
                )
            ]
        },
        config=config,
    )

    print("\nSecond answer:")
    print(result2["messages"][-1].content)

    print("\nMessage count:")
    print(len(result2["messages"]))

if __name__ == "__main__":
    main()