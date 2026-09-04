import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from src.infrastructure.llm import create_bailian_chat_model


def main() -> None:
    load_dotenv()

    model = create_bailian_chat_model(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_BASE_URL"],
        model=os.getenv(
            "DASHSCOPE_CHAT_MODEL",
            "qwen-plus",
        ),
    )

    def generate(state: MessagesState) -> dict:
        response = model.invoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    builder = StateGraph(MessagesState)

    builder.add_node(
        "generate",
        generate,
    )

    builder.add_edge(
        START,
        "generate",
    )

    builder.add_edge(
        "generate",
        END,
    )

    graph = builder.compile()

    for mode, chunk in graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Explain Hyrum's Law in two sentences."
                )
            ]
        },
        stream_mode=["updates", "messages"],
    ):
        print(mode, chunk)

    print()


if __name__ == "__main__":
    main()