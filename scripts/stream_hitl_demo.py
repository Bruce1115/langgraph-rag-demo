import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from src.infrastructure.llm import create_bailian_chat_model


async def main() -> None:
    load_dotenv()

    model = create_bailian_chat_model(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_BASE_URL"],
        model=os.getenv(
            "DASHSCOPE_CHAT_MODEL",
            "qwen-plus",
        ),
    )

    async def generate(state: MessagesState) -> dict:
        response = await model.ainvoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    builder = StateGraph(MessagesState)

    builder.add_node("generate", generate)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", END)

    graph = builder.compile()

    async for message, metadata in graph.astream(
        {
            "messages": [
                HumanMessage(
                    content="Explain Hyrum's Law in two sentences."
                )
            ]
        },
        stream_mode="messages",
    ):
        if message.content:
            print(
                message.content,
                end="",
                flush=True,
            )

    print()


if __name__ == "__main__":
    asyncio.run(main())