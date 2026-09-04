from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph


def update_message(state: MessagesState) -> dict:
    return {
        "messages": [
            HumanMessage(
                id="message-1",
                content="Updated message",
            )
        ]
    }


def main() -> None:
    builder = StateGraph(MessagesState)

    builder.add_node(
        "update_message",
        update_message,
    )

    builder.add_edge(START, "update_message")
    builder.add_edge("update_message", END)

    graph = builder.compile()

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    id="message-0",
                    content="Original message",
                )
            ]
        }
    )

    for message in result["messages"]:
        print(message.content)


if __name__ == "__main__":
    main()