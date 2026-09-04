from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src.agent.state import AgentState


def update_state(state: AgentState) -> dict:
    return {
        "messages": [
            AIMessage(content="Second message")
        ],
        "current_question": "New question",
    }


def main() -> None:
    builder = StateGraph(AgentState)

    builder.add_node(
        "update_state",
        update_state,
    )

    builder.add_edge(
        START,
        "update_state",
    )

    builder.add_edge(
        "update_state",
        END,
    )

    graph = builder.compile()

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="First message")
            ],
            "current_question": "Old question",
        }
    )

    print("Messages:")
    for message in result["messages"]:
        print(message.content)

    print("\nCurrent question:")
    print(result["current_question"])


if __name__ == "__main__":
    main()