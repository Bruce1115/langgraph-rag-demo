from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class DemoState(TypedDict):
    value: int


def add_one(state: DemoState) -> dict:
    return {
        "value": state["value"] + 1,
    }


def double(state: DemoState) -> dict:
    return {
        "value": state["value"] * 2,
    }


def main() -> None:
    builder = StateGraph(DemoState)

    builder.add_node("add_one", add_one)
    builder.add_node("double", double)

    builder.add_edge(START, "add_one")
    builder.add_edge("add_one", "double")
    builder.add_edge("double", END)

    graph = builder.compile()

    for chunk in graph.stream(
        {"value": 3},
        stream_mode="values",
    ):
        print(chunk)


if __name__ == "__main__":
    main()