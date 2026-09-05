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


def build_child_graph():
    builder = StateGraph(DemoState)

    builder.add_node(
        "add_one",
        add_one,
    )

    builder.add_node(
        "double",
        double,
    )

    builder.add_edge(
        START,
        "add_one",
    )

    builder.add_edge(
        "add_one",
        "double",
    )

    builder.add_edge(
        "double",
        END,
    )

    return builder.compile()


def main() -> None:
    child_graph = build_child_graph()

    parent = StateGraph(DemoState)

    parent.add_node(
        "child_graph",
        child_graph,
    )

    parent.add_edge(
        START,
        "child_graph",
    )

    parent.add_edge(
        "child_graph",
        END,
    )

    graph = parent.compile()

    result = graph.invoke(
        {
            "value": 3,
        }
    )

    print(result)


if __name__ == "__main__":
    main()