from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class ParentState(TypedDict):
    question: str
    answer: str


class ChildState(TypedDict):
    query: str
    result: str


def process_query(state: ChildState) -> dict:
    return {
        "result": f"Processed: {state['query']}",
    }


def build_child_graph():
    builder = StateGraph(ChildState)

    builder.add_node(
        "process_query",
        process_query,
    )

    builder.add_edge(
        START,
        "process_query",
    )

    builder.add_edge(
        "process_query",
        END,
    )

    return builder.compile()


def main() -> None:
    child_graph = build_child_graph()

    def run_child(state: ParentState) -> dict:
        child_result = child_graph.invoke(
            {
                "query": state["question"],
                "result": "",
            }
        )

        return {
            "answer": child_result["result"],
        }

    parent = StateGraph(ParentState)

    parent.add_node(
        "child",
        run_child,
    )

    parent.add_edge(
        START,
        "child",
    )

    parent.add_edge(
        "child",
        END,
    )

    graph = parent.compile()

    result = graph.invoke(
        {
            "question": "What is Hyrum's Law?",
            "answer": "",
        }
    )

    print(result)


if __name__ == "__main__":
    main()