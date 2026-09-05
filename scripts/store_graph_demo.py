import os
from dataclasses import dataclass
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.postgres import PostgresStore


class MemoryState(TypedDict):
    action: Literal["write", "read"]
    result: str


@dataclass
class Context:
    user_id: str


def memory_node(
    state: MemoryState,
    runtime: Runtime[Context],
) -> dict:
    assert runtime.store is not None

    namespace = (
        "users",
        runtime.context.user_id,
        "preferences",
    )

    if state["action"] == "write":
        runtime.store.put(
            namespace,
            "response-style",
            {
                "preference": "concise technical answers",
            },
        )

        return {
            "result": "Memory saved",
        }

    item = runtime.store.get(
        namespace,
        "response-style",
    )

    if item is None:
        return {
            "result": "Memory not found",
        }

    return {
        "result": item.value["preference"],
    }


def main() -> None:
    load_dotenv()

    db_uri = os.environ["LANGGRAPH_CHECKPOINT_DB_URI"]

    with (
        PostgresStore.from_conn_string(db_uri) as store,
        PostgresSaver.from_conn_string(db_uri) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()

        builder = StateGraph(
            MemoryState,
            context_schema=Context,
        )

        builder.add_node(
            "memory",
            memory_node,
        )

        builder.add_edge(
            START,
            "memory",
        )

        builder.add_edge(
            "memory",
            END,
        )

        graph = builder.compile(
            checkpointer=checkpointer,
            store=store,
        )

        result_a = graph.invoke(
            {
                "action": "write",
                "result": "",
            },
            config={
                "configurable": {
                    "thread_id": "thread-A",
                }
            },
            context=Context(
                user_id="demo-user",
            ),
        )

        print("Thread A:")
        print(result_a["result"])

        result_b = graph.invoke(
            {
                "action": "read",
                "result": "",
            },
            config={
                "configurable": {
                    "thread_id": "thread-B",
                }
            },
            context=Context(
                user_id="demo-user",
            ),
        )

        print("\nThread B:")
        print(result_b["result"])


if __name__ == "__main__":
    main()