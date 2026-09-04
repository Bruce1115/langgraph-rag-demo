import argparse
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class HitlState(TypedDict):
    task: str
    approved: bool


def human_review(state: HitlState) -> dict:
    print(">>> human_review started")

    approved = interrupt(
        {
            "question": "Do you approve this task?",
            "task": state["task"],
        }
    )

    print(">>> human_review resumed")

    return {
        "approved": bool(approved),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        choices=["y", "n"],
    )
    args = parser.parse_args()

    load_dotenv()

    db_uri = os.environ["LANGGRAPH_CHECKPOINT_DB_URI"]

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "hitl-durable-1",
        }
    }

    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        checkpointer.setup()

        builder = StateGraph(HitlState)  # ty: ignore[invalid-argument-type]

        builder.add_node(
            "human_review",
            human_review,
        )

        builder.add_edge(
            START,
            "human_review",
        )

        builder.add_edge(
            "human_review",
            END,
        )

        graph = builder.compile(
            checkpointer=checkpointer
        )

        if args.resume is None:
            result = graph.invoke(
                {
                    "task": "Deploy version 1.0",
                    "approved": False,
                },
                config=config,
            )

            print("\nGraph paused:")
            print(result["__interrupt__"])

            print("\nProcess will exit now.")

        else:
            approved = args.resume == "y"

            result = graph.invoke(
                Command(resume=approved),
                config=config,
            )

            print("\nFinal state:")
            print(result)


if __name__ == "__main__":
    main()