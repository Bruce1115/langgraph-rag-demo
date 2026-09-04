# scripts/hitl_demo.py

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class HitlState(TypedDict):
    task: str
    approved: bool


def human_review(
    state: HitlState,
) -> Command[Literal["proceed", "cancel"]]:
    review = interrupt(
        {
            "question": "Review this task",
            "task": state["task"],
        }
    )

    if review["action"] == "reject":
        return Command(  # ty: ignore[invalid-return-type]
            goto="cancel",
        )

    return Command(  # ty: ignore[invalid-return-type]
        update={
            "task": review["task"],
            "approved": True,
        },
        goto="proceed",
    )

def proceed(state: HitlState) -> dict:
    print(">>> Task approved. Proceeding...")
    return {}


def cancel(state: HitlState) -> dict:
    print(">>> Task rejected. Cancelling...")
    return {}


def main() -> None:
    load_dotenv()

    db_uri = os.environ["LANGGRAPH_CHECKPOINT_DB_URI"]

    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        checkpointer.setup()

        builder = StateGraph(HitlState)  # ty: ignore[invalid-argument-type]

        builder.add_node(
            "human_review",
            human_review,
        )

        builder.add_node(
            "proceed",
            proceed,
        )

        builder.add_node(
            "cancel",
            cancel,
        )

        builder.add_edge(
            START,
            "human_review",
        )

        builder.add_edge(
            "proceed",
            END,
        )

        builder.add_edge(
            "cancel",
            END,
        )

        graph = builder.compile(
            checkpointer=checkpointer
        )

        config: RunnableConfig = {
            "configurable": {
                "thread_id": "hitl-basic-1",
            }
        }

        result = graph.invoke(
            {
                "task": "Deploy version 1.0",
                "approved": False,
            },
            config=config,
        )

        print("\nInterrupt:")
        print(result["__interrupt__"])

        answer = input(
            "\nApprove, reject, or edit? [a/r/e]: "
        ).strip().lower()

        if answer == "e":
            edited_task = input("Edited task: ")

            resume_value = {
                "action": "approve",
                "task": edited_task,
            }

        elif answer == "a":
            resume_value = {
                "action": "approve",
                "task": "Deploy version 1.0",
            }

        else:
            resume_value = {
                "action": "reject",
                "task": "Deploy version 1.0",
            }

        result = graph.invoke(
            Command(resume=resume_value),
            config=config,
        )

        print("\nFinal state:")
        print(result)


if __name__ == "__main__":
    main()