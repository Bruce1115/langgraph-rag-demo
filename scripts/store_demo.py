import os

from dotenv import load_dotenv
from langgraph.store.postgres import PostgresStore


def main() -> None:
    load_dotenv()

    db_uri = os.environ["LANGGRAPH_CHECKPOINT_DB_URI"]

    with PostgresStore.from_conn_string(db_uri) as store:
        store.setup()

        namespace = (
            "users",
            "demo-user",
            "memories",
        )

        store.put(
            namespace,
            "response-style",
            {
                "preference": "concise technical answers",
            },
        )

        item = store.get(
            namespace,
            "response-style",
        )

        print("Get:")
        print(item)

        memories = store.search(namespace)

        print("\nSearch:")
        for memory in memories:
            print(memory.value)


if __name__ == "__main__":
    main()