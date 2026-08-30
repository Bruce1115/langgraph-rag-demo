from dataclasses import asdict

from langchain_core.tools import BaseTool, tool

from src.retrieval.retriever import Retriever


def create_retrieval_tool(
    retriever: Retriever,
    *,
    top_k: int = 5,
) -> BaseTool:

    @tool(
        "retrieve_book",
        response_format="content_and_artifact",
    )
    def retrieve_book(
        query: str,
    ) -> tuple[str, list[dict[str, object]]]:
        """Search the Software Engineering at Google book for passages relevant to the query."""

        chunks = retriever.retrieve(
            query,
            limit=top_k,
        )

        if not chunks:
            return "No relevant passages found.", []

        content = "\n\n".join(
            (
                f"[Page {chunk.page_number}, chunk {chunk.chunk_index}]\n"
                f"{chunk.text}"
            )
            for chunk in chunks
        )

        artifact = [
            asdict(chunk)
            for chunk in chunks
        ]

        return content, artifact

    return retrieve_book