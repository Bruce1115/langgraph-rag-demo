from src.retrieval.retriever import MilvusRetriever


class FakeEmbedder:
    def __init__(self) -> None:
        self.received_query: str | None = None

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        self.received_query = text
        return [0.1, 0.2, 0.3]


class FakeStore:
    def __init__(self) -> None:
        self.received_vector: list[float] | None = None
        self.received_limit: int | None = None

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        document_type: str | None = None,
    ) -> list[dict]:
        self.received_vector = query_vector
        self.received_limit = limit
        self.received_document_type = document_type

        return [
            {
                "distance": 0.88,
                "entity": {
                    "chunk_id": "chunk-001",
                    "source_id": "book-001",
                    "page_number": 36,
                    "chunk_index": 0,
                    "text": "Hyrum's Law ...",
                },
            }
        ]


def test_retrieve_maps_search_result() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()

    retriever = MilvusRetriever(
        embedder=embedder,
        store=store,
    )

    results = retriever.retrieve(
        "What is Hyrum's Law?",
        limit=3,
    )

    assert embedder.received_query == "What is Hyrum's Law?"
    assert store.received_vector == [0.1, 0.2, 0.3]
    assert store.received_limit == 3

    assert len(results) == 1

    chunk = results[0]

    assert chunk.chunk_id == "chunk-001"
    assert chunk.source_id == "book-001"
    assert chunk.page_number == 36
    assert chunk.chunk_index == 0
    assert chunk.text == "Hyrum's Law ..."
    assert chunk.score == 0.88