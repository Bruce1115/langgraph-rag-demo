from dotenv import load_dotenv

from src.config import EmbeddingConfig
from src.infrastructure.embeddings import BailianEmbedder

load_dotenv()


def test_bailian_embedding() -> None:
    embedder = BailianEmbedder(
        EmbeddingConfig()
    )

    vector = embedder.embed_query(
        "Software engineering is programming integrated over time."
    )

    assert len(vector) == embedder.dimension
    assert len(vector) == 1024