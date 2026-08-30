import os
from typing import Protocol

from langchain_openai import OpenAIEmbeddings

from src.config import EmbeddingConfig


class Embedder(Protocol):
    @property
    def dimension(self) -> int:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class BailianEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

        self._client = OpenAIEmbeddings(
            model=config.model,
            dimensions=config.dimensions,
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["DASHSCOPE_BASE_URL"],

            # 百炼 OpenAI-compatible API 单次最多 10 条
            chunk_size=config.batch_size,

            # 非 OpenAI provider，避免 LangChain 使用
            # OpenAI 自己的 tokenizer 预处理输入
            check_embedding_ctx_length=False,
        )

    @property
    def dimension(self) -> int:
        return self._config.dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)