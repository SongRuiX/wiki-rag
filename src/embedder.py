from abc import ABC, abstractmethod
import numpy as np
import requests
from openai import OpenAI


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class MockEmbedder(Embedder):
    def __init__(self, dimension: int = 2560, seed: int | None = None):
        self._dimension = dimension
        self._rng = np.random.RandomState(seed)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._rng.randn(len(texts), self._dimension).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / norms).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OllamaEmbedder(Embedder):
    def __init__(self, model: str, base_url: str, dimension: int, batch_size: int = 16):
        self._dimension = dimension
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=60,
            )
            resp.raise_for_status()
            all_vectors.extend(resp.json()["embeddings"])
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, dimension: int = 1536):
        self._dimension = dimension
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
