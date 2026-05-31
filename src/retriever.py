from src.vector_store import VectorStore
from src.embedder import Embedder
from src.models import SearchResult


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 10, semantic_weight: float = 1.0, keyword_weight: float = 0.0) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(query)
        return self.vector_store.hybrid_search(
            query_vector=query_vector,
            query_text=query,
            top_k=top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )
