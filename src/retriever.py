import logging
import time

from src.vector_store import VectorStore
from src.embedder import Embedder
from src.models import SearchResult

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 10, semantic_weight: float = 1.0, keyword_weight: float = 0.0) -> list[SearchResult]:
        logger.info("检索请求: query=%s, top_k=%d", query[:100], top_k)
        start = time.perf_counter()
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.hybrid_search(
            query_vector=query_vector,
            query_text=query,
            top_k=top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("检索完成: 返回 %d 条结果, elapsed=%.1fms", len(results), elapsed)
        return results
