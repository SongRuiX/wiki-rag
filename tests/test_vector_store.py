from src.vector_store import MockVectorStore
from src.models import Chunk


def make_chunk(sn: str, title: str, content: str, embedding: list[float] | None = None) -> Chunk:
    return Chunk(source_sn=sn, source_title=title, content=content, embedding=embedding or [0.1, 0.2])


class TestMockVectorStore:
    def test_add_and_search(self):
        store = MockVectorStore()
        c = make_chunk("doc", "Title", "hello world")
        store.add_chunks([c])
        results = store.hybrid_search([0.1, 0.2], "hello", top_k=5, semantic_weight=1.0, keyword_weight=0.0)
        assert len(results) == 1
        assert results[0].chunk.source_sn == "doc"

    def test_delete_by_sn(self):
        store = MockVectorStore()
        store.add_chunks([make_chunk("a", "A", "content"), make_chunk("b", "B", "content")])
        store.delete_by_sn(["a"])
        results = store.hybrid_search([0.1, 0.2], "content", top_k=10, semantic_weight=1.0, keyword_weight=0.0)
        assert len(results) == 1
        assert results[0].chunk.source_sn == "b"

    def test_search_limits_top_k(self):
        store = MockVectorStore()
        chunks = [make_chunk(f"doc{i}", f"T{i}", f"c{i}") for i in range(10)]
        store.add_chunks(chunks)
        results = store.hybrid_search([0.1, 0.2], "query", top_k=3, semantic_weight=1.0, keyword_weight=0.0)
        assert len(results) == 3

    def test_search_results_have_rrf_score(self):
        store = MockVectorStore()
        store.add_chunks([make_chunk("x", "X", "data")])
        results = store.hybrid_search([0.5, 0.5], "data", top_k=5, semantic_weight=1.0, keyword_weight=0.0)
        assert results[0].rrf_score > 0
        assert results[0].rrf_semantic_rank >= 0
