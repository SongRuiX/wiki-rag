import tempfile
import pytest
from src.vector_store import MilvusVectorStore
from src.models import Chunk


@pytest.fixture
def milvus_store():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        store = MilvusVectorStore(
            uri=f"{d}/test.db",
            collection="test_wiki",
            dimension=4,
        )
        yield store
        store.close()


def make_chunk(sn: str, content: str) -> Chunk:
    return Chunk(
        source_sn=sn,
        source_title="Test",
        content=content,
        embedding=[0.5, 0.5, 0.5, 0.5],
    )


class TestMilvusVectorStore:
    def test_add_and_search(self, milvus_store):
        milvus_store.add_chunks([make_chunk("a", "hello world")])
        results = milvus_store.hybrid_search(
            [0.5, 0.5, 0.5, 0.5], "hello", top_k=5,
            semantic_weight=1.0, keyword_weight=0.0,
        )
        assert len(results) == 1
        assert results[0].chunk.source_sn == "a"

    def test_delete_by_sn(self, milvus_store):
        milvus_store.add_chunks([make_chunk("del-me", "x"), make_chunk("keep", "y")])
        n = milvus_store.delete_by_sn(["del-me"])
        assert n == 1
        results = milvus_store.hybrid_search(
            [0.5, 0.5, 0.5, 0.5], "query", top_k=10,
            semantic_weight=1.0, keyword_weight=0.0,
        )
        sns = {r.chunk.source_sn for r in results}
        assert "del-me" not in sns
        assert "keep" in sns

    def test_add_duplicate_overwrites(self, milvus_store):
        c1 = make_chunk("dup", "first version")
        c2 = make_chunk("dup", "second version")
        milvus_store.add_chunks([c1])
        milvus_store.delete_by_sn(["dup"])
        milvus_store.add_chunks([c2])
        results = milvus_store.hybrid_search(
            [0.5, 0.5, 0.5, 0.5], "second", top_k=5,
            semantic_weight=1.0, keyword_weight=0.0,
        )
        assert len(results) == 1
