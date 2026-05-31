from src.retriever import Retriever
from src.vector_store import MockVectorStore
from src.embedder import MockEmbedder
from src.models import Chunk


def test_retriever_search_returns_results():
    store = MockVectorStore()
    embedder = MockEmbedder(dimension=4, seed=1)
    chunks = [
        Chunk(source_sn="a", source_title="A", content="hello", embedding=embedder.embed_query("hello")),
        Chunk(source_sn="b", source_title="B", content="world", embedding=embedder.embed_query("world")),
    ]
    store.add_chunks(chunks)
    retriever = Retriever(store, embedder)
    results = retriever.search("hello", top_k=5)
    assert len(results) == 2
    assert results[0].chunk.source_sn in ("a", "b")


def test_retriever_respects_top_k():
    store = MockVectorStore()
    embedder = MockEmbedder(dimension=4, seed=1)
    chunks = [
        Chunk(source_sn=f"doc{i}", source_title=f"T{i}", content=f"c{i}",
              embedding=embedder.embed_query(f"doc{i}"))
        for i in range(10)
    ]
    store.add_chunks(chunks)
    retriever = Retriever(store, embedder)
    results = retriever.search("query", top_k=3)
    assert len(results) == 3


def test_retriever_empty_store_returns_empty():
    store = MockVectorStore()
    embedder = MockEmbedder(dimension=4, seed=1)
    retriever = Retriever(store, embedder)
    results = retriever.search("nothing", top_k=5)
    assert results == []
