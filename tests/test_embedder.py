import pytest
from src.embedder import MockEmbedder


def test_mock_embedder_returns_correct_dimensions():
    embedder = MockEmbedder(dimension=128)
    vectors = embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 128
    assert embedder.dimension == 128


def test_mock_embedder_embed_query():
    embedder = MockEmbedder(dimension=64)
    v = embedder.embed_query("test")
    assert len(v) == 64


def test_mock_embedder_deterministic():
    embedder = MockEmbedder(dimension=16, seed=42)
    v1 = embedder.embed_query("hello")
    embedder2 = MockEmbedder(dimension=16, seed=42)
    v2 = embedder2.embed_query("hello")
    assert v1 == v2


def test_ollama_embedder_construction():
    from src.embedder import OllamaEmbedder
    emb = OllamaEmbedder(model="test-model", base_url="http://localhost:11434", dimension=768)
    assert emb.dimension == 768
