"""Tests for main.py - MCP server entry point."""
from main import build_components
from src.config import load_config


def test_build_components_with_mock_config():
    """Default config should work with mock provider (no external deps)."""
    config = {
        "embedding": {"provider": "mock", "dimension": 4},
        "vector_store": {"type": "mock"},
        "retrieval": {"top_k": 10, "semantic_weight": 1.0, "keyword_weight": 0.0},
    }
    sync_mgr, retriever, retrieval_cfg = build_components(config)
    assert sync_mgr is not None
    assert retriever is not None
    assert retrieval_cfg["top_k"] == 10


def test_build_components_with_ollama_config():
    """Ollama config should construct without immediate connection attempt."""
    config = {
        "embedding": {
            "provider": "ollama",
            "model": "test-model",
            "base_url": "http://localhost:11434",
            "dimension": 768,
            "batch_size": 8,
        },
        "vector_store": {"type": "mock"},
        "retrieval": {"top_k": 5, "semantic_weight": 0.8, "keyword_weight": 0.2},
    }
    sync_mgr, retriever, retrieval_cfg = build_components(config)
    assert sync_mgr is not None
    assert retriever is not None


import tempfile


def test_build_components_with_milvus_temp_dir():
    """MilvusVectorStore should work when using a unique temp directory.

    Verifies that the eager-connect behavior works when no lock contention
    exists (fresh temp dir each time).
    """
    config = {
        "embedding": {"provider": "mock", "dimension": 4},
        "vector_store": {
            "type": "milvus",
            "milvus": {
                "mode": "local",
                "local_uri": "",  # filled below
                "collection": "test_wiki",
            },
        },
        "retrieval": {"top_k": 10, "semantic_weight": 1.0, "keyword_weight": 0.0},
    }
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        config["vector_store"]["milvus"]["local_uri"] = f"{d}/test.db"
        sync_mgr, retriever, _ = build_components(config)
        assert sync_mgr is not None
        assert retriever is not None
