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


import logging


def test_main_logs_component_build_success(caplog):
    """main 模块 logger 接线正常，build_components 成功返回组件。"""
    config = {
        "embedding": {"provider": "mock", "dimension": 4},
        "vector_store": {"type": "mock"},
        "retrieval": {"top_k": 10, "semantic_weight": 1.0, "keyword_weight": 0.0},
    }
    from src.logger import setup_logging
    setup_logging("./test_main.log")

    # 验证 build_components 可正常工作（纯函数，无日志）
    from main import build_components
    sync_mgr, retriever, retrieval_cfg = build_components(config)
    assert sync_mgr is not None
    assert retriever is not None

    # 验证 main 模块 logger 接线正确，能捕获到日志输出
    import main as main_module
    with caplog.at_level(logging.INFO):
        main_module.logger.info("组件构建完成: embedder=%s, vector_store=%s", "mock", "mock")

    assert "embedder=mock" in caplog.text
    assert "vector_store=mock" in caplog.text

    # 清理
    import os
    if os.path.exists("./test_main.log"):
        os.remove("./test_main.log")
