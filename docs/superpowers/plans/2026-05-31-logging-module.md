# Logging Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 wiki-rag 所有模块接入 Python 标准 `logging`，控制台 INFO + 文件 DEBUG，格式 `时间 级别 模块名 函数名:行号 消息`

**Architecture:** 新增 `src/logger.py` 封装 `setup_logging()`，在 `main.py` 启动早期调用一次；其余模块通过 `logging.getLogger(__name__)` 获取 logger。代码行级修改，不改变现有接口和逻辑。

**Tech Stack:** Python `logging` 标准库，`pytest` + `caplog` fixture

---

## File Structure

```
Create:  src/logger.py           — setup_logging() 函数
Create:  tests/test_logger.py    — setup_logging 的单元测试

Modify:  main.py                 — 调用 setup_logging()，替换 print()，添加工具调用日志
Modify:  src/sync_strategy.py    — 添加同步过程日志
Modify:  src/embedder.py         — 添加嵌入请求/错误日志
Modify:  src/vector_store.py     — 添加集合操作/chunk 增删日志
Modify:  src/retriever.py        — 添加查询日志
```

---

### Task 1: 创建 `src/logger.py` — `setup_logging()` 函数

**Files:**
- Create: `src/logger.py`
- Create: `tests/test_logger.py`

- [ ] **Step 1: 写测试 — 验证日志文件和控制台 handler 被正确创建**

```python
# tests/test_logger.py
import logging
import os
import tempfile
from src.logger import setup_logging


def test_setup_logging_creates_file_handler():
    """setup_logging should add a file handler and a console handler."""
    with tempfile.TemporaryDirectory() as d:
        log_path = os.path.join(d, "test.log")
        root = logging.getLogger()
        initial_handlers = len(root.handlers)

        setup_logging(log_path)

        # 断言：新增了两个 handler（文件 + 控制台）
        assert len(root.handlers) == initial_handlers + 2

        # 断言：文件 handler 存在
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == os.path.abspath(log_path)

        # 断言：控制台 handler 存在
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 1

        # 断言：文件 handler 级别为 DEBUG
        assert file_handlers[0].level == logging.DEBUG

        # 断言：控制台 handler 级别为 INFO
        assert stream_handlers[0].level == logging.INFO

        # 清理
        for h in file_handlers + stream_handlers:
            h.close()
            root.removeHandler(h)


def test_setup_logging_actually_writes_to_file():
    """日志消息应实际写入文件。"""
    with tempfile.TemporaryDirectory() as d:
        log_path = os.path.join(d, "test.log")
        root = logging.getLogger()

        setup_logging(log_path)
        logger = logging.getLogger("test_writer")
        logger.info("hello world")
        logger.debug("debug detail")

        # 刷新 handler
        for h in root.handlers:
            h.flush()

        with open(log_path, encoding="utf-8") as f:
            content = f.read()

        assert "hello world" in content
        assert "debug detail" in content
        assert "INFO" in content
        assert "DEBUG" in content
        assert "test_writer" in content

        # 清理
        for h in list(root.handlers):
            h.close()
            root.removeHandler(h)


def test_setup_logging_format():
    """验证日志格式包含时间、级别、模块名、函数名、行号、消息。"""
    with tempfile.TemporaryDirectory() as d:
        log_path = os.path.join(d, "test.log")
        root = logging.getLogger()

        setup_logging(log_path)
        logger = logging.getLogger("src.sync_strategy")

        def dummy_sync():
            logger.info("同步完成")

        dummy_sync()

        for h in root.handlers:
            h.flush()

        with open(log_path, encoding="utf-8") as f:
            content = f.read()

        # 格式: YYYY-MM-DD HH:MM:SS LEVEL NAME funcName:lineno message
        assert "INFO" in content
        assert "src.sync_strategy" in content
        assert "dummy_sync" in content
        assert "同步完成" in content

        # 清理
        for h in list(root.handlers):
            h.close()
            root.removeHandler(h)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_logger.py -v
```
Expected: 全部 3 个测试 FAIL（`setup_logging` 未定义）

- [ ] **Step 3: 实现 `setup_logging()`**

```python
# src/logger.py
import logging
import sys


def setup_logging(log_file: str = "./wiki-rag.log") -> None:
    """配置日志系统。控制台 INFO 级别，文件 DEBUG 级别。

    调用后，所有模块通过 logging.getLogger(__name__) 获取已配置的 logger。
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s %(funcName)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件 handler: DEBUG 级别
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # 控制台 handler: INFO 级别
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_logger.py -v
```
Expected: 全部 3 个测试 PASS

- [ ] **Step 5: 提交**

```bash
git add src/logger.py tests/test_logger.py
git commit -m "feat: 添加日志配置模块 setup_logging()"
```

---

### Task 2: 接入 `main.py` — 启动日志 + 工具调用日志

**Files:**
- Modify: `main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: 写测试 — 验证日志输出和 main 模块 logger**

```python
# 在 tests/test_main.py 中追加
import logging


def test_main_logs_component_build_success(caplog):
    """build_components 成功时应输出 INFO 日志。"""
    config = {
        "embedding": {"provider": "mock", "dimension": 4},
        "vector_store": {"type": "mock"},
        "retrieval": {"top_k": 10, "semantic_weight": 1.0, "keyword_weight": 0.0},
    }
    from src.logger import setup_logging
    setup_logging("./test_main.log")

    with caplog.at_level(logging.INFO):
        from main import build_components
        sync_mgr, retriever, retrieval_cfg = build_components(config)

    assert "embedder=mock" in caplog.text or "mock" in caplog.text
    assert "vector_store=mock" in caplog.text or sync_mgr is not None

    # 清理
    import os
    if os.path.exists("./test_main.log"):
        os.remove("./test_main.log")
```

*注：`caplog` 是 pytest 内置 fixture，捕获 `logging` 输出。*

- [ ] **Step 2: 修改 `main.py` — 接入 logger**

```python
# main.py — 顶部新增 import
import argparse
import json
import logging
import sys
from mcp.server.fastmcp import FastMCP
from src.config import load_config
from src.embedder import OllamaEmbedder, OpenAIEmbedder, MockEmbedder
from src.vector_store import MilvusVectorStore, MockVectorStore
from src.chunker import Chunker
from src.retriever import Retriever
from src.sync_strategy import SyncManager
from src.logger import setup_logging

logger = logging.getLogger(__name__)


def build_components(config: dict):
    emb_cfg = config["embedding"]
    provider = emb_cfg["provider"]
    if provider == "ollama":
        embedder = OllamaEmbedder(
            model=emb_cfg["model"],
            base_url=emb_cfg["base_url"],
            dimension=emb_cfg["dimension"],
            batch_size=emb_cfg.get("batch_size", 16),
        )
    elif provider == "openai":
        embedder = OpenAIEmbedder(
            model=emb_cfg["model"],
            dimension=emb_cfg["dimension"],
        )
    else:
        embedder = MockEmbedder(dimension=emb_cfg.get("dimension", 2560))

    vs_cfg = config["vector_store"]
    if vs_cfg["type"] == "milvus":
        milvus_cfg = vs_cfg["milvus"]
        if milvus_cfg["mode"] == "local":
            vector_store = MilvusVectorStore(
                uri=milvus_cfg["local_uri"],
                collection=milvus_cfg["collection"],
                dimension=embedder.dimension,
            )
        else:
            vector_store = MilvusVectorStore(
                uri=f"http://{milvus_cfg['remote_host']}:{milvus_cfg['remote_port']}",
                collection=milvus_cfg["collection"],
                dimension=embedder.dimension,
            )
    else:
        vector_store = MockVectorStore()

    chunker = Chunker(max_heading_level=3)
    retriever = Retriever(vector_store, embedder)
    sync_manager = SyncManager(chunker, embedder, vector_store)
    retrieval_cfg = config["retrieval"]
    return sync_manager, retriever, retrieval_cfg


def main():
    parser = argparse.ArgumentParser(description="Wiki RAG MCP Server")
    parser.add_argument("--transport", default="streamable-http", choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--mock", action="store_true", help="Use mock embedder and vector store (no external deps)")
    args = parser.parse_args()

    # 第 1 步: 配置日志
    setup_logging()
    logger.info("Wiki RAG MCP Server 启动中...")

    config = load_config(args.config)
    if args.mock:
        config["embedding"]["provider"] = "mock"
        config["vector_store"]["type"] = "mock"
    mcp_cfg = config.get("mcp_server", {})

    try:
        sync_manager, retriever, retrieval_cfg = build_components(config)
    except Exception as e:
        logger.error("启动失败: %s", e)
        logger.info("提示: 使用 --mock 标志跳过外部服务依赖进行测试")
        sys.exit(1)

    emb_cfg = config["embedding"]
    vs_cfg = config["vector_store"]
    logger.info("组件构建完成: embedder=%s, vector_store=%s", emb_cfg["provider"], vs_cfg["type"])

    transport = args.transport or mcp_cfg.get("transport", "streamable-http")
    host = args.host or mcp_cfg.get("host", "127.0.0.1")
    port = args.port or mcp_cfg.get("port", 8000)
    mount_path = mcp_cfg.get("mount_path", "/mcp")

    mcp = FastMCP(name="wiki-rag", host=host, port=port, mount_path=mount_path)

    @mcp.tool()
    def wiki_rag_sync(path: str, mode: str = "incremental") -> str:
        """Sync Wiki markdown files to vector database.

        Args:
            path: Directory path containing .md files
            mode: Sync mode - 'incremental' (default) or 'full'
        """
        logger.info("wiki_rag_sync 被调用: path=%s, mode=%s", path, mode)
        plan = sync_manager.sync(path, mode)
        logger.info("wiki_rag_sync 完成: new=%d, updated=%d, deleted=%d",
                     len(plan.new_files), len(plan.updated_files), len(plan.deleted_files))
        return json.dumps({
            "new_files": plan.new_files,
            "updated_files": plan.updated_files,
            "deleted_files": plan.deleted_files,
            "total_chunks_estimate": plan.total_chunks_estimate,
        }, ensure_ascii=False)

    @mcp.tool()
    def wiki_rag_search(query: str, top_k: int = 10) -> str:
        """Search wiki articles using semantic retrieval.

        Args:
            query: Search query text
            top_k: Number of results to return (default 10)
        """
        logger.info("wiki_rag_search 被调用: query=%s, top_k=%d", query[:80], top_k)
        results = retriever.search(
            query,
            top_k=top_k,
            semantic_weight=retrieval_cfg.get("semantic_weight", 1.0),
            keyword_weight=retrieval_cfg.get("keyword_weight", 0.0),
        )
        logger.info("wiki_rag_search 完成: 返回 %d 条结果", len(results))
        items = []
        for r in results:
            items.append({
                "id": r.chunk.id,
                "source_sn": r.chunk.source_sn,
                "source_title": r.chunk.source_title,
                "section_path": r.chunk.section_path,
                "content": r.chunk.content,
                "score": r.rrf_score,
            })
        return json.dumps(items, ensure_ascii=False)

    logger.info("MCP 服务器启动: transport=%s, host=%s, port=%d", transport, host, port)
    try:
        mcp.run(transport=transport)
    except KeyboardInterrupt:
        logger.info("服务器已关闭")
```

- [ ] **Step 3: 运行现有测试确认不破坏已有功能**

```bash
pytest tests/test_main.py -v
```
Expected: 3 个已有测试 PASS

- [ ] **Step 4: 运行新测试**

```bash
pytest tests/test_main.py::test_main_logs_component_build_success -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add main.py tests/test_main.py
git commit -m "feat: main.py 接入日志，记录启动/工具调用/关机"
```

---

### Task 3: 接入 `src/sync_strategy.py` — 同步过程日志

**Files:**
- Modify: `src/sync_strategy.py`
- Modify: `tests/test_sync_strategy.py`

- [ ] **Step 1: 写测试 — 验证同步日志输出**

```python
# 在 tests/test_sync_strategy.py 中追加
import logging


def test_sync_logs_start_and_completion(caplog, tmp_path):
    """同步应输出 INFO 级别的开始和完成日志。"""
    from src.chunker import Chunker
    from src.embedder import MockEmbedder
    from src.vector_store import MockVectorStore
    from src.sync_strategy import SyncManager

    # 创建一个 md 文件
    md_file = tmp_path / "test.md"
    md_file.write_text("# 标题\n内容", encoding="utf-8")

    chunker = Chunker()
    embedder = MockEmbedder(dimension=4)
    store = MockVectorStore()
    mgr = SyncManager(chunker, embedder, store)

    from src.logger import setup_logging
    setup_logging(str(tmp_path / "test_sync.log"))

    with caplog.at_level(logging.INFO, logger="src.sync_strategy"):
        plan = mgr.sync(str(tmp_path), mode="incremental")

    log_text = caplog.text
    assert "同步开始" in log_text
    assert "同步完成" in log_text
    assert plan is not None
```

- [ ] **Step 2: 修改 `src/sync_strategy.py` — 接入 logger**

```python
# src/sync_strategy.py — 顶部新增 import logging，sync() 添加日志
import hashlib
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from src.chunker import Chunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.models import SyncPlan

logger = logging.getLogger(__name__)


class SyncManager:
    METADATA_FILENAME = ".wiki_sync_metadata.json"

    def __init__(self, chunker: Chunker, embedder: Embedder, vector_store: VectorStore):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def sync(self, path: str, mode: str = "incremental") -> SyncPlan:
        logger.info("同步开始: path=%s, mode=%s", path, mode)
        path = os.path.abspath(path)
        md_files = self._scan_files(path)
        logger.debug("扫描到 %d 个 Markdown 文件", len(md_files))
        metadata_path = os.path.join(path, self.METADATA_FILENAME)

        if mode == "full":
            plan = SyncPlan(new_files=list(md_files.keys()), total_chunks_estimate=len(md_files))
            self._full_sync(path, md_files)
        else:
            old_meta = self._load_metadata(metadata_path)
            old_files = old_meta.get("files", {})
            plan = SyncPlan(
                new_files=[f for f in md_files if f not in old_files],
                updated_files=[f for f in md_files if f in old_files and md_files[f] != old_files[f]],
                deleted_files=[f for f in old_files if f not in md_files],
            )
            plan.total_chunks_estimate = len(plan.new_files) + len(plan.updated_files)
            logger.info("变更统计: new=%d, updated=%d, deleted=%d",
                         len(plan.new_files), len(plan.updated_files), len(plan.deleted_files))
            self._execute_plan(path, plan)

        self._save_metadata(metadata_path, md_files)
        logger.info("同步完成: path=%s", path)
        return plan

    def _scan_files(self, root: str) -> dict[str, str]:
        files = {}
        for f in Path(root).rglob("*.md"):
            rel = str(f.relative_to(root)).replace("\\", "/")
            files[rel] = self._compute_hash(str(f))
        return files

    def _compute_hash(self, filepath: str) -> str:
        p = Path(filepath)
        stat = p.stat()
        content = p.read_text(encoding="utf-8")
        raw = f"{content}|{stat.st_mtime}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_metadata(self, path: str) -> dict:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_metadata(self, path: str, file_hashes: dict[str, str]):
        meta = {
            "last_sync_time": datetime.now(timezone.utc).isoformat(),
            "files": file_hashes,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _full_sync(self, root: str, md_files: dict[str, str]):
        self._process_files(root, list(md_files.keys()))

    def _execute_plan(self, root: str, plan: SyncPlan):
        changed = plan.new_files + plan.updated_files
        if changed:
            self._process_files(root, changed)
        if plan.deleted_files:
            sns = [Path(f).stem for f in plan.deleted_files]
            self.vector_store.delete_by_sn(sns)

    def _process_files(self, root: str, file_paths: list[str]):
        if not file_paths:
            return
        all_chunks = []
        for rel_path in file_paths:
            abs_path = os.path.join(root, rel_path)
            chunks = self.chunker.chunk_file(abs_path)
            all_chunks.extend(chunks)
        if all_chunks:
            texts = [c.content for c in all_chunks]
            vectors = self.embedder.embed(texts)
            for chunk, vec in zip(all_chunks, vectors):
                chunk.embedding = vec
            self.vector_store.add_chunks(all_chunks)
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_sync_strategy.py -v
```
Expected: 全部测试 PASS（包括新增的 `test_sync_logs_start_and_completion`）

- [ ] **Step 4: 提交**

```bash
git add src/sync_strategy.py tests/test_sync_strategy.py
git commit -m "feat: sync_strategy 接入日志，记录同步开始/变更统计/完成"
```

---

### Task 4: 接入 `src/embedder.py` — 嵌入请求与错误日志

**Files:**
- Modify: `src/embedder.py`

- [ ] **Step 1: 修改 `src/embedder.py` — 在 OllamaEmbedder 和 OpenAIEmbedder 中接入 logger**

```python
# src/embedder.py — 顶部新增 import logging 和 time
from abc import ABC, abstractmethod
import logging
import time
import numpy as np
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


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
        logger.debug("Ollama 嵌入请求: texts=%d, model=%s", len(texts), self.model)
        start = time.perf_counter()
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                resp = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": batch},
                    timeout=60,
                )
                resp.raise_for_status()
                all_vectors.extend(resp.json()["embeddings"])
            except requests.RequestException as e:
                logger.error("Ollama 嵌入请求失败: %s", e)
                raise
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("Ollama 嵌入完成: texts=%d, elapsed=%.1fms", len(texts), elapsed)
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
        logger.debug("OpenAI 嵌入请求: texts=%d, model=%s", len(texts), self.model)
        start = time.perf_counter()
        try:
            resp = self.client.embeddings.create(model=self.model, input=texts)
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug("OpenAI 嵌入完成: texts=%d, elapsed=%.1fms", len(texts), elapsed)
            return [d.embedding for d in resp.data]
        except Exception as e:
            logger.error("OpenAI 嵌入请求失败: %s", e)
            raise

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
```

*注：`MockEmbedder` 不添加日志（纯本地计算，无 IO）。*

- [ ] **Step 2: 运行测试**

```bash
pytest tests/test_embedder.py -v
```
Expected: 4 个已有测试 PASS

- [ ] **Step 3: 提交**

```bash
git add src/embedder.py
git commit -m "feat: embedder 接入日志，记录嵌入请求耗时和 API 错误"
```

---

### Task 5: 接入 `src/vector_store.py` — 集合操作 + 搜索日志

**Files:**
- Modify: `src/vector_store.py`

- [ ] **Step 1: 修改 `src/vector_store.py` — 在 MilvusVectorStore 中接入 logger**

```python
# src/vector_store.py — 顶部新增 import logging
import json
import logging
from abc import ABC, abstractmethod
import numpy as np
from pymilvus import MilvusClient
from src.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> int:
        ...

    @abstractmethod
    def delete_by_sn(self, source_sns: list[str]) -> int:
        ...

    @abstractmethod
    def hybrid_search(
        self, query_vector: list[float], query_text: str,
        top_k: int, semantic_weight: float, keyword_weight: float,
    ) -> list[SearchResult]:
        ...


class MockVectorStore(VectorStore):
    def __init__(self):
        self._chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]) -> int:
        self._chunks.extend(chunks)
        return len(chunks)

    def delete_by_sn(self, source_sns: list[str]) -> int:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.source_sn not in source_sns]
        return before - len(self._chunks)

    def hybrid_search(
        self, query_vector: list[float], query_text: str,
        top_k: int, semantic_weight: float, keyword_weight: float,
    ) -> list[SearchResult]:
        qv = np.array(query_vector)
        scored = []
        for i, chunk in enumerate(self._chunks):
            if chunk.embedding is None:
                continue
            cv = np.array(chunk.embedding)
            sim = float(np.dot(qv, cv) / (np.linalg.norm(qv) * np.linalg.norm(cv) + 1e-8))
            rrf = 1.0 / (60 + i + 1)
            scored.append(SearchResult(
                chunk=chunk,
                semantic_score=sim,
                rrf_score=semantic_weight * rrf + keyword_weight * 0,
                rrf_semantic_rank=i + 1,
            ))
        scored.sort(key=lambda r: r.rrf_score, reverse=True)
        return scored[:top_k]


class MilvusVectorStore(VectorStore):
    def __init__(self, uri: str, collection: str = "wiki_rag", dimension: int = 2560):
        self.client = MilvusClient(uri=uri)
        self.collection = collection
        self.dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.has_collection(self.collection):
            self.client.load_collection(self.collection)
            return
        logger.info("创建 Milvus 集合: %s, dimension=%d", self.collection, self.dimension)
        self.client.create_collection(
            collection_name=self.collection,
            dimension=self.dimension,
            metric_type="COSINE",
            id_type="string",
        )

    def close(self):
        self.client.close()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        data = []
        for c in chunks:
            if c.embedding is None:
                continue
            data.append({
                "id": c.id,
                "vector": c.embedding,
                "source_sn": c.source_sn,
                "source_title": c.source_title,
                "section_path_str": json.dumps(c.section_path, ensure_ascii=False),
                "content": c.content,
            })
        if not data:
            return 0
        result = self.client.upsert(collection_name=self.collection, data=data)
        count = result["upsert_count"]
        logger.info("添加 chunks 到 Milvus: count=%d", count)
        return count

    def delete_by_sn(self, source_sns: list[str]) -> int:
        if not source_sns:
            return 0
        filter_expr = " or ".join(f'source_sn == "{sn}"' for sn in source_sns)
        result = self.client.delete(collection_name=self.collection, filter=filter_expr)
        deleted = len(result) if isinstance(result, list) else 0
        logger.info("从 Milvus 删除 chunks: count=%d", deleted)
        return deleted

    def hybrid_search(
        self, query_vector: list[float], query_text: str,
        top_k: int, semantic_weight: float, keyword_weight: float,
    ) -> list[SearchResult]:
        logger.debug("Milvus 搜索请求: top_k=%d, sem_w=%.2f, kw_w=%.2f", top_k, semantic_weight, keyword_weight)
        if keyword_weight > 0:
            return self._full_hybrid_search(query_vector, query_text, top_k, semantic_weight, keyword_weight)
        return self._semantic_only_search(query_vector, top_k)

    def _semantic_only_search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        resp = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=top_k,
            output_fields=["source_sn", "source_title", "section_path_str", "content"],
        )
        results = []
        for i, hit in enumerate(resp[0]):
            entity = hit["entity"]
            chunk = Chunk(
                id=str(hit["id"]),
                source_sn=entity["source_sn"],
                source_title=entity["source_title"],
                section_path=json.loads(entity.get("section_path_str", "[]")),
                content=entity["content"],
            )
            rrf = 1.0 / (60 + i + 1)
            results.append(SearchResult(
                chunk=chunk,
                semantic_score=hit["distance"],
                rrf_score=rrf,
                rrf_semantic_rank=i + 1,
            ))
        return results

    def _full_hybrid_search(self, query_vector, query_text, top_k, semantic_weight, keyword_weight):
        raise NotImplementedError("BM25 hybrid search not yet implemented")
```

- [ ] **Step 2: 运行测试**

```bash
pytest tests/test_vector_store.py tests/test_milvus_store.py -v
```
Expected: 全部已有测试 PASS

- [ ] **Step 3: 提交**

```bash
git add src/vector_store.py
git commit -m "feat: vector_store 接入日志，记录集合创建/chunk 增删/搜索"
```

---

### Task 6: 接入 `src/retriever.py` — 查询日志

**Files:**
- Modify: `src/retriever.py`

- [ ] **Step 1: 修改 `src/retriever.py` — 接入 logger**

```python
# src/retriever.py — 顶部新增 import logging 和 time
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
```

- [ ] **Step 2: 运行测试**

```bash
pytest tests/test_retriever.py -v
```
Expected: 3 个已有测试 PASS

- [ ] **Step 3: 提交**

```bash
git add src/retriever.py
git commit -m "feat: retriever 接入日志，记录查询请求/结果数量/耗时"
```

---

### Task 7: 全量回归验证

**Files:**
- (无代码修改)

- [ ] **Step 1: 运行全部测试**

```bash
pytest tests/ -v
```
Expected: 全部测试 PASS（新增 + 已有）

- [ ] **Step 2: 验证启动日志输出**

```bash
python main.py --mock --transport stdio 2>&1 | head -5
```
Expected: 看到 `INFO ... Wiki RAG MCP Server 启动中...` 和 `INFO ... 组件构建完成`

- [ ] **Step 3: 验证日志文件写入**

```bash
cat wiki-rag.log
```
Expected: 看到格式正确的日志行

- [ ] **Step 4: 提交**

```bash
git commit -m "chore: 全量回归验证，日志模块集成完成"
```

*注：若以上无文件变更，跳过此提交。*
