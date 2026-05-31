# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Wiki RAG — 基于 MCP 的 Wiki 检索增强生成系统。将本地 Markdown 文件目录同步到 Milvus 向量数据库，通过 MCP 协议暴露语义搜索能力。

## 快速开始

```bash
pip install -r requirements.txt        # 安装依赖
pytest tests/ -v                       # 运行全部测试（31 个）
pytest tests/test_chunker.py -v        # 运行单个测试文件
pytest --cov=src tests/                # 覆盖率报告
```

## 启动 MCP 服务器

```bash
# 无外部依赖（测试用，使用 Mock 组件）
python main.py --mock --transport stdio

# 生产模式（需 Ollama + Milvus 运行中）
python main.py --transport streamable-http --port 8000

# SSE 模式
python main.py --transport sse --port 8000

# 指定配置文件
python main.py --config /path/to/config.yaml
```

## 架构

**处理管线**: `SyncManager → Chunker → Embedder → VectorStore → Retriever`

```
Markdown 文件目录
    │
    ▼
SyncManager (src/sync_strategy.py)
    │  扫描 *.md, MD5 哈希增量对比, 持久化 .wiki_sync_metadata.json
    ▼
Chunker (src/chunker.py)
    │  按 H1-H3 标题切分为 Chunk 列表, 维护 section_path 层级
    ▼
Embedder (src/embedder.py)
    │  抽象接口, 三个实现: OllamaEmbedder / OpenAIEmbedder / MockEmbedder
    ▼
VectorStore (src/vector_store.py)
    │  抽象接口, 两个实现: MilvusVectorStore (local/remote) / MockVectorStore
    ▼
Retriever (src/retriever.py)
      调用 embedder.embed_query() → vector_store.hybrid_search() 返回结果
```

**MCP 层**: `main.py` 通过 FastMCP 暴露两个工具 — `wiki_rag_sync` 和 `wiki_rag_search`

## 核心抽象

| 接口 | 文件 | 方法 |
|------|------|------|
| `Embedder` | `src/embedder.py` | `embed(texts)`, `embed_query(text)`, `dimension` |
| `VectorStore` | `src/vector_store.py` | `add_chunks(chunks)`, `delete_by_sn(sns)`, `hybrid_search(vector, text, top_k, sem_w, kw_w)` |

所有模块通过依赖注入组装（见 `main.py:build_components()`），上层依赖抽象接口。

## 数据模型 (`src/models.py`)

- **Chunk** — 文章片段，含 id(UUID)、source_sn(文件名)、source_title(首个 H1)、section_path(标题层级)、content、embedding
- **SearchResult** — 搜索结果，含 semantic_score、rrf_score、预留的 keyword_score
- **SyncPlan** — 同步计划，含 new_files / updated_files / deleted_files

## 关键设计决策

- **分块策略**: 按 Markdown 标题层级（H1-H3）切分，章节路径维护在 `section_path` 中，首个 H1 作为 `source_title`
- **向量存储**: Milvus local 模式（`./milvus_wiki.db`），IVF_FLAT 索引 + COSINE 相似度。**已知问题**：Windows 下 MilvusLite 进程崩溃后 LOCK 文件残留，需手动删除 `milvus_wiki.db/` 目录后重启
- **同步**: 增量模式通过 MD5(`content|mtime`) 哈希对比变更，元数据写入目标目录的 `.wiki_sync_metadata.json`
- **检索**: 当前为纯语义搜索（`semantic_weight=1.0, keyword_weight=0.0`），BM25 混合检索通过 `hybrid_search` 接口预留，`MilvusVectorStore` 中 `_full_hybrid_search` 抛出 `NotImplementedError`
- **配置**: `config.yaml` 驱动，`src/config.py` 提供 `load_config()` / `get_config()`
- **测试覆盖**: 31 个测试，87% 覆盖率。未覆盖部分主要是 Ollama/OpenAI 实现（需外部服务）和 BM25 预留接口

## 测试

```bash
pytest tests/ -v                                   # 全部 31 个测试
pytest tests/test_main.py -v                       # main.py 构建逻辑
pytest tests/test_chunker.py -v                    # 分块器 (8 个)
pytest tests/test_embedder.py -v                   # 嵌入器 (4 个)
pytest tests/test_vector_store.py -v               # Mock 向量存储 (4 个)
pytest tests/test_milvus_store.py -v               # Milvus 向量存储 (3 个)
pytest tests/test_retriever.py -v                  # 检索器 (3 个)
pytest tests/test_sync_strategy.py -v              # 同步管理器 (6 个)
```

测试依赖 MockEmbedder 和 MockVectorStore，无需外部服务。Milvus 测试需要 `pymilvus[milvus_lite]` 已安装。
