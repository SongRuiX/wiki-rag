# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

Wiki RAG — 一个基于 MCP (Model Context Protocol) 的通用 Wiki 检索增强生成系统，支持语义搜索、增量同步、混合检索（语义 + 关键词 + RRF 融合）以及可插拔的嵌入/向量存储后端。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 MCP 服务器（默认: streamable-http 模式，监听 127.0.0.1:8000/mcp）
python main.py

# 指定传输协议
python main.py --transport stdio
python main.py --transport sse --port 8000
python main.py --transport streamable-http

# 运行所有测试
pytest tests/ -v

# 带覆盖率运行
pytest --cov=src tests/
```

## 架构

**数据处理管线**: `SyncManager → Chunker → Embedder → VectorStore → Retriever (混合检索 + RRF 融合)`

**MCP 层**: `main.py` 通过 `mcp` 库暴露两个工具 — `wiki_rag_sync` 和 `wiki_rag_search`。

### 核心抽象（位于 `src/`）

| 模块 | 关键接口/类 | 用途 |
|---|---|---|
| `models.py` | `Chunk`, `SearchResult`, `SyncPlan` | 各模块共享的数据模型 |
| `chunker.py` | Chunker | 按 Markdown 标题层级切分文章（1-6 级，最大深度可配置） |
| `embedder.py` | `Embedder`（抽象接口）, `OllamaEmbedder`, `OpenAIEmbedder`, `MockEmbedder` | 文本 → 向量嵌入 |
| `vector_store.py` | `VectorStore`（抽象接口）, `MilvusVectorStore`, `MockVectorStore` | 向量增删 + 混合搜索 |
| `retriever.py` | Retriever | 混合检索：语义搜索 + 关键词搜索 + RRF 融合（`k=60`） |
| `sync_strategy.py` | SyncManager | 基于内容哈希对比的增量同步（对 `title\|content\|modify_time` 做 MD5） |

### 关键设计决策

- **MilvusVectorStore** 使用 IVF_FLAT 索引，COSINE 相似度（nlist=128）。连接管理通过 `MilvusConnectionManager` 单例模式实现，支持跨进程文件锁定重试。
- **RRF 融合**: `score = Σ(1 / (k + rank))`，其中 `k=60`。最终得分加权：`semantic_weight * rrf_semantic + keyword_weight * rrf_keyword`。
- **同步元数据** 持久化到 `.wiki_sync_metadata.json`（记录最后同步时间和每篇文章的内容哈希）。
- **嵌入向量维度** 取决于模型选择（如 qwen3-embedding:4b → 2560）。

### 配置

`config.yaml` 驱动整个系统。主要配置项：`wiki`、`embedding`（provider/model/batch_size）、`vector_store`（milvus mode/local-uri/collection）、`retrieval`（top_k/weights）、`mcp_server`（transport/host/port）。

### 依赖

`mcp`, `pymilvus`, `openai`, `requests`, `pyyaml`, `pytest`
