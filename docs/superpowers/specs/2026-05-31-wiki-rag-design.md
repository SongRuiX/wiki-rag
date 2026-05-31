# Wiki-RAG 系统设计文档

**日期**: 2026-05-31
**状态**: 已确认

## 1. 概述

基于 MCP (Model Context Protocol) 的通用 Wiki RAG 系统，支持语义搜索、增量同步。数据源为本地 Markdown 文件目录，向量存储使用 Milvus local 模式。BM25 全文检索预留为后续扩展。

## 2. 架构

```
┌──────────────┐    ┌──────────┐    ┌───────────────┐    ┌─────────────┐
│   Markdown   │───▶│ Chunker  │───▶│   Embedder    │───▶│  Milvus     │
│  文件目录     │    │(按标题切分)│    │(Ollama/OpenAI)│    │(向量索引)   │
└──────────────┘    └──────────┘    └───────────────┘    └─────────────┘
       ▲                                                       │
       |                                                       ▼
┌──────────────┐                                      ┌─────────────────┐
│ SyncManager  │                                      │   Retriever     │
│(哈希增量同步) │                                      │(语义搜索+RRF预留)│
└──────────────┘                                      └─────────────────┘
       ▲                                                       │
       |                                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        MCP Server (main.py)                           │
│  工具: wiki_rag_sync(path, mode)  |  wiki_rag_search(query, top_k)   │
└──────────────────────────────────────────────────────────────────────┘
```

### 数据流

**同步**: 调用方传入 `path` → SyncManager 扫描 .md → 哈希对比 → Chunker 切分 → Embedder 向量化 → Milvus 写入

**搜索**: 调用方传入 `query` → Embedder 向量化 → Milvus 向量检索 → Retriever 返回排序结果

## 3. 数据模型 (`src/models.py`)

```python
@dataclass
class Chunk:
    id: str                      # UUID
    source_sn: str               # 文件名（不含 .md）
    source_title: str            # 文章标题（第一个 H1 或文件名）
    section_path: list[str]      # 章节路径，如 ["概述", "安装"]
    content: str                 # 片段文本
    embedding: list[float] | None  # 嵌入向量

@dataclass
class SearchResult:
    chunk: Chunk
    semantic_score: float          # 向量相似度 [0,1]
    keyword_score: float | None    # BM25 分数，当前为 None（预留）
    rrf_score: float               # 融合分数
    rrf_semantic_rank: int         # 语义排名
    rrf_keyword_rank: int | None   # BM25 排名，当前为 None（预留）

@dataclass
class SyncPlan:
    new_files: list[str]
    updated_files: list[str]
    deleted_files: list[str]
    total_chunks_estimate: int
```

## 4. 分块器 (`src/chunker.py`)

- 按 H1-H3 标题切分（`max_heading_level` 可配置，默认 3）
- 章节路径保留层级
- 无标题时整篇一个 Chunk
- `source_title` 优先取自 H1，回退到文件名

## 5. 嵌入器 (`src/embedder.py`)

```python
class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...
    @property
    @abstractmethod
    def dimension(self) -> int: ...
```

| 实现类 | 用途 |
|--------|------|
| `OllamaEmbedder` | 本地 Ollama（生产推荐） |
| `OpenAIEmbedder` | OpenAI API |
| `MockEmbedder` | np.random.randn（测试） |

## 6. 向量存储 (`src/vector_store.py`)

```python
class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> int: ...
    @abstractmethod
    def delete_by_sn(self, source_sns: list[str]) -> int: ...
    @abstractmethod
    def hybrid_search(
        self, query_vector: list[float], query_text: str,
        top_k: int, semantic_weight: float, keyword_weight: float
    ) -> list[SearchResult]: ...
```

### MilvusVectorStore

- **模式**: local（本地文件 `./milvus_wiki.db`）
- **集合**: `wiki_rag`
- **Schema**: id(VARCHAR PK), source_sn, source_title, section_path_str(JSON), content, vector(FLOAT_VECTOR)
- **索引**: IVF_FLAT, COSINE, nlist=128
- **连接**: `MilvusConnectionManager` 单例，跨进程文件锁重试

### 当前阶段限制

- 仅支持语义搜索（向量相似度），`keyword_weight=0`
- BM25 全文检索后续扩展（需要 remote Milvus + ANALYZER 索引）
- `hybrid_search` 接口已预留扩展参数

## 7. 检索器 (`src/retriever.py`)

```python
class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder): ...
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        # 1. embedder.embed_query(query)
        # 2. vector_store.hybrid_search(query_vector, query, top_k, 1.0, 0.0)
```

当前 RRF 退化为纯语义分数。接入 BM25 时，`Retriever` 接口不变。

## 8. 同步策略 (`src/sync_strategy.py`)

### 增量同步流程

1. 扫描 `path` 下所有 `*.md`
2. 计算 `MD5(title|content|mtime)` 哈希
3. 与 `.wiki_sync_metadata.json` 对比
4. 生成 SyncPlan（new/updated/deleted）
5. 执行 Chunker → Embedder → VectorStore 管线
6. 更新元数据文件

### 元数据文件 (`.wiki_sync_metadata.json`)

```json
{
  "last_sync_time": "2026-05-31T10:30:00",
  "files": {
    "getting-started.md": "a1b2c3d4...",
    "api-reference.md": "e5f6g7h8..."
  }
}
```

### 同步模式

| 模式 | 行为 |
|------|------|
| `incremental` | 哈希对比，只处理变更 |
| `full` | 清空集合，全量重同步 |

## 9. MCP 服务器 (`main.py`)

### 工具

| 工具 | 参数 | 返回 |
|------|------|------|
| `wiki_rag_sync` | `path: str`, `mode: str` ("incremental"/"full") | SyncPlan JSON |
| `wiki_rag_search` | `query: str`, `top_k: int` (默认 10) | list[SearchResult] JSON |

### 配置 (`config.yaml`)

```yaml
embedding:
  provider: "ollama"
  model: "qwen3-embedding:4b"
  base_url: "http://localhost:11434"
  batch_size: 16
  dimension: 2560

vector_store:
  type: "milvus"
  milvus:
    mode: "local"
    local_uri: "./milvus_wiki.db"
    remote_host: "localhost"
    remote_port: 19530
    collection: "wiki_rag"

retrieval:
  top_k: 10
  semantic_weight: 1.0    # 当前纯语义，接入 BM25 后调为 0.7
  keyword_weight: 0.0     # 预留

mcp_server:
  transport: "streamable-http"
  host: "127.0.0.1"
  port: 8000
  mount_path: "/mcp"
```

## 10. 项目文件结构

```
wiki-rag/
├── main.py
├── config.yaml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── config.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── sync_strategy.py
└── tests/
    ├── __init__.py
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_vector_store.py
    ├── test_retriever.py
    └── test_sync_strategy.py
```

## 11. 依赖

| 包 | 版本 | 用途 |
|----|------|------|
| mcp | >=1.0.0 | MCP 服务器框架 |
| pymilvus | >=2.4.0 | Milvus 向量数据库客户端 |
| openai | >=1.0.0 | OpenAI 嵌入 API |
| requests | >=2.31.0 | Ollama HTTP 调用 |
| pyyaml | >=6.0 | 配置解析 |
| pytest | >=7.4.0 | 测试框架 |
| pytest-cov | >=4.0 | 覆盖率 |
