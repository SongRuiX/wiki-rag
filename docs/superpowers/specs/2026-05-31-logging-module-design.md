# Logging Module Design

**Date**: 2026-05-31
**Status**: approved

## Overview

为 wiki-rag 项目增加日志系统，记录运行日志到当前目录的 `wiki-rag.log` 文件，控制台同步输出 INFO 级别。

## Decisions

| 决策 | 选项 |
|------|------|
| 日志库 | Python 标准 `logging` 模块 |
| 输出目标 | 控制台 INFO + 文件 DEBUG |
| 文件轮转 | 单文件 `wiki-rag.log`，不轮转 |
| 覆盖范围 | `main.py` + `src/` 下全部模块 |
| 日志格式 | `时间 级别 模块名 函数名:行号 消息` |

## Module Structure

新增 `src/logger.py`，提供 `setup_logging()` 函数：

```
src/
├── logger.py          # 新增：日志配置模块
├── config.py
├── models.py
├── chunker.py         # 修改：接入 logger
├── embedder.py        # 修改：接入 logger
├── vector_store.py    # 修改：接入 logger
├── retriever.py       # 修改：接入 logger
├── sync_strategy.py   # 修改：接入 logger
└── __init__.py

main.py                # 修改：启动时调用 setup_logging()，替换 print()
```

## API: `setup_logging()`

```python
def setup_logging(log_file: str = "./wiki-rag.log") -> None:
    """配置日志系统。控制台 INFO 级别，文件 DEBUG 级别。"""
```

- 在 `main.py:build_components()` 之前调用
- 配置完成后，其他模块通过 `logging.getLogger(__name__)` 获取 logger
- 使用 `DictConfigurator` 或手动构造 handler/formatter，避免多个 handler 重复添加

## Format

```
2026-05-31 19:55:27 INFO  src.sync_strategy sync:42 同步开始: /path/to/wiki, 12 文件
```

格式字符串：`%(asctime)s %(levelname)-5s %(name)s %(funcName)s:%(lineno)d %(message)s`

- 时间精度：秒（`%Y-%m-%d %H:%M:%S`）
- 控制台和文件使用相同格式

## Log Points

### main.py

| 事件 | 级别 |
|------|------|
| 组件构建成功 (embedder/vector_store 类型) | INFO |
| 启动失败 | ERROR |
| MCP 工具调用: wiki_rag_sync | INFO |
| MCP 工具调用: wiki_rag_search | INFO |
| 服务器关闭 | INFO |

### sync_strategy.py

| 事件 | 级别 |
|------|------|
| 同步开始 (path, mode) | INFO |
| 扫描文件数量 | DEBUG |
| 变更统计 (new/updated/deleted) | INFO |
| 同步完成 | INFO |

### embedder.py

| 事件 | 级别 |
|------|------|
| 嵌入请求 (texts 数量) | DEBUG |
| 嵌入完成 (耗时 ms) | DEBUG |
| 连接/API 错误 | ERROR（重新 raise） |

### vector_store.py

| 事件 | 级别 |
|------|------|
| 集合创建 | INFO |
| chunks 添加数量 | INFO |
| chunks 删除数量 | INFO |
| 搜索请求 | DEBUG |

### retriever.py

| 事件 | 级别 |
|------|------|
| 查询请求 (query, top_k) | INFO |
| 结果数量 | INFO |
| 搜索耗时 (ms) | DEBUG |

### chunker.py

不接入日志（纯本地计算，无 IO）。

## Error Strategy

- `embedder.py` 记录 ERROR 后重新抛出，由上层决定是否继续
- `vector_store.py` 同样：记录 ERROR 后重新抛出
- `main.py` 启动失败记录 ERROR 后 `sys.exit(1)`

## Non-Goals

- 不添加 loguru/structlog 等第三方依赖
- 不实现日志轮转
- 不记录请求体/响应体等可能含用户数据的内容

## Testing

- 新增 `tests/test_logger.py`，验证 `setup_logging()` 创建了文件和控制台 handler
- 使用 `caplog` fixture 在现有测试中验证关键日志输出
