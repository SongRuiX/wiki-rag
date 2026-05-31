# MCP URL 直连配置

**Date**: 2026-05-31
**Status**: approved

## Overview

将 `.mcp.json` 从 stdio 模式改为 streamable-http URL 直连模式，手动启动服务器。

## Design

### `.mcp.json` 配置

```json
{
  "mcpServers": {
    "wiki-rag": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### 使用流程

1. 启动 Ollama + Milvus（生产模式）
2. 终端运行 `python main.py --transport streamable-http --port 8000`
3. 重启 Claude Code，自动连接

### 联通性测试

```bash
curl -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

预期返回 `wiki_rag_sync` 和 `wiki_rag_search`。
