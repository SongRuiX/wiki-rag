import argparse
import json
from mcp.server.fastmcp import FastMCP
from src.config import load_config
from src.embedder import OllamaEmbedder, OpenAIEmbedder, MockEmbedder
from src.vector_store import MilvusVectorStore, MockVectorStore
from src.chunker import Chunker
from src.retriever import Retriever
from src.sync_strategy import SyncManager


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
    args = parser.parse_args()

    config = load_config(args.config)
    mcp_cfg = config.get("mcp_server", {})
    sync_manager, retriever, retrieval_cfg = build_components(config)

    mcp = FastMCP(name="wiki-rag")

    @mcp.tool()
    def wiki_rag_sync(path: str, mode: str = "incremental") -> str:
        """Sync Wiki markdown files to vector database.

        Args:
            path: Directory path containing .md files
            mode: Sync mode - 'incremental' (default) or 'full'
        """
        plan = sync_manager.sync(path, mode)
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
        results = retriever.search(
            query,
            top_k=top_k,
            semantic_weight=retrieval_cfg.get("semantic_weight", 1.0),
            keyword_weight=retrieval_cfg.get("keyword_weight", 0.0),
        )
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

    transport = args.transport or mcp_cfg.get("transport", "streamable-http")
    host = args.host or mcp_cfg.get("host", "127.0.0.1")
    port = args.port or mcp_cfg.get("port", 8000)
    mount_path = mcp_cfg.get("mount_path", "/mcp")

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse", host=host, port=port, mount_path=mount_path)
    else:
        mcp.run(transport="streamable-http", host=host, port=port, mount_path=mount_path)


if __name__ == "__main__":
    main()
