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
            actual_chunks = self._full_sync(path, md_files)
        else:
            old_meta = self._load_metadata(metadata_path)
            old_files = old_meta.get("files", {})
            plan = SyncPlan(
                new_files=[f for f in md_files if f not in old_files],
                updated_files=[f for f in md_files if f in old_files and md_files[f] != old_files[f]],
                deleted_files=[f for f in old_files if f not in md_files],
            )
            plan.total_chunks_estimate = len(plan.new_files) + len(plan.updated_files)
            actual_chunks = self._execute_plan(path, plan)

        plan.total_chunks_estimate = actual_chunks
        logger.info("变更统计: new=%d, updated=%d, deleted=%d, chunks=%d",
                     len(plan.new_files), len(plan.updated_files),
                     len(plan.deleted_files), actual_chunks)
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

    def _full_sync(self, root: str, md_files: dict[str, str]) -> int:
        return self._process_files(root, list(md_files.keys()))

    def _execute_plan(self, root: str, plan: SyncPlan) -> int:
        changed = plan.new_files + plan.updated_files
        if plan.deleted_files:
            sns = [Path(f).stem for f in plan.deleted_files]
            self.vector_store.delete_by_sn(sns)
        if changed:
            return self._process_files(root, changed)
        return 0

    def _process_files(self, root: str, file_paths: list[str]) -> int:
        if not file_paths:
            return 0
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
        return len(all_chunks)
