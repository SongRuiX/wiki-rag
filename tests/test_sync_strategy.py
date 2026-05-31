import logging
import tempfile
from pathlib import Path
import json
import pytest
from src.sync_strategy import SyncManager
from src.chunker import Chunker
from src.embedder import MockEmbedder
from src.vector_store import MockVectorStore


@pytest.fixture
def sync_mgr():
    return SyncManager(
        chunker=Chunker(max_heading_level=3),
        embedder=MockEmbedder(dimension=128, seed=42),
        vector_store=MockVectorStore(),
    )


def make_files(tmp_path: Path, files: dict[str, str]):
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


class TestSyncManager:
    def test_incremental_sync_new_files(self, sync_mgr, tmp_path):
        make_files(tmp_path, {"a.md": "# A\ncontent A"})
        plan = sync_mgr.sync(str(tmp_path), mode="incremental")
        assert len(plan.new_files) == 1
        assert "a.md" in plan.new_files
        assert plan.total_chunks_estimate > 0

    def test_incremental_sync_detects_updates(self, sync_mgr, tmp_path):
        make_files(tmp_path, {"doc.md": "# V1\noriginal"})
        sync_mgr.sync(str(tmp_path), mode="incremental")

        make_files(tmp_path, {"doc.md": "# V2\nmodified"})
        plan = sync_mgr.sync(str(tmp_path), mode="incremental")
        assert plan.updated_files == ["doc.md"]

    def test_incremental_sync_detects_deletions(self, sync_mgr, tmp_path):
        make_files(tmp_path, {"keep.md": "# Keep\nx", "gone.md": "# Gone\ny"})
        sync_mgr.sync(str(tmp_path), mode="incremental")

        (tmp_path / "gone.md").unlink()
        plan = sync_mgr.sync(str(tmp_path), mode="incremental")
        assert "gone.md" in plan.deleted_files
        assert "keep.md" not in plan.deleted_files

    def test_full_sync(self, sync_mgr, tmp_path):
        make_files(tmp_path, {"doc.md": "# Doc\nhello"})
        plan = sync_mgr.sync(str(tmp_path), mode="full")
        assert len(plan.new_files) == 1

    def test_metadata_file_persisted(self, sync_mgr, tmp_path):
        make_files(tmp_path, {"x.md": "# X\ntext"})
        sync_mgr.sync(str(tmp_path), mode="incremental")
        meta_path = tmp_path / ".wiki_sync_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "x.md" in meta["files"]

    def test_sync_subdirectories(self, sync_mgr, tmp_path):
        make_files(tmp_path, {
            "sub/a.md": "# A\nA",
            "sub/deep/b.md": "# B\nB",
        })
        plan = sync_mgr.sync(str(tmp_path), mode="incremental")
        assert len(plan.new_files) == 2


def test_sync_logs_start_and_completion(caplog, tmp_path):
    """同步应输出 INFO 级别的开始和完成日志。"""
    from src.chunker import Chunker
    from src.embedder import MockEmbedder
    from src.vector_store import MockVectorStore
    from src.sync_strategy import SyncManager

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
