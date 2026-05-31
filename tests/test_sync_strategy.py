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
