import tempfile
from pathlib import Path
import pytest
from src.chunker import Chunker


def make_md(dir_path: Path, name: str, content: str) -> str:
    p = dir_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestChunker:
    def test_single_h1_heading(self, tmp_dir):
        fp = make_md(tmp_dir, "doc.md", "# Intro\nhello world")
        chunker = Chunker()
        chunks = chunker.chunk_file(fp)
        assert len(chunks) == 1
        assert chunks[0].source_title == "Intro"
        assert chunks[0].section_path == ["Intro"]
        assert chunks[0].source_sn == "doc"
        assert chunks[0].content.startswith("# Intro")

    def test_no_headings_whole_file_one_chunk(self, tmp_dir):
        fp = make_md(tmp_dir, "plain.md", "just text\nmore text")
        chunker = Chunker()
        chunks = chunker.chunk_file(fp)
        assert len(chunks) == 1
        assert chunks[0].source_title == "plain"
        assert chunks[0].section_path == []
        assert "just text" in chunks[0].content

    def test_multilevel_headings(self, tmp_dir):
        content = "# A\ncontent A\n## A.1\ncontent A1\n## A.2\ncontent A2\n# B\ncontent B"
        fp = make_md(tmp_dir, "multi.md", content)
        chunker = Chunker()
        chunks = chunker.chunk_file(fp)
        assert len(chunks) == 4
        assert chunks[0].section_path == ["A"]
        assert chunks[1].section_path == ["A", "A.1"]
        assert chunks[2].section_path == ["A", "A.2"]
        assert chunks[3].section_path == ["B"]

    def test_headings_beyond_max_level_treated_as_content(self, tmp_dir):
        content = "# H1\n## H2\n### H3\n#### H4 deep\nstill content"
        fp = make_md(tmp_dir, "deep.md", content)
        chunker = Chunker(max_heading_level=3)
        chunks = chunker.chunk_file(fp)
        assert len(chunks) == 3  # H1, H2, H3 each start a chunk; H4 is content

    def test_h1_becomes_source_title(self, tmp_dir):
        fp = make_md(tmp_dir, "t.md", "# Real Title\ncontent")
        chunker = Chunker()
        chunks = chunker.chunk_file(fp)
        assert chunks[0].source_title == "Real Title"

    def test_empty_file(self, tmp_dir):
        fp = make_md(tmp_dir, "empty.md", "")
        chunker = Chunker()
        chunks = chunker.chunk_file(fp)
        assert len(chunks) == 0

    def test_max_heading_level_configurable(self, tmp_dir):
        content = "# A\n## B\n### C\n#### D"
        fp = make_md(tmp_dir, "cfg.md", content)
        chunker = Chunker(max_heading_level=2)
        chunks = chunker.chunk_file(fp)
        assert len(chunks) == 2  # only H1 and H2 create chunks

    def test_content_before_first_heading(self, tmp_dir):
        content = "preamble\n# Title\nbody"
        fp = make_md(tmp_dir, "preamble.md", content)
        chunker = Chunker()
        chunks = chunker.chunk_file(fp)
        assert len(chunks) == 1
        assert "preamble" in chunks[0].content
        assert "Title" in chunks[0].content
