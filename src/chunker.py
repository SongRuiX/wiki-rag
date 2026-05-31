import re
from pathlib import Path
from src.models import Chunk


class Chunker:
    def __init__(self, max_heading_level: int = 3):
        self.max_heading_level = max_heading_level

    def chunk_file(self, filepath: str) -> list[Chunk]:
        content = Path(filepath).read_text(encoding="utf-8")
        source_sn = Path(filepath).stem
        lines = content.splitlines()

        chunks: list[Chunk] = []
        section_path: list[tuple[int, str]] = []
        current_lines: list[str] = []
        source_title = source_sn
        heading_re = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            m = heading_re.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                if level <= self.max_heading_level:
                    text = "\n".join(current_lines).strip()
                    if text and section_path:
                        chunks.append(Chunk(
                            source_sn=source_sn,
                            source_title=source_title,
                            section_path=[s[1] for s in section_path],
                            content=text,
                        ))
                        current_lines = [line]
                    else:
                        current_lines.append(line)
                    while section_path and section_path[-1][0] >= level:
                        section_path.pop()
                    section_path.append((level, title))
                    if level == 1 and source_title == source_sn:
                        source_title = title
                    continue
            if line.strip() or current_lines:
                current_lines.append(line)

        text = "\n".join(current_lines).strip()
        if text:
            chunks.append(Chunk(
                source_sn=source_sn,
                source_title=source_title,
                section_path=[s[1] for s in section_path],
                content=text,
            ))

        return chunks
