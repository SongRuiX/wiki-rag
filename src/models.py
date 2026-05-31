from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class Chunk:
    source_sn: str
    source_title: str
    content: str
    section_path: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex)
    embedding: list[float] | None = None


@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    semantic_score: float
    rrf_score: float = 0.0
    keyword_score: float | None = None
    rrf_semantic_rank: int = 0
    rrf_keyword_rank: int | None = None


@dataclass
class SyncPlan:
    new_files: list[str] = field(default_factory=list)
    updated_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    total_chunks_estimate: int = 0
