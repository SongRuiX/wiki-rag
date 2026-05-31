import json
from abc import ABC, abstractmethod
import numpy as np
from pymilvus import MilvusClient
from src.models import Chunk, SearchResult


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk]) -> int:
        ...

    @abstractmethod
    def delete_by_sn(self, source_sns: list[str]) -> int:
        ...

    @abstractmethod
    def hybrid_search(
        self, query_vector: list[float], query_text: str,
        top_k: int, semantic_weight: float, keyword_weight: float,
    ) -> list[SearchResult]:
        ...


class MockVectorStore(VectorStore):
    def __init__(self):
        self._chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]) -> int:
        self._chunks.extend(chunks)
        return len(chunks)

    def delete_by_sn(self, source_sns: list[str]) -> int:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.source_sn not in source_sns]
        return before - len(self._chunks)

    def hybrid_search(
        self, query_vector: list[float], query_text: str,
        top_k: int, semantic_weight: float, keyword_weight: float,
    ) -> list[SearchResult]:
        qv = np.array(query_vector)
        scored = []
        for i, chunk in enumerate(self._chunks):
            if chunk.embedding is None:
                continue
            cv = np.array(chunk.embedding)
            sim = float(np.dot(qv, cv) / (np.linalg.norm(qv) * np.linalg.norm(cv) + 1e-8))
            rrf = 1.0 / (60 + i + 1)
            scored.append(SearchResult(
                chunk=chunk,
                semantic_score=sim,
                rrf_score=semantic_weight * rrf + keyword_weight * 0,
                rrf_semantic_rank=i + 1,
            ))
        scored.sort(key=lambda r: r.rrf_score, reverse=True)
        return scored[:top_k]


class MilvusVectorStore(VectorStore):
    def __init__(self, uri: str, collection: str = "wiki_rag", dimension: int = 2560):
        self.client = MilvusClient(uri=uri)
        self.collection = collection
        self.dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.has_collection(self.collection):
            self.client.load_collection(self.collection)
            return
        self.client.create_collection(
            collection_name=self.collection,
            dimension=self.dimension,
            metric_type="COSINE",
            id_type="string",
        )

    def close(self):
        self.client.close()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        data = []
        for c in chunks:
            if c.embedding is None:
                continue
            data.append({
                "id": c.id,
                "vector": c.embedding,
                "source_sn": c.source_sn,
                "source_title": c.source_title,
                "section_path_str": json.dumps(c.section_path, ensure_ascii=False),
                "content": c.content,
            })
        if not data:
            return 0
        result = self.client.upsert(collection_name=self.collection, data=data)
        return result["upsert_count"]

    def delete_by_sn(self, source_sns: list[str]) -> int:
        if not source_sns:
            return 0
        filter_expr = " or ".join(f'source_sn == "{sn}"' for sn in source_sns)
        result = self.client.delete(collection_name=self.collection, filter=filter_expr)
        return len(result) if isinstance(result, list) else 0

    def hybrid_search(
        self, query_vector: list[float], query_text: str,
        top_k: int, semantic_weight: float, keyword_weight: float,
    ) -> list[SearchResult]:
        if keyword_weight > 0:
            return self._full_hybrid_search(query_vector, query_text, top_k, semantic_weight, keyword_weight)
        return self._semantic_only_search(query_vector, top_k)

    def _semantic_only_search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        resp = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=top_k,
            output_fields=["source_sn", "source_title", "section_path_str", "content"],
        )
        results = []
        for i, hit in enumerate(resp[0]):
            entity = hit["entity"]
            chunk = Chunk(
                id=str(hit["id"]),
                source_sn=entity["source_sn"],
                source_title=entity["source_title"],
                section_path=json.loads(entity.get("section_path_str", "[]")),
                content=entity["content"],
            )
            rrf = 1.0 / (60 + i + 1)
            results.append(SearchResult(
                chunk=chunk,
                semantic_score=hit["distance"],
                rrf_score=rrf,
                rrf_semantic_rank=i + 1,
            ))
        return results

    def _full_hybrid_search(self, query_vector, query_text, top_k, semantic_weight, keyword_weight):
        raise NotImplementedError("BM25 hybrid search not yet implemented")
