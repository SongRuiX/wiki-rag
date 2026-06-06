import json
import logging
from abc import ABC, abstractmethod
import numpy as np
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from pymilvus.exceptions import MilvusException
from pymilvus.milvus_client.index import IndexParams
from src.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


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
            logger.info("加载已有 Milvus 集合: %s", self.collection)
            self._ensure_index()
            self.client.load_collection(self.collection)
            return
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="source_sn", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="source_title", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="section_path_str", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            ],
            description="Wiki RAG collection",
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            metric_type="COSINE",
        )
        self._ensure_index()
        self.client.load_collection(self.collection)
        logger.info("创建 Milvus 集合: %s, dimension=%d", self.collection, self.dimension)

    def _ensure_index(self):
        """确保向量字段上有索引，已存在则跳过。"""
        try:
            indexes = self.client.list_indexes(self.collection)
        except MilvusException:
            indexes = []
        if indexes:
            logger.info("索引已存在，跳过创建: %s", indexes)
            return
        index_params = IndexParams()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            nlist=128,
        )
        self.client.create_index(
            collection_name=self.collection,
            index_params=index_params,
        )
        logger.info("创建索引: IVF_FLAT / COSINE / nlist=128")

    def _ensure_loaded(self):
        """确保集合已加载到内存（幂等操作）。"""
        try:
            self.client.load_collection(self.collection)
        except MilvusException:
            logger.warning("加载集合失败，继续尝试操作", exc_info=True)

    def close(self):
        self.client.close()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        self._ensure_loaded()
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
        count = result["upsert_count"]
        logger.info("添加 chunks 到 Milvus: count=%d", count)
        return count

    def delete_by_sn(self, source_sns: list[str]) -> int:
        self._ensure_loaded()
        if not source_sns:
            return 0
        filter_expr = " or ".join(f'source_sn == "{sn}"' for sn in source_sns)
        result = self.client.delete(collection_name=self.collection, filter=filter_expr)
        deleted = len(result) if isinstance(result, list) else 0
        logger.info("从 Milvus 删除 chunks: count=%d", deleted)
        return deleted

    def hybrid_search(
        self, query_vector: list[float], query_text: str,
        top_k: int, semantic_weight: float, keyword_weight: float,
    ) -> list[SearchResult]:
        logger.debug("Milvus 搜索请求: top_k=%d, sem_w=%.2f, kw_w=%.2f", top_k, semantic_weight, keyword_weight)
        if keyword_weight > 0:
            return self._full_hybrid_search(query_vector, query_text, top_k, semantic_weight, keyword_weight)
        return self._semantic_only_search(query_vector, top_k)

    def _semantic_only_search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        self._ensure_loaded()
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
