"""
Semantic Cache module cho RAG pipeline.

Sử dụng Qdrant để index query embeddings và so sánh cosine similarity.
Hỗ trợ bộ đệm 2 lớp: Lớp 1 (Exact Match In-Memory) và Lớp 2 (Semantic Match Qdrant).
"""

import os
import time
import hashlib
import uuid
import re
from typing import Optional, Dict, List, Any

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.config import (
    URL_QDRANT,
    QDRANT_API_KEY,
    QDRANT_CACHE_COLLECTION,
    EMBEDDING_DIM,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_MAX_SIZE
)


class SemanticCache:
    """
    Semantic Cache sử dụng bộ đệm 2 lớp (2-Tier Caching):
    - Lớp 1: Exact Match Cache (In-Memory Bounded Dict) -> Phản hồi <1ms, 0 API calls.
    - Lớp 2: Semantic Match Cache (Qdrant Vector DB) -> So khớp cosine similarity.
    """

    def __init__(
        self,
        embeddings,
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
        max_size: int = CACHE_MAX_SIZE,
    ):
        # Đồng nhất task embedding cho cache để tránh lệch phân phối giữa query và passage
        class CacheEmbeddings(Embeddings):
            def __init__(self, base_embeddings):
                self.base_embeddings = base_embeddings
                
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                if hasattr(self.base_embeddings, "_call_api"):
                    return self.base_embeddings._call_api(texts, task="text-matching")
                return self.base_embeddings.embed_documents(texts)
                
            def embed_query(self, text: str) -> List[float]:
                if hasattr(self.base_embeddings, "_call_api"):
                    result = self.base_embeddings._call_api([text], task="text-matching")
                    return result[0]
                return self.base_embeddings.embed_query(text)

            async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
                if hasattr(self.base_embeddings, "_call_api_async"):
                    return await self.base_embeddings._call_api_async(texts, task="text-matching")
                return await self.base_embeddings.aembed_documents(texts)

            async def aembed_query(self, text: str) -> List[float]:
                if hasattr(self.base_embeddings, "_call_api_async"):
                    result = await self.base_embeddings._call_api_async([text], task="text-matching")
                    return result[0]
                return await self.base_embeddings.aembed_query(text)

        self.embeddings = CacheEmbeddings(embeddings)
        self.threshold = threshold
        self.max_size = max_size
        self.collection_name = QDRANT_CACHE_COLLECTION

        # Thống kê
        self.hit_count = 0
        self.miss_count = 0

        # Lớp 1: Exact Match Cache (In-Memory Bounded Dict)
        self._exact_cache = {}

        # Khởi tạo Qdrant Clients
        if not URL_QDRANT or not QDRANT_API_KEY:
            raise ValueError("Chưa cấu hình URL_QDRANT hoặc QDRANT_API_KEY trong file .env.")
            
        self.client = QdrantClient(url=URL_QDRANT, api_key=QDRANT_API_KEY)
        self.async_client = AsyncQdrantClient(url=URL_QDRANT, api_key=QDRANT_API_KEY)
        
        # Đảm bảo collection tồn tại (thực hiện đồng bộ lúc khởi động)
        self._ensure_collection()

    def _ensure_collection(self):
        """Kiểm tra và tạo collection cho cache nếu chưa có."""
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            print(f"ℹ️ Creating Qdrant collection '{self.collection_name}' for cache...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE
                )
            )

    def _normalize_query(self, query: str) -> str:
        """Chuẩn hóa query (lowercase, xóa ký tự đặc biệt, trim) để tối ưu Exact Match."""
        cleaned = query.strip().lower()
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        return " ".join(cleaned.split())

    def save(self):
        """No-op."""
        pass

    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Tìm kiếm query trong cache bằng bộ đệm 2 lớp (Đồng bộ).
        """
        norm_query = self._normalize_query(query)

        # --- LỚP 1: Exact Match Cache Lookup ---
        if norm_query in self._exact_cache:
            self.hit_count += 1
            print(f"⚡ [SemanticCache] Tier 1 HIT (Exact Match): '{query[:30]}...'")
            return self._exact_cache[norm_query]

        # --- LỚP 2: Semantic Match Cache Lookup ---
        try:
            query_vector = self.embeddings.embed_query(query)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=1,
            )
        except Exception as e:
            print(f"⚠️ Lỗi tìm kiếm trong cache Qdrant: {e}")
            self.miss_count += 1
            return None

        if not results or not results.points:
            self.miss_count += 1
            return None

        point = results.points[0]
        cosine_sim = point.score

        if cosine_sim >= self.threshold:
            payload = point.payload
            if payload and "query" in payload:
                self.hit_count += 1
                print(f"⚡ [SemanticCache] Tier 2 HIT (Similarity={cosine_sim:.3f}): '{query[:30]}...'")
                
                # Deserialization cho context_docs
                context_docs = []
                for doc_dict in payload.get("context_docs", []):
                    context_docs.append(
                        Document(
                            page_content=doc_dict.get("page_content", ""),
                            metadata=doc_dict.get("metadata", {})
                        )
                    )
                
                cache_result = {
                    "answer": payload.get("answer", ""),
                    "context_docs": context_docs,
                    "cached_query": payload.get("query"),
                    "similarity": cosine_sim,
                }

                # Cập nhật ngược lại Lớp 1 Exact Match Cache
                if len(self._exact_cache) > 1000:
                    self._exact_cache.clear()
                self._exact_cache[norm_query] = cache_result
                
                return cache_result

        self.miss_count += 1
        return None

    async def alookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Tìm kiếm query trong cache bằng bộ đệm 2 lớp (Bất đồng bộ).
        """
        norm_query = self._normalize_query(query)

        # --- LỚP 1: Exact Match Cache Lookup ---
        if norm_query in self._exact_cache:
            self.hit_count += 1
            print(f"⚡ [SemanticCache] Tier 1 HIT (Exact Match - Async): '{query[:30]}...'")
            return self._exact_cache[norm_query]

        # --- LỚP 2: Semantic Match Cache Lookup ---
        try:
            query_vector = await self.embeddings.aembed_query(query)
            results = await self.async_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=1,
            )
        except Exception as e:
            print(f"⚠️ Lỗi tìm kiếm trong cache Qdrant async: {e}")
            self.miss_count += 1
            return None

        if not results or not results.points:
            self.miss_count += 1
            return None

        point = results.points[0]
        cosine_sim = point.score

        if cosine_sim >= self.threshold:
            payload = point.payload
            if payload and "query" in payload:
                self.hit_count += 1
                print(f"⚡ [SemanticCache] Tier 2 HIT (Similarity={cosine_sim:.3f} - Async): '{query[:30]}...'")
                
                context_docs = []
                for doc_dict in payload.get("context_docs", []):
                    context_docs.append(
                        Document(
                            page_content=doc_dict.get("page_content", ""),
                            metadata=doc_dict.get("metadata", {})
                        )
                    )
                
                cache_result = {
                    "answer": payload.get("answer", ""),
                    "context_docs": context_docs,
                    "cached_query": payload.get("query"),
                    "similarity": cosine_sim,
                }

                # Cập nhật ngược lại Lớp 1 Exact Match Cache
                if len(self._exact_cache) > 1000:
                    self._exact_cache.clear()
                self._exact_cache[norm_query] = cache_result
                
                return cache_result

        self.miss_count += 1
        return None

    def store(self, query: str, answer: str, context_docs: List[Document]):
        """
        Lưu query-answer mới vào cache (Đồng bộ).
        """
        try:
            query_vector = self.embeddings.embed_query(query)
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
            point_id = str(uuid.UUID(query_hash))
            
            is_new = True
            try:
                existing = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_payload=False,
                )
                if existing:
                    is_new = False
            except Exception:
                pass

            if is_new:
                count_result = self.client.count(collection_name=self.collection_name)
                if count_result.count >= self.max_size:
                    records, _ = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=self.max_size,
                        with_payload=True,
                        with_vectors=False,
                    )
                    records_with_ts = [r for r in records if r.payload and "timestamp" in r.payload]
                    if records_with_ts:
                        oldest = min(records_with_ts, key=lambda r: r.payload["timestamp"])
                        self.client.delete(
                            collection_name=self.collection_name,
                            points_selector=models.PointIdsList(
                                points=[oldest.id]
                            )
                        )
                        print(f"🗑️ Evicted oldest cache entry with ID {oldest.id} (query: {oldest.payload.get('query')[:30]}...)")

            payload = {
                "query": query,
                "answer": answer,
                "context_docs": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in context_docs],
                "timestamp": time.time(),
            }

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=query_vector,
                        payload=payload
                    )
                ]
            )

            # Lưu vào Lớp 1 Exact Match Cache
            norm_query = self._normalize_query(query)
            if len(self._exact_cache) > 1000:
                self._exact_cache.clear()
            self._exact_cache[norm_query] = {
                "answer": answer,
                "context_docs": context_docs,
                "cached_query": query,
                "similarity": 1.0,
            }
        except Exception as e:
            print(f"⚠️ Lỗi khi lưu vào cache Qdrant: {e}")

    async def astore(self, query: str, answer: str, context_docs: List[Document]):
        """
        Lưu query-answer mới vào cache (Bất đồng bộ).
        """
        try:
            query_vector = await self.embeddings.aembed_query(query)
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
            point_id = str(uuid.UUID(query_hash))
            
            is_new = True
            try:
                existing = await self.async_client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_payload=False,
                )
                if existing:
                    is_new = False
            except Exception:
                pass

            if is_new:
                count_result = await self.async_client.count(collection_name=self.collection_name)
                if count_result.count >= self.max_size:
                    records, _ = await self.async_client.scroll(
                        collection_name=self.collection_name,
                        limit=self.max_size,
                        with_payload=True,
                        with_vectors=False,
                    )
                    records_with_ts = [r for r in records if r.payload and "timestamp" in r.payload]
                    if records_with_ts:
                        oldest = min(records_with_ts, key=lambda r: r.payload["timestamp"])
                        await self.async_client.delete(
                            collection_name=self.collection_name,
                            points_selector=models.PointIdsList(
                                points=[oldest.id]
                            )
                        )
                        print(f"🗑️ Evicted oldest cache entry async with ID {oldest.id} (query: {oldest.payload.get('query')[:30]}...)")

            payload = {
                "query": query,
                "answer": answer,
                "context_docs": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in context_docs],
                "timestamp": time.time(),
            }

            await self.async_client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=query_vector,
                        payload=payload
                    )
                ]
            )

            # Lưu vào Lớp 1 Exact Match Cache
            norm_query = self._normalize_query(query)
            if len(self._exact_cache) > 1000:
                self._exact_cache.clear()
            self._exact_cache[norm_query] = {
                "answer": answer,
                "context_docs": context_docs,
                "cached_query": query,
                "similarity": 1.0,
            }
        except Exception as e:
            print(f"⚠️ Lỗi khi lưu vào cache Qdrant async: {e}")

    def _rebuild_index(self):
        """No-op."""
        pass

    def clear(self):
        """Xóa toàn bộ cache (Đồng bộ)."""
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE
                )
            )
            self.hit_count = 0
            self.miss_count = 0
            self._exact_cache.clear()
            print("🗑️ Đã xóa toàn bộ semantic cache trên Qdrant và Exact Cache.")
        except Exception as e:
            print(f"⚠️ Lỗi khi xóa semantic cache: {e}")

    async def aclear(self):
        """Xóa toàn bộ cache (Bất đồng bộ)."""
        try:
            await self.async_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE
                )
            )
            self.hit_count = 0
            self.miss_count = 0
            self._exact_cache.clear()
            print("🗑️ Đã xóa toàn bộ semantic cache trên Qdrant và Exact Cache async.")
        except Exception as e:
            print(f"⚠️ Lỗi khi xóa semantic cache async: {e}")

    def stats(self) -> Dict[str, Any]:
        """Thống kê cache (Đồng bộ)."""
        try:
            count_result = self.client.count(collection_name=self.collection_name)
            total_entries = count_result.count
        except Exception:
            total_entries = 0
            
        total = self.hit_count + self.miss_count
        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": f"{(self.hit_count / total * 100):.1f}%" if total > 0 else "N/A",
        }

    async def astats(self) -> Dict[str, Any]:
        """Thống kê cache (Bất đồng bộ)."""
        try:
            count_result = await self.async_client.count(collection_name=self.collection_name)
            total_entries = count_result.count
        except Exception:
            total_entries = 0
            
        total = self.hit_count + self.miss_count
        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": f"{(self.hit_count / total * 100):.1f}%" if total > 0 else "N/A",
        }


