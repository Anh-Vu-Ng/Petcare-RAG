"""
Semantic Cache module cho RAG pipeline.

Sử dụng Qdrant để index query embeddings và so sánh cosine similarity.
Nếu tìm thấy query tương tự (similarity > threshold), trả về cached answer
thay vì chạy lại toàn bộ pipeline RAG.
"""

import os
import time
import hashlib
import uuid
from typing import Optional, Dict, List, Any

from qdrant_client import QdrantClient
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
    Semantic Cache sử dụng Qdrant để lưu trữ và tìm kiếm query embeddings.
    Khi query mới có cosine similarity >= threshold với query đã cache,
    trả về answer đã lưu thay vì chạy lại pipeline.
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

        self.embeddings = CacheEmbeddings(embeddings)
        self.threshold = threshold
        self.max_size = max_size
        self.collection_name = QDRANT_CACHE_COLLECTION

        # Thống kê
        self.hit_count = 0
        self.miss_count = 0

        # Khởi tạo Qdrant Client
        if not URL_QDRANT or not QDRANT_API_KEY:
            raise ValueError("Chưa cấu hình URL_QDRANT hoặc QDRANT_API_KEY trong file .env.")
            
        self.client = QdrantClient(url=URL_QDRANT, api_key=QDRANT_API_KEY)
        
        # Đảm bảo collection tồn tại
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

    def save(self):
        """Qdrant Cloud tự động lưu trữ, không cần lưu thủ công xuống disk (No-op)."""
        pass

    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Tìm kiếm query trong cache bằng cosine similarity.

        Args:
            query: Standalone query (đã qua rewriter).

        Returns:
            Dict với answer + context_docs nếu cache hit, None nếu miss.
        """
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
                
                # Deserialization cho context_docs
                context_docs = []
                for doc_dict in payload.get("context_docs", []):
                    context_docs.append(
                        Document(
                            page_content=doc_dict.get("page_content", ""),
                            metadata=doc_dict.get("metadata", {})
                        )
                    )
                    
                return {
                    "answer": payload.get("answer", ""),
                    "context_docs": context_docs,
                    "cached_query": payload.get("query"),
                    "similarity": cosine_sim,
                }

        self.miss_count += 1
        return None

    def store(self, query: str, answer: str, context_docs: List[Document]):
        """
        Lưu query-answer mới vào cache.

        Args:
            query: Standalone query đã qua rewriter.
            answer: Câu trả lời từ LLM.
            context_docs: Danh sách Document đã retrieve được.
        """
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Tạo UUID dạng định danh duy nhất từ nội dung query (tránh trùng lặp)
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
            point_id = str(uuid.UUID(query_hash))
            
            # Kiểm tra xem query này đã có trong cache chưa
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

            # Nếu là query mới, kiểm tra và dọn dẹp cache (Eviction) nếu đầy
            if is_new:
                count_result = self.client.count(collection_name=self.collection_name)
                if count_result.count >= self.max_size:
                    # Scroll lấy các points để lọc ra point cũ nhất
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

            # Chuẩn bị payload để lưu
            payload = {
                "query": query,
                "answer": answer,
                "context_docs": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in context_docs],
                "timestamp": time.time(),
            }

            # Lưu (Upsert) lên Qdrant
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
        except Exception as e:
            print(f"⚠️ Lỗi khi lưu vào cache Qdrant: {e}")

    def _rebuild_index(self):
        """Qdrant hỗ trợ ghi đè trực tiếp, không cần rebuild index (No-op)."""
        pass

    def clear(self):
        """Xóa toàn bộ cache."""
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
            print("🗑️ Đã xóa toàn bộ semantic cache trên Qdrant.")
        except Exception as e:
            print(f"⚠️ Lỗi khi xóa semantic cache: {e}")

    def stats(self) -> Dict[str, Any]:
        """Trả về thống kê cache."""
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

