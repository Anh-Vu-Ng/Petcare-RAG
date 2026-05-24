"""
Semantic Cache module cho RAG pipeline.

Sử dụng FAISS để index query embeddings và so sánh cosine similarity.
Nếu tìm thấy query tương tự (similarity > threshold), trả về cached answer
thay vì chạy lại toàn bộ pipeline RAG.
"""

import os
import pickle
import time
import shutil
from typing import Optional, Dict, List, Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.config import CACHE_SIMILARITY_THRESHOLD, CACHE_INDEX_PATH, CACHE_MAX_SIZE


class SemanticCache:
    """
    Semantic Cache sử dụng FAISS để lưu trữ và tìm kiếm query embeddings.
    Khi query mới có cosine similarity >= threshold với query đã cache,
    trả về answer đã lưu thay vì chạy lại pipeline.
    """

    def __init__(
        self,
        embeddings,
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
        index_path: str = CACHE_INDEX_PATH,
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
        self.index_path = index_path
        self.max_size = max_size
        self.metadata_path = os.path.join(index_path, "cache_metadata.pkl")

        # Thống kê
        self.hit_count = 0
        self.miss_count = 0

        # Key: query text, Value: {"answer", "context_docs", "timestamp"}
        self.cache_entries: Dict[str, Dict[str, Any]] = {}
        self.vectorstore = None

        # Load từ disk nếu có
        self._load()

    def _load(self):
        """Load cache từ disk."""
        faiss_path = os.path.join(self.index_path, "index.faiss")
        if os.path.exists(faiss_path) and os.path.exists(self.metadata_path):
            try:
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                with open(self.metadata_path, "rb") as f:
                    self.cache_entries = pickle.load(f)
                print(f"✅ Loaded semantic cache: {len(self.cache_entries)} entries")
            except Exception as e:
                print(f"⚠️ Không thể load cache, khởi tạo mới: {e}")
                self.cache_entries = {}
                self.vectorstore = None

    def save(self):
        """Persist cache xuống disk."""
        if self.vectorstore is not None:
            os.makedirs(self.index_path, exist_ok=True)
            self.vectorstore.save_local(self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.cache_entries, f)

    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Tìm kiếm query trong cache bằng cosine similarity.

        Args:
            query: Standalone query (đã qua rewriter).

        Returns:
            Dict với answer + context_docs nếu cache hit, None nếu miss.
        """
        if not self.vectorstore or not self.cache_entries:
            self.miss_count += 1
            return None

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=1)
        except Exception:
            self.miss_count += 1
            return None

        if not results:
            self.miss_count += 1
            return None

        doc, score = results[0]

        # FAISS IndexFlatL2 trả về L2 distance squared.
        # Với normalized vectors: L2² = 2 * (1 - cosine_similarity)
        # => cosine_similarity = 1 - L2² / 2
        cosine_sim = 1.0 - (score / 2.0)

        if cosine_sim >= self.threshold:
            cached_query = doc.page_content
            if cached_query in self.cache_entries:
                self.hit_count += 1
                entry = self.cache_entries[cached_query]
                return {
                    "answer": entry["answer"],
                    "context_docs": entry["context_docs"],
                    "cached_query": cached_query,
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
        # Nếu query đã tồn tại, update
        if query in self.cache_entries:
            self.cache_entries[query].update({
                "answer": answer,
                "context_docs": context_docs,
                "timestamp": time.time(),
            })
            self.save()
            return

        # Evict entry cũ nhất nếu đầy
        if len(self.cache_entries) >= self.max_size:
            oldest = min(self.cache_entries, key=lambda k: self.cache_entries[k]["timestamp"])
            del self.cache_entries[oldest]
            self._rebuild_index()

        # Thêm entry mới vào FAISS index
        doc = Document(page_content=query)
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents([doc], self.embeddings)
        else:
            self.vectorstore.add_documents([doc])

        self.cache_entries[query] = {
            "answer": answer,
            "context_docs": context_docs,
            "timestamp": time.time(),
        }

        self.save()

    def _rebuild_index(self):
        """Rebuild FAISS index từ các cache entries còn lại."""
        if not self.cache_entries:
            self.vectorstore = None
            return
        docs = [Document(page_content=q) for q in self.cache_entries]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def clear(self):
        """Xóa toàn bộ cache."""
        self.cache_entries = {}
        self.vectorstore = None
        self.hit_count = 0
        self.miss_count = 0
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        print("🗑️ Đã xóa toàn bộ semantic cache.")

    def stats(self) -> Dict[str, Any]:
        """Trả về thống kê cache."""
        total = self.hit_count + self.miss_count
        return {
            "total_entries": len(self.cache_entries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": f"{(self.hit_count / total * 100):.1f}%" if total > 0 else "N/A",
        }
