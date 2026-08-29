"""
Custom LangChain Embeddings wrapper cho Jina Embeddings API.

Gọi API tại https://api.jina.ai/v1/embeddings để lấy embedding vectors
thay vì chạy model local. Hỗ trợ task-specific LoRA adapters.
"""

import os
import time
import requests
import httpx
import asyncio
from typing import List
from langchain_core.embeddings import Embeddings
from src.config import EMBEDDING_MODEL


class JinaEmbeddings(Embeddings):
    """
    LangChain-compatible Embeddings class sử dụng Jina Embeddings API.
    
    Model: jina-embeddings-v5-text-small (default 1024 dims).
    Hỗ trợ task parameter để kích hoạt LoRA adapters phù hợp:
    - retrieval.query: cho câu truy vấn
    - retrieval.passage: cho passages/documents
    - text-matching: cho so sánh văn bản
    """

    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.api_key = os.getenv("JINA_API_KEY")  # Dùng chung API key Jina
        if not self.api_key:
            raise ValueError("Chưa thiết lập JINA_API_KEY. Kiểm tra lại file .env đi.")
        self.api_url = "https://api.jina.ai/v1/embeddings"
        
        # Bộ nhớ đệm lưu trữ query embeddings để tránh gọi API trùng lặp
        self._query_cache = {}
        self._async_client: Optional[httpx.AsyncClient] = None

    def _get_async_client(self) -> httpx.AsyncClient:
        """Lấy hoặc khởi tạo persistent AsyncClient với connection pooling."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
        return self._async_client

    async def aclose(self):
        """Đóng persistent AsyncClient khi dừng ứng dụng."""
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()

    def _call_api(self, texts: List[str], task: str = "retrieval.passage") -> List[List[float]]:
        """
        Gọi Jina Embeddings API (Đồng bộ).
        """
        if not texts:
            return []

        # So khớp chính xác trong cache nếu là query đơn lẻ
        if len(texts) == 1:
            cache_key = (texts[0], task)
            if cache_key in self._query_cache:
                return [self._query_cache[cache_key]]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "input": texts,
            "task": task,
            "embedding_type": "float",
        }

        for attempt in range(5):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                if response.status_code == 429:
                    if attempt == 4:
                        raise RuntimeError("[Jina Embeddings] Rate limited (429) sau 5 lần thử.")
                    wait_time = (2 ** attempt) + 1
                    print(f"⚠️ [Jina Embeddings] Rate limited (429). Retrying in {wait_time}s (attempt {attempt + 1}/5)...")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                result = response.json()
                break
            except requests.exceptions.RequestException as e:
                if attempt == 4:
                    raise RuntimeError(f"[Jina Embeddings] Lỗi gọi API sau 5 lần thử: {e}")
                wait_time = (2 ** attempt) + 1
                print(f"⚠️ [Jina Embeddings] Lỗi kết nối: {e}. Thử lại sau {wait_time}s (attempt {attempt + 1}/5)...")
                time.sleep(wait_time)

        # Sắp xếp theo index để đảm bảo thứ tự đúng
        data = sorted(result.get("data", []), key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in data]

        # Lưu lại vào cache
        if len(texts) == 1:
            if len(self._query_cache) > 1000:
                self._query_cache.clear()
            self._query_cache[cache_key] = embeddings[0]

        return embeddings

    async def _call_api_async(self, texts: List[str], task: str = "retrieval.passage") -> List[List[float]]:
        """
        Gọi Jina Embeddings API (Bất đồng bộ) sử dụng persistent connection pool.
        """
        if not texts:
            return []

        # So khớp chính xác trong cache nếu là query đơn lẻ
        if len(texts) == 1:
            cache_key = (texts[0], task)
            if cache_key in self._query_cache:
                return [self._query_cache[cache_key]]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "input": texts,
            "task": task,
            "embedding_type": "float",
        }

        client = self._get_async_client()
        result = None
        for attempt in range(5):
            try:
                response = await client.post(self.api_url, headers=headers, json=payload)
                if response.status_code == 429:
                    if attempt == 4:
                        raise RuntimeError("[Jina Embeddings] Rate limited (429) sau 5 lần thử.")
                    wait_time = (2 ** attempt) + 1
                    print(f"⚠️ [Jina Embeddings] Async Rate limited (429). Retrying in {wait_time}s (attempt {attempt + 1}/5)...")
                    await asyncio.sleep(wait_time)
                    continue
                response.raise_for_status()
                result = response.json()
                break
            except Exception as e:
                if attempt == 4:
                    raise RuntimeError(f"[Jina Embeddings] Lỗi gọi API async sau 5 lần thử: {e}")
                wait_time = (2 ** attempt) + 1
                print(f"⚠️ [Jina Embeddings] Lỗi kết nối async: {e}. Thử lại sau {wait_time}s (attempt {attempt + 1}/5)...")
                await asyncio.sleep(wait_time)

        # Sắp xếp theo index để đảm bảo thứ tự đúng
        data = sorted(result.get("data", []), key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in data]

        # Lưu lại vào cache
        if len(texts) == 1:
            if len(self._query_cache) > 1000:
                self._query_cache.clear()
            self._query_cache[cache_key] = embeddings[0]

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed danh sách documents (passages).
        Sử dụng task='retrieval.passage' để tối ưu cho indexing.
        """
        batch_size = 20
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            if i > 0:
                time.sleep(5) 
            batch = texts[i : i + batch_size]
            embeddings = self._call_api(batch, task="retrieval.passage")
            all_embeddings.extend(embeddings)
            print(f"[Jina Embeddings] Embedded batch {i // batch_size + 1} ({len(batch)} texts)")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed một câu truy vấn.
        Sử dụng task='retrieval.query' để tối ưu cho search.
        """
        result = self._call_api([text], task="retrieval.query")
        return result[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed danh sách documents (passages) một cách bất đồng bộ.
        """
        batch_size = 20
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            if i > 0:
                await asyncio.sleep(5) 
            batch = texts[i : i + batch_size]
            embeddings = await self._call_api_async(batch, task="retrieval.passage")
            all_embeddings.extend(embeddings)
            print(f"[Jina Embeddings] Async Embedded batch {i // batch_size + 1} ({len(batch)} texts)")
        return all_embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """
        Embed một câu truy vấn một cách bất đồng bộ.
        """
        result = await self._call_api_async([text], task="retrieval.query")
        return result[0]