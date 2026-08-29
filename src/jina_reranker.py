import os
import requests
import httpx
from typing import List, Optional
from langchain_core.documents import Document
from src.config import RERANKER_MODEL, TOP_K_RERANK


class JinaReranker:
    """
    Reranker sử dụng Jina Reranker v2 Base Multilingual API.
    
    Nhận danh sách documents từ Hybrid Retriever, gọi API để re-rank
    và trả về top-k documents có điểm relevance cao nhất.
    """

    def __init__(self, model: str = RERANKER_MODEL, top_k: int = TOP_K_RERANK):
        self.model = model
        self.top_k = top_k
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Chưa thiết lập JINA_API_KEY")
        self.api_url = "https://api.jina.ai/v1/rerank"
        self._async_client: Optional[httpx.AsyncClient] = None

    def _get_async_client(self) -> httpx.AsyncClient:
        """Lấy hoặc khởi tạo persistent AsyncClient với connection pooling."""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
        return self._async_client

    async def aclose(self):
        """Đóng persistent AsyncClient khi dừng ứng dụng."""
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank danh sách documents dựa trên query sử dụng Jina Reranker API (Đồng bộ).
        """
        if not documents:
            return []

        # Chuẩn bị payload cho Jina API
        texts = [doc.page_content for doc in documents]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": self.top_k,
            "return_documents": False,  # Không cần trả lại text, ta đã có sẵn
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[Jina Reranker] Lỗi gọi API: {e}. Trả về documents gốc (không rerank).")
            return documents[:self.top_k]

        # Parse kết quả và map lại về LangChain Documents
        reranked_docs = []
        for item in result.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]

            doc = documents[idx]
            # Gắn thêm rerank_score vào metadata
            doc.metadata["rerank_score"] = score
            reranked_docs.append(doc)

        print(f"[Jina Reranker] Re-ranked {len(documents)} → {len(reranked_docs)} documents.")
        return reranked_docs

    async def arerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank danh sách documents dựa trên query sử dụng Jina Reranker API (Bất đồng bộ).
        """
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": self.top_k,
            "return_documents": False,
        }

        try:
            client = self._get_async_client()
            response = await client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            print(f"[Jina Reranker] Lỗi gọi API async: {e}. Trả về documents gốc (không rerank).")
            return documents[:self.top_k]

        # Parse kết quả và map lại về LangChain Documents
        reranked_docs = []
        for item in result.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]

            doc = documents[idx]
            doc.metadata["rerank_score"] = score
            reranked_docs.append(doc)

        print(f"[Jina Reranker] Async Re-ranked {len(documents)} → {len(reranked_docs)} documents.")
        return reranked_docs

