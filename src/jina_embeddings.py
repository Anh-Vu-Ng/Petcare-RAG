"""
Custom LangChain Embeddings wrapper cho Jina Embeddings API.

Gọi API tại https://api.jina.ai/v1/embeddings để lấy embedding vectors
thay vì chạy model local. Hỗ trợ task-specific LoRA adapters.
"""

import os
import requests
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

    def _call_api(self, texts: List[str], task: str = "retrieval.passage") -> List[List[float]]:
        """
        Gọi Jina Embeddings API.

        Args:
            texts: Danh sách văn bản cần embedding.
            task: Loại task để kích hoạt LoRA adapter phù hợp.

        Returns:
            Danh sách embedding vectors.
        """
        if not texts:
            return []

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

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[Jina Embeddings] Lỗi gọi API: {e}")

        # Sắp xếp theo index để đảm bảo thứ tự đúng
        data = sorted(result.get("data", []), key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in data]

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed danh sách documents (passages).
        Sử dụng task='retrieval.passage' để tối ưu cho indexing.
        """
        # Jina API giới hạn batch size, chia thành batches nếu cần
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
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
