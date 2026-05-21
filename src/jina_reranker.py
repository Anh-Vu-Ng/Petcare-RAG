import os
import requests
from typing import List
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

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank danh sách documents dựa trên query sử dụng Jina Reranker API.

        Args:
            query: Câu truy vấn (standalone query sau khi rewrite).
            documents: Danh sách LangChain Documents từ Hybrid Retriever.

        Returns:
            Danh sách documents đã được re-rank, giới hạn top_k,
            với metadata bổ sung 'rerank_score'.
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
