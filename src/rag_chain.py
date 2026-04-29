import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from src.config import LLM_MODEL, REWRITER_MODEL
from src.prompts import contextualize_q_prompt, qa_prompt
from src.bm25_retriever import get_bm25_retriever
from src.hybrid_retriever import HybridRetriever
from src.vector_store import get_faiss_retriever, get_embeddings
from src.semantic_cache import SemanticCache
from src.jina_reranker import JinaReranker

load_dotenv()

def get_llm():
    """Khởi tạo LLM qua OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Chưa thiết lập OPENROUTER_API_KEY. Kiểm tra lại file .env đi.")
        
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3, 
        max_tokens=512,
        streaming=True
    )


def get_rewriter_llm():
    """Khởi tạo LLM riêng cho Query Rewriter qua Groq (llama-3.1-8b-instant)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Chưa thiết lập GROQ_API_KEY. Kiểm tra lại file .env đi.")

    return ChatOpenAI(
        model=REWRITER_MODEL,
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.1,
        max_tokens=128,
    )


class CachedRAGPipeline:
    """
    Pipeline RAG với Semantic Cache + Jina Reranker.
    
    Luồng xử lý:
    1. Query Rewriter (Groq llama-3.1-8b-instant): Viết lại câu hỏi thành standalone query
    2. Semantic Cache Lookup: Kiểm tra cache với standalone query
    3. (Cache Miss) Hybrid Retrieval: FAISS + BM25 + RRF
    4. (Cache Miss) Jina Reranker: Re-rank documents theo relevance score
    5. (Cache Miss) QA Generation (OpenRouter): LLM sinh câu trả lời
    6. (Cache Miss) Cache Store: Lưu kết quả vào cache
    """

    def __init__(self, llm, rewriter_llm, hybrid_retriever, reranker, semantic_cache, qa_chain):
        self.llm = llm
        self.rewriter_llm = rewriter_llm
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.semantic_cache = semantic_cache
        self.qa_chain = qa_chain
        # Chain để rewrite query: contextualize prompt -> Groq LLM -> string
        self.rewrite_chain = contextualize_q_prompt | self.rewriter_llm

    def _rewrite_query(self, query: str, chat_history: list) -> str:
        """
        Viết lại câu hỏi thành standalone query dựa trên lịch sử hội thoại.
        Nếu không có lịch sử, trả về query gốc (bỏ qua LLM call).
        """
        if not chat_history:
            return query
        
        response = self.rewrite_chain.invoke({
            "chat_history": chat_history,
            "input": query,
        })
        return response.content

    def invoke(self, inputs: dict) -> dict:
        """
        Chạy pipeline RAG với semantic cache.

        Args:
            inputs: {"input": str, "chat_history": list}

        Returns:
            {
                "answer": str,
                "context": List[Document],
                "from_cache": bool,
                "standalone_query": str,
                "similarity": float (nếu from_cache=True),
                "timing": dict  (thời gian từng bước, đơn vị giây)
            }
        """
        query = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])
        timing = {}

        # --- Step 1: Rewrite query ---
        t0 = time.time()
        standalone_query = self._rewrite_query(query, chat_history)
        timing["rewrite_query"] = time.time() - t0

        # --- Step 2: Check semantic cache ---
        t0 = time.time()
        cached = self.semantic_cache.lookup(standalone_query)
        timing["cache_lookup"] = time.time() - t0

        if cached:
            return {
                "answer": cached["answer"],
                "context": cached["context_docs"],
                "from_cache": True,
                "standalone_query": standalone_query,
                "similarity": cached["similarity"],
                "timing": timing,
            }

        # --- Step 3: Hybrid Retrieval (cache miss) ---
        t0 = time.time()
        docs = self.hybrid_retriever.invoke(standalone_query)
        timing["hybrid_retrieval"] = time.time() - t0

        # --- Step 4: Jina Reranker ---
        t0 = time.time()
        docs = self.reranker.rerank(standalone_query, docs)
        timing["jina_reranker"] = time.time() - t0

        # --- Step 5: QA Generation ---
        t0 = time.time()
        answer = self.qa_chain.invoke({
            "context": docs,
            "chat_history": chat_history,
            "input": standalone_query,
        })
        timing["qa_generation"] = time.time() - t0

        # --- Step 6: Store vào cache ---
        t0 = time.time()
        self.semantic_cache.store(standalone_query, answer, docs)
        timing["cache_store"] = time.time() - t0

        return {
            "answer": answer,
            "context": docs,
            "from_cache": False,
            "standalone_query": standalone_query,
            "timing": timing,
        }


def build_conversational_rag_chain():
    """Khởi tạo toàn bộ pipeline cho Conversational RAG với Semantic Cache + Jina Reranker."""
    llm = get_llm()
    rewriter_llm = get_rewriter_llm()
    
    # 1. Khởi tạo các retrievers
    dense_retriever = get_faiss_retriever()
    sparse_retriever = get_bm25_retriever()
    hybrid_retriever = HybridRetriever(dense_retriever=dense_retriever, sparse_retriever=sparse_retriever)
    
    # 2. Khởi tạo Jina Reranker
    reranker = JinaReranker()
    
    # 3. Khởi tạo Semantic Cache (dùng chung embedding model)
    embeddings = get_embeddings()
    semantic_cache = SemanticCache(embeddings)
    
    # 4. Khởi tạo QA chain (stuff documents -> LLM)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 5. Kết hợp tất cả vào CachedRAGPipeline
    pipeline = CachedRAGPipeline(llm, rewriter_llm, hybrid_retriever, reranker, semantic_cache, qa_chain)
    
    return pipeline
