import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from src.config import LLM_MODEL, ROUTER_MODEL, GREETINGS_MODEL, CSV_PRICING_PATH, REWRITE_MODEL
from src.prompts import contextualize_q_prompt, qa_prompt, tool_qa_prompt, greetings_prompt
from src.bm25_retriever import get_bm25_retriever
from src.hybrid_retriever import HybridRetriever
from src.vector_store import get_faiss_retriever, get_embeddings
from src.semantic_cache import SemanticCache
from src.jina_reranker import JinaReranker
from src.intent_router import IntentRouter
from src.service_db import ServiceDB
from src.tools import lookup_service_price, calculate_final_price, format_final_price_for_llm, _resolve_service_type, _resolve_all_service_types

load_dotenv()

def get_llm():
    """Khởi tạo LLM qua OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Chưa thiết lập OPENROUTER_API_KEY. Kiểm tra lại file .env đi.")
        
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0, 
        top_p= 0.7,
        presence_penalty=0.0,
        frequency_penalty= 0.0,
        max_tokens=768,
        streaming=True,
        timeout=30,
        extra_body={
            "provider": {
                "only": ["novita"] 
            }
        }
    )


def get_router_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Chưa thiết lập OPENROUTER_API_KEY. Kiểm tra lại file .env đi.")

    return ChatOpenAI(
        model=ROUTER_MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        top_p = 0.1,
        presence_penalty=0.0,
        frequency_penalty= 0.0,
        max_tokens=10,
        timeout=20,
        extra_body={
            "reasoning": {"enabled": False}
        }

    )


def get_greetings_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Chưa thiết lập OPENROUTER_API_KEY. Kiểm tra lại file .env đi.")
        
    return ChatOpenAI(
        model=GREETINGS_MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2, 
        top_p = 0.7,
        max_tokens=256,
        timeout=15
    )


def get_rewrite_llm():
    """Khởi tạo LLM cho Query Rewriter qua OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Chưa thiết lập OPENROUTER_API_KEY. Kiểm tra lại file .env đi.")
        
    return ChatOpenAI(
        model=REWRITE_MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0, 
        top_p=0.7,
        max_tokens=256,
        timeout=20,
        extra_body={
            "provider": {
                "only": ["phala"] 
            }
        }
    )


import re

def _is_greeting_fast(query: str) -> bool:
    """
    Kiểm tra nhanh xem câu truy vấn có phải là lời chào xã giao bằng regex hay không.
    """
    cleaned = query.strip().lower()
    cleaned = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', cleaned).strip()
    
    greeting_pattern = (
        r"^(xin\s+)?chào(\s+(bạn|shop|ad|admin|assistant|ad_min|mọi\s+người|cả\s+nhà))?$|"
        r"^(hi|hello|helo|halo|hế|bạn\s+lô|hey|alo|ê|lô|ơi)(\s+(bạn|shop|ad|admin|assistant))?$"
    )
    return bool(re.match(greeting_pattern, cleaned))


GREETING_RESPONSE = (
    "Petcare Assistant chào bạn! 😊 Chúng mình có các dịch vụ Grooming và Boarding đã được niêm yết giá cụ thể:\n"
    "+ Cạo lông\n"
    "+ Cắt mài móng\n"
    "+ Lưu trú 24h(gửi thú cưng)\n"
    "+ Nặn tuyến hôi\n"
    "+ Tắm\n"
    "+ Vệ sinh tai\n"
    "\nBạn muốn biết giá của dịch vụ nào không? Hãy cho mình biết nhé! 🌟"
)


class AgenticRAGPipeline:
    """
    Pipeline Agentic RAG với Selective Caching + Intent Router.
    
    Luồng xử lý:
    1. Query Rewriter: Viết lại câu hỏi thành standalone query
    2. Intent Router: Phân loại KNOWLEDGE hoặc TOOL
    3a. KNOWLEDGE path:
        - Semantic Cache Lookup
        - (Cache Miss) Hybrid Retrieval: FAISS + BM25 + RRF
        - (Cache Miss) Jina Reranker
        - (Cache Miss) QA Generation
        - (Cache Miss) Cache Store
    3b. TOOL path (bypass cache):
        - lookup_service_price → Supabase
        - calculate_final_price (nếu lưu trú)
        - LLM format câu trả lời
    """

    def __init__(
        self,
        llm,
        hybrid_retriever,
        reranker,
        semantic_cache,
        qa_chain,
        router,
        service_db,
        greetings_llm,
        rewrite_llm,
    ):
        self.llm = llm
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.semantic_cache = semantic_cache
        self.qa_chain = qa_chain
        self.router = router
        self.service_db = service_db
        # Chain để rewrite query (dùng rewrite_llm)
        self.rewrite_chain = contextualize_q_prompt | rewrite_llm
        # Chain để trả lời dựa trên price data
        self.tool_qa_chain = tool_qa_prompt | self.llm
        # Chain để trả lời greetings (dùng gemma-3-4b-it)
        self.greetings_chain = greetings_prompt | greetings_llm

        # Load parent documents mapping
        from src.config import PARENT_DOCS_PATH
        import pickle
        if os.path.exists(PARENT_DOCS_PATH):
            print(f"Loading parent documents mapping from {PARENT_DOCS_PATH}...")
            with open(PARENT_DOCS_PATH, "rb") as f:
                self.parent_docs = pickle.load(f)
        else:
            print("Warning: parent_docs.pkl not found. Parent expansion will not work.")
            self.parent_docs = {}

    def _expand_to_parent(self, child_docs: list) -> list:
        """
        Map child documents to their parent documents using parent_id.
        Deduplicates parent documents if multiple children point to the same parent.
        """
        if not self.parent_docs:
            print("Warning: self.parent_docs is empty. Returning original child documents.")
            return child_docs
            
        parent_docs_list = []
        seen_parent_ids = set()
        
        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id in self.parent_docs:
                if parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    
                    # Tạo bản sao của parent document để tránh chỉnh sửa tham chiếu gốc,
                    # đồng thời copy lại các metadata động từ child document (như điểm số rerank, rrf)
                    from langchain_core.documents import Document
                    orig_parent = self.parent_docs[parent_id]
                    parent_meta = orig_parent.metadata.copy()
                    
                    if "rerank_score" in doc.metadata:
                        parent_meta["rerank_score"] = doc.metadata["rerank_score"]
                    if "rrf_score" in doc.metadata:
                        parent_meta["rrf_score"] = doc.metadata["rrf_score"]
                        
                    parent_doc = Document(
                        page_content=orig_parent.page_content,
                        metadata=parent_meta
                    )
                    parent_docs_list.append(parent_doc)
            else:
                # Nếu không tìm thấy parent_id, giữ lại child doc làm phương án dự phòng
                parent_docs_list.append(doc)
                
        return parent_docs_list

    def _rewrite_query(self, query: str, chat_history: list) -> str:
        """
        Viết lại câu hỏi thành standalone query dựa trên lịch sử hội thoại.
        Nếu không có lịch sử, trả về query gốc (bỏ qua LLM call).
        """
        if not chat_history:
            return query
        
        try:
            response = self.rewrite_chain.invoke({
                "chat_history": chat_history,
                "input": query,
            })
            rewritten = response.content.strip()
            if rewritten:
                return rewritten
            return query
        except Exception as e:
            print(f"[Query Rewriter] Lỗi rewrite query, fallback query gốc: {e}")
            return query

    def _handle_knowledge(self, standalone_query: str, chat_history: list, timing: dict) -> dict:
        """
        Xử lý nhánh KNOWLEDGE: Semantic Cache → Hybrid RAG.
        """
        # --- Semantic Cache lookup ---
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
                "intent": "KNOWLEDGE",
                "timing": timing,
            }

        # --- Hybrid Retrieval (cache miss) ---
        t0 = time.time()
        docs = self.hybrid_retriever.invoke(standalone_query)
        timing["hybrid_retrieval"] = time.time() - t0

        # --- Jina Reranker ---
        t0 = time.time()
        docs = self.reranker.rerank(standalone_query, docs)
        timing["jina_reranker"] = time.time() - t0

        # --- Parent Document Expansion ---
        docs = self._expand_to_parent(docs)

        # --- QA Generation ---
        t0 = time.time()
        answer = self.qa_chain.invoke({
            "context": docs,
            "chat_history": chat_history,
            "input": standalone_query,
        })
        timing["qa_generation"] = time.time() - t0

        # --- Store vào cache ---
        t0 = time.time()
        self.semantic_cache.store(standalone_query, answer, docs)
        timing["cache_store"] = time.time() - t0

        return {
            "answer": answer,
            "context": docs,
            "from_cache": False,
            "standalone_query": standalone_query,
            "intent": "KNOWLEDGE",
            "timing": timing,
        }

    def _handle_tool(self, standalone_query: str, chat_history: list, timing: dict) -> dict:
        """
        Xử lý nhánh TOOL: Bypass cache → Supabase lookup → calculate_final_price → LLM.
        """
        t0 = time.time()

        # --- Trích xuất thông tin từ query ---
        resolved_types = _resolve_all_service_types(standalone_query)
        weights = self._extract_all_numbers(standalone_query, ["kg", "kí", "ký", "cân", "ki", "can"])
        days_list = self._extract_all_numbers(standalone_query, ["ngày", "ngay", "hôm", "đêm", "hom", "dem"])

        # --- Bước 1: Tra giá gốc từ Supabase ---
        all_price_data = []

        if not resolved_types:
            # Fallback nếu không tìm thấy dịch vụ nào cụ thể
            price_data = lookup_service_price(
                db=self.service_db,
                query=standalone_query,
            )
            all_price_data.append(price_data)
        else:
            for i, stype in enumerate(resolved_types):
                weight = None
                if len(weights) == 1:
                    weight = weights[0]
                elif i < len(weights):
                    weight = weights[i]

                res = lookup_service_price(
                    db=self.service_db,
                    query=standalone_query,
                    service_type=stype,
                    weight_kg=weight,
                )

                # --- Bước 2: Nếu là lưu trú, thử tính giá cuối cùng ---
                if stype == "luu_tru_24h" and weight:
                    num_days = days_list[0] if days_list else None
                    base_info = self.service_db.lookup_price("luu_tru_24h", weight)
                    
                    if base_info and num_days:
                        final_price_result = calculate_final_price(
                            base_price_per_day=base_info["price"],
                            num_days=num_days,
                            service_type="luu_tru_24h",
                            weight_kg=weight,
                            db=self.service_db,
                        )
                        final_price_info = format_final_price_for_llm(final_price_result)
                        res = res + "\n\n" + final_price_info
                
                all_price_data.append(res)
        
        price_data = "\n\n---\n\n".join(all_price_data)

        timing["tool_lookup"] = time.time() - t0

        # --- Bước 3: LLM format câu trả lời ---
        t0 = time.time()
        response = self.tool_qa_chain.invoke({
            "price_data": price_data,
            "chat_history": chat_history,
            "input": standalone_query,
        })
        answer = response.content
        timing["qa_generation"] = time.time() - t0

        return {
            "answer": answer,
            "context": [],  # Không có retrieved docs cho nhánh TOOL
            "from_cache": False,
            "standalone_query": standalone_query,
            "intent": "TOOL",
            "price_data": price_data,
            "timing": timing,
        }

    def _extract_number(self, text: str, keywords: list) -> float | None:
        """
        Trích xuất số từ text gần các keyword.
        VD: "chó 10kg lưu trú 7 ngày" → keywords=["ngày"] → 7
        """
        import re
        text_lower = text.lower()
        
        for kw in keywords:
            # Tìm pattern: số + keyword (VD: "10kg", "7 ngày")
            pattern = rf'(\d+(?:[.,]\d+)?)\s*{kw}'
            match = re.search(pattern, text_lower)
            if match:
                num_str = match.group(1).replace(",", ".")
                try:
                    if "." in num_str:
                        return float(num_str)
                    return int(num_str)
                except ValueError:
                    pass
        
        return None

    def _extract_all_numbers(self, text: str, keywords: list) -> list:
        """
        Trích xuất tất cả các số từ text gần các keyword.
        VD: "chó 10kg mèo 5kg" → keywords=["kg"] → [10, 5]
        """
        import re
        text_lower = text.lower()
        numbers = []
        
        kw_pattern = "|".join(keywords)
        pattern = rf'(\d+(?:[.,]\d+)?)\s*(?:{kw_pattern})'
        
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            num_str = match.group(1).replace(",", ".")
            try:
                if "." in num_str:
                    numbers.append(float(num_str))
                else:
                    numbers.append(int(num_str))
            except ValueError:
                pass
            
        return numbers

    def _handle_greeting(self, standalone_query: str, chat_history: list, timing: dict) -> dict:
        """
        Xử lý nhánh GREETING (LLM fallback): dùng greetings model để sinh câu trả lời.
        """
        t0 = time.time()
        response = self.greetings_chain.invoke({
            "chat_history": chat_history,
            "input": standalone_query
        })
        answer = response.content
        timing["greetings_generation"] = time.time() - t0

        return {
            "answer": answer,
            "context": [],
            "from_cache": False,
            "standalone_query": standalone_query,
            "intent": "GREETING",
            "timing": timing
        }

    def invoke(self, inputs: dict) -> dict:
        """
        Chạy pipeline Agentic RAG với Selective Caching.

        Args:
            inputs: {"input": str, "chat_history": list}

        Returns:
            {
                "answer": str,
                "context": List[Document],
                "from_cache": bool,
                "standalone_query": str,
                "intent": str ("KNOWLEDGE" | "TOOL" | "GREETING"),
                "timing": dict,
                "similarity": float (nếu from_cache=True),
                "price_data": str (nếu intent="TOOL"),
            }
        """
        query = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])
        timing = {}

        # --- Fast-path Regex Check ---
        if _is_greeting_fast(query):
            return {
                "answer": GREETING_RESPONSE,
                "context": [],
                "from_cache": False,
                "standalone_query": query,
                "intent": "GREETING",
                "timing": {"fast_regex_match": 0.0}
            }

        # --- Step 1: Rewrite query ---
        t0 = time.time()
        standalone_query = self._rewrite_query(query, chat_history)
        timing["rewrite_query"] = time.time() - t0

        # --- Step 2: Intent Router ---
        t0 = time.time()
        intent = self.router.classify(standalone_query)
        timing["intent_router"] = time.time() - t0
        print(f"🧭 Intent Router: {standalone_query} → {intent}")

        # --- Step 3: Route theo intent ---
        if intent == "GREETING":
            return self._handle_greeting(standalone_query, chat_history, timing)
        elif intent == "TOOL":
            return self._handle_tool(standalone_query, chat_history, timing)
        else:
            return self._handle_knowledge(standalone_query, chat_history, timing)


def build_conversational_rag_chain():
    """Khởi tạo toàn bộ pipeline cho Agentic RAG với Selective Caching."""
    llm = get_llm()
    router_llm = get_router_llm()
    greetings_llm = get_greetings_llm()
    rewrite_llm = get_rewrite_llm()
    
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
    
    # 5. Khởi tạo Intent Router (dùng router LLM — nhẹ, nhanh)
    router = IntentRouter(router_llm)
    
    # 6. Khởi tạo Service DB + import CSV
    service_db = ServiceDB()
    service_db.import_from_csv(CSV_PRICING_PATH)
    
    # 7. Kết hợp tất cả vào AgenticRAGPipeline
    pipeline = AgenticRAGPipeline(
        llm=llm,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        semantic_cache=semantic_cache,
        qa_chain=qa_chain,
        router=router,
        service_db=service_db,
        greetings_llm=greetings_llm,
        rewrite_llm=rewrite_llm,
    )
    
    return pipeline
