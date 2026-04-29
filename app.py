import time
import streamlit as st
from src.rag_chain import build_conversational_rag_chain

# Cấu hình trang
st.set_page_config(
    page_title="Alya Petcare",
    page_icon="🛍️",
    layout="wide"
)

# Khởi tạo pipeline RAG (chỉ load 1 lần)
@st.cache_resource
def load_chain():
    return build_conversational_rag_chain()

st.title("🛍️ PETCARE BOT (Hybrid RAG)")
st.markdown("Hệ thống sử dụng Hybrid Retriever(BM25 & FAISS) kết hợp thuật toán RRF + **Jina Reranker v2** + **Semantic Cache**.")

# Khởi tạo biến trạng thái để lưu lịch sử
if "messages" not in st.session_state:
    st.session_state.messages = [] # Dùng để render UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Dùng để nạp vào LangChain

try:
    pipeline = load_chain()
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống: {e}")
    st.stop()

# --- Sidebar: Cache Stats & Controls ---
with st.sidebar:
    st.header("⚡ Semantic Cache")
    
    cache_stats = pipeline.semantic_cache.stats()
    col1, col2 = st.columns(2)
    col1.metric("📦 Entries", cache_stats["total_entries"])
    col2.metric("🎯 Hit Rate", cache_stats["hit_rate"])
    
    col3, col4 = st.columns(2)
    col3.metric("✅ Hits", cache_stats["hit_count"])
    col4.metric("❌ Misses", cache_stats["miss_count"])
    
    st.markdown(f"**Threshold:** {pipeline.semantic_cache.threshold}")
    
    if st.button("🗑️ Xóa toàn bộ Cache", use_container_width=True):
        pipeline.semantic_cache.clear()
        st.success("Đã xóa cache!")
        st.rerun()

# Hiển thị lại lịch sử chat trên UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Hiển thị lại metadata (time + context) nếu có
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            # Cache badge
            cache_badge = "⚡ **Từ cache**" if meta.get("from_cache") else "🔄 **Pipeline**"
            st.caption(
                f"{cache_badge} · ⏱️ **{meta['elapsed_time']:.2f}s** · 📄 {meta['num_docs']} tài liệu"
            )
            if meta.get("standalone_query"):
                st.caption(f"🔍 Standalone query: _{meta['standalone_query']}_")
            with st.expander("🔍 Xem dữ liệu đã truy xuất", expanded=False):
                for i, doc_info in enumerate(meta["context_docs"], 1):
                    scores = []
                    if "rerank_score" in doc_info:
                        scores.append(f"🏆 Rerank: {doc_info['rerank_score']:.4f}")
                    if "rrf_score" in doc_info:
                        scores.append(f"🎯 RRF: {doc_info['rrf_score']:.4f}")
                    score_text = f" — {' · '.join(scores)}" if scores else ""
                    st.markdown(f"**Tài liệu {i}** — `{doc_info['source']}`{score_text}")
                    st.code(doc_info["content"], language=None)
            if meta.get("timing"):
                with st.expander("⏱️ Timing breakdown", expanded=False):
                    timing_labels = {
                        "rewrite_query": "✏️ Query Rewrite",
                        "cache_lookup": "⚡ Cache Lookup",
                        "hybrid_retrieval": "🔎 Hybrid Retrieval",
                        "jina_reranker": "🏆 Jina Reranker",
                        "qa_generation": "🤖 QA Generation",
                        "cache_store": "💾 Cache Store",
                    }
                    for key, label in timing_labels.items():
                        if key in meta["timing"]:
                            val = meta["timing"][key]
                            st.markdown(f"{label}: **{val:.3f}s**")

# Xử lý input từ người dùng
if prompt := st.chat_input("Trò chuyện với Petcare ở đây nhen!"):
    # Render câu hỏi
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Render câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Alya đang suy nghĩ nhenn..."):
            try:
                # Đo thời gian phản hồi
                start_time = time.time()
                response = pipeline.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": prompt
                })
                elapsed_time = time.time() - start_time
                
                answer = response["answer"]
                from_cache = response.get("from_cache", False)
                standalone_query = response.get("standalone_query", "")
                timing = response.get("timing", {})
                # st.markdown(answer)
                # Hàm generator để nhả chữ từ từ
                def stream_data(text, delay=0.02):
                    for word in text.split(" "):
                        yield word + " "
                        time.sleep(delay)
                
                st.write_stream(stream_data(answer))
                
                # Trích xuất context documents từ response
                context_docs = []
                retrieved_docs = response.get("context", [])
                for doc in retrieved_docs:
                    source = doc.metadata.get("source", "N/A")
                    page = doc.metadata.get("page_number", None)
                    rrf_score = doc.metadata.get("rrf_score", None)
                    rerank_score = doc.metadata.get("rerank_score", None)
                    source_label = f"{source} (trang {page})" if page else source
                    
                    doc_data = {
                        "source": source_label,
                        "content": doc.page_content[:500]  # Giới hạn hiển thị 500 ký tự
                    }
                    if rrf_score is not None:
                        doc_data["rrf_score"] = rrf_score
                    if rerank_score is not None:
                        doc_data["rerank_score"] = rerank_score
                    context_docs.append(doc_data)

                # Cache badge + thời gian + số tài liệu
                cache_badge = "⚡ **Từ cache**" if from_cache else "🔄 **Pipeline**"
                similarity_info = ""
                if from_cache and "similarity" in response:
                    similarity_info = f" · 🎯 Similarity: {response['similarity']:.3f}"
                
                st.caption(
                    f"{cache_badge}{similarity_info} · ⏱️ **{elapsed_time:.2f}s** · 📄 {len(retrieved_docs)} tài liệu"
                )
                
                if standalone_query and standalone_query != prompt:
                    st.caption(f"🔍 Standalone query: _{standalone_query}_")
                
                # Hiển thị chi tiết tài liệu đã truy xuất
                with st.expander("🔍 Xem dữ liệu đã truy xuất", expanded=False):
                    if context_docs:
                        for i, doc_info in enumerate(context_docs, 1):
                            scores = []
                            if "rerank_score" in doc_info:
                                scores.append(f"🏆 Rerank: {doc_info['rerank_score']:.4f}")
                            if "rrf_score" in doc_info:
                                scores.append(f"🎯 RRF: {doc_info['rrf_score']:.4f}")
                            score_text = f" — {' · '.join(scores)}" if scores else ""
                            st.markdown(f"**Tài liệu {i}** — `{doc_info['source']}`{score_text}")
                            st.code(doc_info["content"], language=None)
                    else:
                        st.info("Không có tài liệu nào được truy xuất.")
                
                # Hiển thị timing breakdown
                if timing:
                    with st.expander("⏱️ Timing breakdown", expanded=True):
                        timing_labels = {
                            "rewrite_query": "✏️ Query Rewrite",
                            "cache_lookup": "⚡ Cache Lookup",
                            "hybrid_retrieval": "🔎 Hybrid Retrieval",
                            "jina_reranker": "🏆 Jina Reranker",
                            "qa_generation": "🤖 QA Generation",
                            "cache_store": "💾 Cache Store",
                        }
                        for key, label in timing_labels.items():
                            if key in timing:
                                val = timing[key]
                                bar_len = int(val / elapsed_time * 20) if elapsed_time > 0 else 0
                                bar = "█" * bar_len + "░" * (20 - bar_len)
                                pct = (val / elapsed_time * 100) if elapsed_time > 0 else 0
                                st.markdown(f"`{bar}` {label}: **{val:.3f}s** ({pct:.1f}%)")
                
                # Cập nhật trạng thái (lưu kèm metadata để render lại khi rerun)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "elapsed_time": elapsed_time,
                        "num_docs": len(retrieved_docs),
                        "context_docs": context_docs,
                        "from_cache": from_cache,
                        "standalone_query": standalone_query,
                        "timing": timing,
                    }
                })
                st.session_state.chat_history.extend([
                    ("human", prompt),
                    ("ai", answer)
                ])
                
                # Cắt tỉa lịch sử (K=5)
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]
                
                # Rerun để cập nhật sidebar stats
                st.rerun()
                    
            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong lúc suy luận: {e}")