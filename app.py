import time
import pandas as pd
import streamlit as st
from src.rag_chain import build_conversational_rag_chain
from src.service_db import ServiceDB, SERVICE_NAME_MAP

# Cấu hình trang
st.set_page_config(
    page_title="Petcare Bot",
    page_icon="🛍️",
    layout="wide"
)

# Khởi tạo pipeline RAG (chỉ load 1 lần)
@st.cache_resource
def load_chain():
    return build_conversational_rag_chain()

@st.cache_resource
def load_service_db():
    return ServiceDB()

st.title("🛍️ PETCARE BOT (Agentic RAG)")
st.markdown("Rất vui được phục vụ bạn và bé iu nhenn.")

# Khởi tạo biến trạng thái để lưu lịch sử
if "messages" not in st.session_state:
    st.session_state.messages = [] # Dùng để render UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Dùng để nạp vào LangChain

try:
    pipeline = load_chain()
    service_db = load_service_db()
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    # Tab selector
    sidebar_tab = st.radio(
        "📌 Sidebar",
        ["⚡ Cache Stats", "📋 Bảng giá dịch vụ", "➕ Quản lý dịch vụ"],
        label_visibility="collapsed",
    )

    if sidebar_tab == "⚡ Cache Stats":
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

    elif sidebar_tab == "📋 Bảng giá dịch vụ":
        st.header("📋 Bảng giá dịch vụ")
        
        # Filter theo loại dịch vụ
        service_types = service_db.get_service_types()
        type_options = ["Tất cả"] + [s["name"] for s in service_types]
        selected_service = st.selectbox("Lọc theo dịch vụ:", type_options)
        
        if selected_service == "Tất cả":
            services = service_db.get_all_services()
        else:
            # Tìm service_type từ service_name
            stype = next(
                (s["type"] for s in service_types if s["name"] == selected_service),
                None,
            )
            services = service_db.get_price_table_for_service(stype) if stype else []
        
        if services:
            df = pd.DataFrame(services)
            df = df[["weight_kg", "service_name", "price"]]
            df.columns = ["Cân nặng (kg)", "Dịch vụ", "Giá (VND)"]
            df["Giá (VND)"] = df["Giá (VND)"].apply(lambda x: f"{x:,}đ".replace(",", "."))
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có dữ liệu dịch vụ.")
        
        # Bảng chính sách discount
        st.markdown("---")
        st.subheader("🏷️ Chính sách giảm giá lưu trú")
        discount_data = {
            "Số ngày": ["≤ 3 ngày", "4-5 ngày", "6-10 ngày", "> 10 ngày"],
            "Giảm giá": ["0%", "5%", "10%", "15%"],
            "Bonus": ["—", "—", "—", "🎁 Tắm free"],
        }
        st.table(pd.DataFrame(discount_data))
        st.caption("📌 Giá lưu trú đã bao gồm ăn uống.")

    elif sidebar_tab == "➕ Quản lý dịch vụ":
        st.header("➕ Quản lý dịch vụ")
        
        with st.form("add_service_form"):
            st.subheader("Thêm/Cập nhật dịch vụ")
            col_a, col_b = st.columns(2)
            
            with col_a:
                new_weight = st.number_input("Cân nặng (kg)", min_value=1, max_value=100, value=5)
                new_service = st.selectbox(
                    "Loại dịch vụ",
                    options=list(SERVICE_NAME_MAP.keys()),
                    format_func=lambda x: SERVICE_NAME_MAP[x],
                )
            with col_b:
                new_price = st.number_input("Giá (VND)", min_value=0, step=10000, value=100000)
            
            submitted = st.form_submit_button("💾 Lưu", use_container_width=True)
            if submitted:
                service_db.add_service(
                    weight_kg=new_weight,
                    service_type=new_service,
                    service_name=SERVICE_NAME_MAP[new_service],
                    price=new_price,
                )
                st.success(f"Đã lưu: {SERVICE_NAME_MAP[new_service]} ({new_weight}kg) = {new_price:,}đ")
                st.rerun()
        
        # Re-import CSV
        st.markdown("---")
        if st.button("🔄 Re-import từ CSV", use_container_width=True):
            service_db.import_from_csv(force=True)
            st.success("Đã re-import dữ liệu từ CSV!")
            st.rerun()

# Hiển thị lại lịch sử chat trên UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Hiển thị lại metadata nếu có
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            # Intent + Cache badge
            intent_badge = "🧭 **KNOWLEDGE**" if meta.get("intent") == "KNOWLEDGE" else "🔧 **TOOL**"
            cache_badge = "⚡ **Cache Hit**" if meta.get("from_cache") else ""
            
            st.caption(
                f"{intent_badge} {cache_badge} · ⏱️ **{meta['elapsed_time']:.2f}s** · 📄 {meta['num_docs']} tài liệu"
            )
            if meta.get("standalone_query"):
                st.caption(f"🔍 Standalone query: _{meta['standalone_query']}_")
            
            # Nhánh KNOWLEDGE: hiển thị retrieved docs
            if meta.get("intent") == "KNOWLEDGE" and meta.get("context_docs"):
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
            
            # Nhánh TOOL: hiển thị price data
            if meta.get("intent") == "TOOL" and meta.get("price_data"):
                with st.expander("💰 Xem dữ liệu giá tra cứu", expanded=False):
                    st.text(meta["price_data"])
            
            if meta.get("timing"):
                with st.expander("⏱️ Timing breakdown", expanded=False):
                    timing_labels = {
                        "rewrite_query": "✏️ Query Rewrite",
                        "intent_router": "🧭 Intent Router",
                        "cache_lookup": "⚡ Cache Lookup",
                        "hybrid_retrieval": "🔎 Hybrid Retrieval",
                        "jina_reranker": "🏆 Jina Reranker",
                        "tool_lookup": "🔧 Tool Lookup (Supabase)",
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
                intent = response.get("intent", "KNOWLEDGE")
                timing = response.get("timing", {})
                price_data = response.get("price_data", "")
                
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
                        "content": doc.page_content[:500]
                    }
                    if rrf_score is not None:
                        doc_data["rrf_score"] = rrf_score
                    if rerank_score is not None:
                        doc_data["rerank_score"] = rerank_score
                    context_docs.append(doc_data)

                # Intent + Cache badge
                intent_badge = "🧭 **KNOWLEDGE**" if intent == "KNOWLEDGE" else "🔧 **TOOL**"
                cache_badge = " · ⚡ **Cache Hit**" if from_cache else ""
                similarity_info = ""
                if from_cache and "similarity" in response:
                    similarity_info = f" · 🎯 Similarity: {response['similarity']:.3f}"
                
                st.caption(
                    f"{intent_badge}{cache_badge}{similarity_info} · ⏱️ **{elapsed_time:.2f}s** · 📄 {len(retrieved_docs)} tài liệu"
                )
                
                if standalone_query and standalone_query != prompt:
                    st.caption(f"🔍 Standalone query: _{standalone_query}_")
                
                # Nhánh KNOWLEDGE: hiển thị retrieved docs
                if intent == "KNOWLEDGE":
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
                
                # Nhánh TOOL: hiển thị price data
                if intent == "TOOL" and price_data:
                    with st.expander("💰 Xem dữ liệu giá tra cứu", expanded=False):
                        st.text(price_data)
                
                # Hiển thị timing breakdown
                if timing:
                    with st.expander("⏱️ Timing breakdown", expanded=True):
                        timing_labels = {
                            "rewrite_query": "✏️ Query Rewrite",
                            "intent_router": "🧭 Intent Router",
                            "cache_lookup": "⚡ Cache Lookup",
                            "hybrid_retrieval": "🔎 Hybrid Retrieval",
                            "jina_reranker": "🏆 Jina Reranker",
                            "tool_lookup": "🔧 Tool Lookup (Supabase)",
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
                
                # Cập nhật trạng thái
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "elapsed_time": elapsed_time,
                        "num_docs": len(retrieved_docs),
                        "context_docs": context_docs,
                        "from_cache": from_cache,
                        "standalone_query": standalone_query,
                        "intent": intent,
                        "price_data": price_data,
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