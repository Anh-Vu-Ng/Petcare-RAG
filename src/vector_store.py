import os
from langchain_community.vectorstores import FAISS
from src.config import INDEX_DIR, TOP_K_DENSE
from src.data_loader import load_all_docs
from src.text_processor import splits_documents
from src.jina_embeddings import JinaEmbeddings

def get_embeddings():
    """Khởi tạo Jina Embeddings (gọi API)."""
    return JinaEmbeddings()

def get_faiss_retriever():
    """Load FAISS index từ ổ cứng hoặc tính toán/khởi tạo mới nếu chưa có."""
    embeddings = get_embeddings()
    
    # FAISS của LangChain lưu trữ mặc định 2 file: index.faiss và index.pkl
    faiss_index_path = os.path.join(INDEX_DIR, "index.faiss")
    
    if os.path.exists(faiss_index_path):
        print(f"Loading FAISS index from {INDEX_DIR}...")
        # Bắt buộc phải có allow_dangerous_deserialization=True ở LangChain bản mới
        vectorstore = FAISS.load_local(
            INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        print("FAISS index not found. Loading data and building new index...")
        docs = load_all_docs()
        
        if not docs:
            raise ValueError("No input data found. Data Ingestion pipeline is broken!")
            
        chunks = splits_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Tạo thư mục và lưu lại để lần sau không phải nhúng lại toàn bộ
        os.makedirs(INDEX_DIR, exist_ok=True)
        vectorstore.save_local(INDEX_DIR)
        print(f"FAISS index saved at {INDEX_DIR}")

    # Trả về đối tượng retriever để cắm vào RAG Chain
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_DENSE})