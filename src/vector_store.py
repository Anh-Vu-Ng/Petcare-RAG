import os
from langchain_community.vectorstores import FAISS
from src.config import INDEX_DIR, TOP_K_DENSE, PARENT_DOCS_PATH, CHILD_DOCS_PATH
from src.data_loader import load_all_docs
from src.text_processor import split_parent_child
from src.jina_embeddings import JinaEmbeddings
import pickle

def get_embeddings():
    """Khởi tạo Jina Embeddings (gọi API)."""
    return JinaEmbeddings()

def get_faiss_retriever():
    """Load FAISS index từ ổ cứng hoặc tính toán/khởi tạo mới nếu chưa có."""
    embeddings = get_embeddings()
    
    # FAISS của LangChain lưu trữ mặc định 2 file: index.faiss và index.pkl
    faiss_index_path = os.path.join(INDEX_DIR, "index.faiss")
    
    if os.path.exists(faiss_index_path) and os.path.exists(PARENT_DOCS_PATH):
        print(f"Loading FAISS index from {INDEX_DIR}...")
        # Bắt buộc phải có allow_dangerous_deserialization=True ở LangChain bản mới
        vectorstore = FAISS.load_local(
            INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        print("FAISS index or Parent Docs not found. Loading data and building new index...")
        docs = load_all_docs()
        
        if not docs:
            raise ValueError("No input data found. Data Ingestion pipeline is broken!")
            
        parent_docs, child_docs = split_parent_child(docs)
        
        # Tạo thư mục và lưu lại để lần sau không phải nhúng lại toàn bộ
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        print(f"Saving parent documents to {PARENT_DOCS_PATH}...")
        with open(PARENT_DOCS_PATH, "wb") as f:
            pickle.dump(parent_docs, f)
            
        print(f"Saving child documents to {CHILD_DOCS_PATH}...")
        with open(CHILD_DOCS_PATH, "wb") as f:
            pickle.dump(child_docs, f)
            
        vectorstore = FAISS.from_documents(child_docs, embeddings)
        vectorstore.save_local(INDEX_DIR)
        print(f"FAISS index saved at {INDEX_DIR}")

    # Trả về đối tượng retriever để cắm vào RAG Chain
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_DENSE})