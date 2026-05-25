from src.config import BM25_INDEX_PATH, TOP_K_SPARSE, CHILD_DOCS_PATH, PARENT_DOCS_PATH
import os
import pickle
from langchain_community.retrievers import BM25Retriever
from src.text_processor import split_parent_child
from src.data_loader import load_all_docs

def create_and_save_bm25(documents, save_path): 
    """create bm25 index from documents chunks and serialize with pickle"""
    #check if folder exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    retriever = BM25Retriever.from_documents(documents, k = TOP_K_SPARSE)
    with open(save_path, "wb") as f:
        pickle.dump(retriever, f)
    return retriever

def load_bm25(save_path):
    """load bm25 index from pickle file"""
    with open(save_path,"rb") as f:
        return pickle.load(f)

def get_bm25_retriever():
    """load index co san trong o cug, neu ko thay thi tao moi"""
    if os.path.exists(BM25_INDEX_PATH):
        print("Loading BM25 from existing file")
        return load_bm25(BM25_INDEX_PATH)
    
    print("Can't find BM25 file... Creating new BM25 index")
    
    # Sử dụng chung các child chunks từ FAISS để đảm bảo đồng bộ parent_id
    if os.path.exists(CHILD_DOCS_PATH):
        print(f"Loading child documents from {CHILD_DOCS_PATH}...")
        with open(CHILD_DOCS_PATH, "rb") as f:
            chunks = pickle.load(f)
    else:
        print("Child docs pickle not found. Re-splitting all docs...")
        docs = load_all_docs()

        if not docs:
            raise ValueError("No docs found")
        
        parent_docs, chunks = split_parent_child(docs)
        
        # Lưu lại cả hai để FAISS có thể dùng chung
        os.makedirs(os.path.dirname(PARENT_DOCS_PATH), exist_ok=True)
        with open(PARENT_DOCS_PATH, "wb") as f:
            pickle.dump(parent_docs, f)
        with open(CHILD_DOCS_PATH, "wb") as f:
            pickle.dump(chunks, f)
            
    return create_and_save_bm25(chunks, BM25_INDEX_PATH)

