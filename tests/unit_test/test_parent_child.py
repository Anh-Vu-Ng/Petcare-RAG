import os
import sys
import shutil
import pickle

# Thêm đường dẫn project
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader import load_all_docs
from src.text_processor import split_parent_child
from src.vector_store import get_faiss_retriever
from src.rag_chain import build_conversational_rag_chain

def test_pipeline():
    # Cấu hình encoding để in ký tự tiếng Việt trên Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
        
    print("=== BẮT ĐẦU TEST PARENT-CHILD PIPELINE ===")
    
    # 1. Xóa các collection cũ trên Qdrant Cloud để ép rebuild
    from qdrant_client import QdrantClient
    from src.config import URL_QDRANT, QDRANT_API_KEY, QDRANT_KB_COLLECTION, QDRANT_PARENT_COLLECTION, QDRANT_CACHE_COLLECTION
    import time
    print("Xóa các collection cũ trên Qdrant Cloud để ép rebuild...")
    try:
        client = QdrantClient(url=URL_QDRANT, api_key=QDRANT_API_KEY)
        for col_name in [QDRANT_KB_COLLECTION, QDRANT_PARENT_COLLECTION, QDRANT_CACHE_COLLECTION]:
            if client.collection_exists(col_name):
                client.delete_collection(col_name)
                # Chờ tối đa 10 giây để xóa xong
                for _ in range(20):
                    if not client.collection_exists(col_name):
                        break
                    time.sleep(0.5)
    except Exception as e:
        print(f"Không thể xóa collection (có thể chưa tồn tại): {e}")
        
    # 2. Chạy load_all_docs
    print("\n1. Đang load documents...")
    docs = load_all_docs()
    print(f"Đã load {len(docs)} documents.")
    
    # 3. Test hàm split
    print("\n2. Đang phân tách parent/child...")
    parents, children = split_parent_child(docs)
    print(f"Số lượng parents: {len(parents)}")
    print(f"Số lượng children: {len(children)}")
    
    # Kiểm tra liên kết
    if children:
        sample_child = children[0]
        parent_id = sample_child.metadata.get("parent_id")
        print(f"Child sample metadata: {sample_child.metadata}")
        print(f"Parent liên kết có tồn tại không: {parent_id in parents}")
        if parent_id in parents:
            print(f"Độ dài Parent chunk: {len(parents[parent_id].page_content)} ký tự.")
            print(f"Độ dài Child chunk: {len(sample_child.page_content)} ký tự.")
            
    # 4. Ép sinh lại Qdrant index
    print("\n3. Đang khởi tạo Qdrant hybrid retriever...")
    qdrant_retriever = get_faiss_retriever() # Sử dụng alias get_faiss_retriever
    
    # Kiểm tra xem các file và Qdrant collection có tồn tại đúng không
    print("\n5. Kiểm tra các file và Qdrant index:")
    from qdrant_client import QdrantClient
    from src.config import URL_QDRANT, QDRANT_API_KEY, QDRANT_KB_COLLECTION, QDRANT_PARENT_COLLECTION
    try:
        client = QdrantClient(url=URL_QDRANT, api_key=QDRANT_API_KEY)
        col = client.get_collection(QDRANT_KB_COLLECTION)
        print(f"Qdrant collection '{QDRANT_KB_COLLECTION}': EXISTS (with {col.points_count} points)")
        col_p = client.get_collection(QDRANT_PARENT_COLLECTION)
        print(f"Qdrant collection '{QDRANT_PARENT_COLLECTION}': EXISTS (with {col_p.points_count} points)")
    except Exception as e:
        print(f"Qdrant check: FAILED ({e})")
        
    pass
    
    # 5. Khởi tạo toàn bộ RAG Chain và test câu hỏi
    print("\n6. Khởi tạo RAG Chain...")
    chain = build_conversational_rag_chain()
    
    query = "Làm gì khi chó bị khó sinh?"
    print(f"\n7. Test truy vấn: '{query}'")
    response = chain.invoke({"chat_history": [], "input": query})
    
    print("\n=== KẾT QUẢ TRẢ VỀ ===")
    print(f"Intent: {response.get('intent')}")
    print(f"From Cache: {response.get('from_cache')}")
    print("\n[Các Document Context dùng làm ngữ cảnh cho LLM]:")
    for i, doc in enumerate(response.get("context", []), 1):
        print(f"\nDocument {i} (Source: {doc.metadata.get('source')}):")
        # In độ dài xem có phải parent document cỡ lớn không
        print(f"Độ dài chunk: {len(doc.page_content)} ký tự.")
        print(f"Nội dung (500 ký tự đầu):\n{doc.page_content[:500]}...")
        
    print("\n[Câu trả lời của LLM]:")
    print(response.get("answer"))

if __name__ == "__main__":
    test_pipeline()
