import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vector_store import get_embeddings
from src.semantic_cache import SemanticCache
from src.rag_chain import build_conversational_rag_chain

def test_cache():
    load_dotenv()
    embeddings = get_embeddings()
    
    # 1. Khởi tạo Semantic Cache
    cache = SemanticCache(embeddings)
    cache.clear()  # Xóa cache cũ để test sạch
    
    # 2. Store "xin chào"
    print("\n--- Store 'xin chào' ---")
    cache.store("xin chào", "Chào bạn! Mình là trợ lý Petcare.", [])
    
    # 3. Lookup "xin chào" (lần 2)
    print("\n--- Lookup 'xin chào' (lần 2) ---")
    res = cache.lookup("xin chào")
    if res:
        print(f"✅ HIT! Answer: {res['answer']}")
        print(f"Similarity: {res['similarity']:.4f}")
    else:
        print("❌ MISS!")
        # Thử lấy score trực tiếp từ FAISS để xem
        if cache.vectorstore:
            results = cache.vectorstore.similarity_search_with_score("xin chào", k=1)
            if results:
                doc, score = results[0]
                cosine_sim = 1.0 - (score / 2.0)
                print(f"Raw FAISS score: {score:.6f}")
                print(f"Calculated similarity: {cosine_sim:.6f}")

    # 4. Kiểm tra xem "xin chào" đi qua RAG Chain được phân loại là gì
    print("\n--- Kiểm tra phân loại intent cho 'xin chào' ---")
    pipeline = build_conversational_rag_chain()
    intent = pipeline.router.classify("xin chào")
    print(f"Query: 'xin chào' → Intent Router classify: {intent}")

if __name__ == "__main__":
    test_cache()
