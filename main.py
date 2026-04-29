from src.rag_chain import build_conversational_rag_chain

def main():
    print("Đang khởi động hệ thống RAG Hybrid + Semantic Cache. Vui lòng đợi...")
    try:
        pipeline = build_conversational_rag_chain()
    except Exception as e:
        print(f"Lỗi khởi tạo hệ thống: {e}")
        return

    chat_history = []
    print("\n✅ Hệ thống đã sẵn sàng! Gõ 'exit' hoặc 'quit' để thoát.")
    print("   Gõ 'cache' để xem thống kê cache, 'clear' để xóa cache.")
    print("-" * 50)
    
    while True:
        query = input("\n🧑 Người dùng: ")
        if query.lower() in ['exit', 'quit']:
            print("Đang thoát hệ thống...")
            break
        
        # Lệnh quản lý cache
        if query.lower() == 'cache':
            stats = pipeline.semantic_cache.stats()
            print(f"📦 Cache: {stats['total_entries']} entries | "
                  f"✅ Hits: {stats['hit_count']} | ❌ Misses: {stats['miss_count']} | "
                  f"🎯 Hit Rate: {stats['hit_rate']}")
            continue
        if query.lower() == 'clear':
            pipeline.semantic_cache.clear()
            continue
            
        print("🤖 AI đang suy nghĩ...")
        try:
            response = pipeline.invoke({
                "chat_history": chat_history,
                "input": query
            })
            
            answer = response["answer"]
            from_cache = response.get("from_cache", False)
            standalone_query = response.get("standalone_query", "")
            
            # Hiển thị badge cache
            if from_cache:
                sim = response.get("similarity", 0)
                print(f"  ⚡ Cache HIT (similarity: {sim:.3f})")
            else:
                print(f"  🔄 Cache MISS — đã lưu vào cache")
            
            if standalone_query and standalone_query != query:
                print(f"  🔍 Standalone query: {standalone_query}")
            
            print(f"\n🤖 AI: {answer}")
            
            # Cập nhật lịch sử (Dùng tuple theo chuẩn của LangChain)
            chat_history.append(("human", query))
            chat_history.append(("ai", answer))
            
            # Giữ lại K=5 cặp hội thoại gần nhất (10 phần tử)
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
                
        except Exception as e:
            print(f"❌ Lỗi khi truy vấn: {e}")

if __name__ == "__main__":
    main()