import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(".")
from src.rag_chain import build_conversational_rag_chain

print("Đang khởi động hệ thống...")
pipeline = build_conversational_rag_chain()

# 3 câu hỏi kiểm tra tính toán dịch vụ (TOOL)
test_queries = [
    # 1. Tra cứu giá khoảng cân nặng lẻ (Cạo lông 4.5kg)
    "giá cạo lông cho chó 4.5kg là bao nhiêu?",
    
    # 2. Tính giá lưu trú ngắn ngày (chó 8kg, gửi 5 ngày)
    "gửi chó 8kg trong 5 ngày hết bao nhiêu tiền?",
    
    # 3. Tra cứu dịch vụ tắm cho mèo (mèo 5kg)
    "tắm cho mèo 5kg giá bao nhiêu thế?"
]

print("\n=== START TOOL CALCULATION TESTS ===\n")
for idx, q in enumerate(test_queries, 1):
    print("-" * 50)
    print(f"Câu hỏi {idx}: {q}")
    response = pipeline.invoke({
        "chat_history": [],
        "input": q
    })
    print(f"Intent: {response.get('intent')}")
    print(f"Price Data sent to LLM:\n{response.get('price_data')}")
    print(f"AI response:\n{response.get('answer')}\n")

print("=== TESTS COMPLETE ===")
