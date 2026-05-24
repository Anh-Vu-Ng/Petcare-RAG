import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_chain import build_conversational_rag_chain

def test_pipeline():
    print("--- Khởi tạo pipeline ---")
    load_dotenv()
    pipeline = build_conversational_rag_chain()
    print("✅ Khởi tạo pipeline thành công!\n")

    # Kịch bản 1: Query đầu tiên (không có lịch sử, không chạy rewrite)
    print("--- Kịch bản 1: Lượt chat đầu tiên (Không có chat history) ---")
    inputs_1 = {
        "input": "Địa chỉ của Petcare ở đâu?",
        "chat_history": []
    }
    res_1 = pipeline.invoke(inputs_1)
    print(f"Query gốc: {inputs_1['input']}")
    print(f"Standalone Query: {res_1.get('standalone_query')}")
    print(f"Intent: {res_1.get('intent')}")
    print(f"Answer: {res_1.get('answer')[:100]}...\n")

    # Kịch bản 2: Lượt chat thứ hai (có lịch sử, chạy rewrite bằng main LLM)
    print("--- Kịch bản 2: Lượt chat tiếp theo (Có chat history) ---")
    chat_history = [
        ("human", "Shop ở đâu vậy?"),
        ("ai", "Dạ chào anh/chị, Petcare Assistant xin chào! Địa chỉ của cửa hàng Petcare ở Hà Nội và TP.HCM ạ.")
    ]
    inputs_2 = {
        "input": "Chi nhánh Hà Nội nằm ở đường nào và giá tắm bé mèo 3kg là bao nhiêu?",
        "chat_history": chat_history
    }
    res_2 = pipeline.invoke(inputs_2)
    print(f"Query gốc: {inputs_2['input']}")
    print(f"Standalone Query: {res_2.get('standalone_query')}")
    print(f"Intent: {res_2.get('intent')}")
    print(f"Answer: {res_2.get('answer')[:150]}...\n")
    print(f"Timing: {res_2.get('timing')}")

if __name__ == "__main__":
    test_pipeline()
