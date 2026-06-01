import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(".")
from src.rag_chain import build_conversational_rag_chain

print("Đang khởi động hệ thống...")
pipeline = build_conversational_rag_chain()

# 3 câu hỏi cực đoan thử thách bộ lọc GREETING và model Gemma-3-4b-it
test_queries = [
    "dự báo thời tiết hôm nay ở Sài Gòn?",
    "chào cưng, dạo này làm ăn khấm khá không hả, bot ơi?",
    "hi shop, hôm nay shop thế nào? có bận rộn nhiều ca tắm rửa cạo lông không nà?",
    "bạn có phải là con người không hay chỉ là một cỗ máy tự động vậy?",
    "shop dùng AI à? hay có nhân viên thật không?",
    "hello, shop có nhận thú cưng lớn không?",
    "Shop tệ quá, nghỉ bán đi",
    "sao rep chậm vậy, chán",
    "Tui bị trầm cảm nặng, cần bạn giúp đỡ",
    "kể chuyện cười đi bạn ơi",
    "aaaaa"
]

print("\n=== START EXTREME GREETING TESTS ===\n")
for idx, q in enumerate(test_queries, 1):
    print("-" * 50)
    print(f"Câu hỏi {idx}: {q}")
    response = pipeline.invoke({
        "chat_history": [],
        "input": q
    })
    print(f"Intent: {response.get('intent')}")
    print(f"Bypass Fast Regex: {not response.get('timing', {}).get('fast_regex_match') is not None}")
    print(f"AI response:\n{response.get('answer')}\n")

print("=== TESTS COMPLETE ===")
