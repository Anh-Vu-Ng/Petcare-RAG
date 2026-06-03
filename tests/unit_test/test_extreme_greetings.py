import sys
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(".")
from src.prompts import greetings_prompt
load_dotenv()
test_queries = [
    "dự báo thời tiết hôm nay ở Sài Gòn?",
    "chào cưng, dạo này làm ăn khấm khá không hả, bot ơi?",
    "hi shop, hôm nay shop thế nào? có bận rộn nhiều ca tắm rửa cạo lông không nà?",
    "bạn có thể cho mình code 1 trang web không",
    "hello, shop có nhận thú cưng lớn không?",
    "Shop tệ quá, nghỉ bán đi",
    "sao rep chậm vậy, chán",
    "Tui bị trầm cảm nặng, cần bạn giúp đỡ",
    "kể chuyện cười đi bạn ơi",
    "qweiuuqwiu",
    "gọi 911 i"
]

print("Đang khởi động hệ thống...")
models = ("meta-llama/llama-3-8b-instruct","meta-llama/llama-3.1-8b-instruct", "google/gemma-3-4b-it")
for model in models: 
    print(f"\n==================================================")
    print(f" KHỞI CHẠY KIỂM THỬ VỚI MODEL: {model}")
    print(f"==================================================")
    llm = ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2, 
            top_p = 0.7,
            max_tokens=256,
            timeout=15
    )
    chain = greetings_prompt | llm
    print("\n=== START EXTREME GREETING TESTS ===\n")
    for idx, q in enumerate(test_queries, 1):
        print("-" * 50)
        print(f"Câu hỏi {idx}: {q}")
        response = chain.invoke({"input": q, "chat_history": []})
        ai_text = response.content
        print(f"AI response:\n{ai_text}\n")

print("=== TESTS COMPLETE ===")
