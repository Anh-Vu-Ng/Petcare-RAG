import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.intent_router import IntentRouter

# Cấu hình tự động nhận diện đường dẫn tuyệt đối của thư mục gốc
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding='utf-8')

TEST_CASES = [
    # --- NHÓM GREETING: Bẫy than vãn, kể chuyện, đùa giỡn, từ khóa nhiễu ---
    {"query": "Ê bot, tắm chó với cạo lông mệt ghê á, nay tui làm mỏi cả tay.", "expected": "GREETING"},
    {"query": "Hôm qua chó nhà tui đi cạo lông về nhìn buồn cười xỉu 🤣🤣🤣", "expected": "GREETING"},
    {"query": "Giá như ngày xưa tôi không nuôi chó thì giờ đâu có tốn tiền mua hạt, haiz.", "expected": "GREETING"},
    {"query": "Bao nhiêu ân tình tui trao cho con Husky mà nó toàn cắn tui, buồn ớn.", "expected": "GREETING"},
    {"query": "1+1 bằng mấy hở shop, shop có rành toán như rành tỉa lông chó không?", "expected": "GREETING"},
    {"query": "Mèo nhà tớ tên là Báo, 5kg, sở thích là cắn chủ, tớ kể vậy thôi chứ chả hỏi gì.", "expected": "GREETING"},
    {"query": "Tháng này kẹt tiền quá, shop cho tui thiếu nợ 1 tháng được không?", "expected": "GREETING"},
    {"query": "dfjgkdfjg lsdkfjsldkfj chó mèo 5kg", "expected": "GREETING"},
    {"query": "Viết cho mình một bài thơ 4 câu về việc đi tắm cho chó nhé.", "expected": "GREETING"},
    {"query": "Hello em gái, có người yêu chưa hay vẫn đang cạo lông chó thuê thế?", "expected": "GREETING"},
    {"query": "Cho mình 100 bài code thiếu nhi nhé 😭😭😭", "expected": "GREETING"},
    {"query": "Tối nay Việt Nam đá với Thái Lan mấy giờ vậy shop?", "expected": "GREETING"},
    {"query": "Tắm, cạo lông, cắt móng. Đọc ba chữ này xong thấy rùng mình vì nhớ lại hồi đi làm thuê.", "expected": "GREETING"},
    {"query": "Shop ơi, nãy tui nhắn lộn nha, tui định nhắn cho tiệm thú y kế bên.", "expected": "GREETING"},
    {"query": "Tui bị trầm cảm, bạn có thể giúp tui không", "expected": "GREETING"},

    # --- NHÓM TOOL: Bẫy hỏi giá lóng, gián tiếp, sai chính tả, đa ý định ---
    {"query": "Cậu vàng nhà mình bị nấm da rụng lông tùm lum, tiệm có nhận tắm không và rổ giá ntn cho bé 4 kí?", "expected": "TOOL"},
    {"query": "Chồng mình bảo đi cắt móng cho con Poodle tốn tiền lắm, shop báo cho mình cái bill để mình đưa ổng coi thử.", "expected": "TOOL"},
    {"query": "Ép tuyến hôi cho mèo xiêm nhiu cành v ad?", "expected": "TOOL"},
    {"query": "Tuần sau mình đi du lịch Đà Lạt, tính vứt 2 con báo thủ 5kg ở lại shop bên bạn 4 hôm, hết bao nhiêu lúa vậy ad?", "expected": "TOOL"},
    {"query": "Tắm rửa sạch sẽ với lấy ráy tai cho chó Corgi chân ngắn thì móc hầu bao mấy đồng?", "expected": "TOOL"},
    {"query": "Sắp Tết rồi tui muốn tân trang lại cho ẻm, combo cạo lông tạo kiểu chó 6kg tính tiền sao shop?", "expected": "TOOL"},
    {"query": "Bên mình có nhận lưu trú cún hung dữ không, mún gửi 2 ngày thì thiệt hại bao nhiêu?", "expected": "TOOL"},
    {"query": "Con Golden 30 cân nhà chị to như con bò, dắt đi tắm với cắt móng thì mang 500k có đủ trả không em?", "expected": "TOOL"},
    {"query": "gửi mồn lèo 10 kí 3 ngày bn tiền?", "expected": "TOOL"},
    {"query": "Cho xin chi phí dọn dẹp lỗ tai với vắt tuyến hôi mèo 2 tháng tuổi.", "expected": "TOOL"},
    {"query": "Cạo lông xù chó Phốc sóc 3kg hết nhiêu lúa gạo vậy shop ơi?", "expected": "TOOL"},
    {"query": "Mèo mình bị tiêu chảy mấy nay, mà dơ quá, đem qua tắm thì giá cả sao bạn?", "expected": "TOOL"},
    {"query": "báo giá giữ chó qua đêm lễ 30/4", "expected": "TOOL"},
    {"query": "Tính sơ sơ cho anh tiền cắt mài móng 3 con chó mỗi con 10kg nha.", "expected": "TOOL"},
    {"query": "Làm sạch tai 2 pé moè, 1 pé 3kg, 1 pé 5kg thì tổng bill nhiêu v ạ?", "expected": "TOOL"},
    {"query": "Nhỏ bạn mới giới thiệu qua shop, cho tui hỏi giặt cún cưng 7 kí là bao nhiêu lúa vại?", "expected": "TOOL"},
    {"query": "Shop coi bộ làm ăn uy tín, tui gửi con mèo Anh lông ngắn 4kg nửa tháng thì lấy tui bao nhiêu tiền?", "expected": "TOOL"},

    # --- NHÓM KNOWLEDGE: Bẫy hỏi giá ngoài luồng (sản phẩm/y tế) & hỏi cách làm dịch vụ cốt lõi ---
    {"query": "Bên mình siêu âm thai cho chó với xét nghiệm máu chó thì tính giá sao vậy?", "expected": "KNOWLEDGE"},
    {"query": "Thuốc xịt rận cho mèo giá bao nhiêu tiền một chai?", "expected": "KNOWLEDGE"},
    {"query": "Làm sao để tự cắt móng cho chó ở nhà mà không bị chảy máu hả shop?", "expected": "KNOWLEDGE"},
    {"query": "Ủa shop dời địa chỉ qua quận 7 rồi hả, cho xin lại định vị nha.", "expected": "KNOWLEDGE"},
    {"query": "Chó pug ăn hạt gì thì mượt lông và không bị hôi miệng?", "expected": "KNOWLEDGE"},
    {"query": "Túi vận chuyển phi hành gia cho mèo có bán không, giá bao nhiêu?", "expected": "KNOWLEDGE"},
    {"query": "Mèo nhà tui đi vệ sinh ra máu, có cần đem qua phòng khám ngay không hay theo dõi ở nhà?", "expected": "KNOWLEDGE"},
    {"query": "Bên mình có dịch vụ đỡ đẻ cho chó không, chi phí thế nào?", "expected": "KNOWLEDGE"},
    {"query": "Cho hỏi vắc xin dại chó tiêm 1 mũi giá nhiêu tiền?", "expected": "KNOWLEDGE"},
    {"query": "Có nên thường xuyên lấy ráy tai cho mèo bằng tăm bông của người không?", "expected": "KNOWLEDGE"},
    {"query": "Khách sạn thú cưng bên mình chuồng trại có máy lạnh với camera theo dõi không?", "expected": "KNOWLEDGE"},
    {"query": "Nặn tuyến hôi cho chó xong bị sưng đỏ thì bôi thuốc gì cho xẹp vậy?", "expected": "KNOWLEDGE"},
    {"query": "Tắm chó bằng dầu gội của người có làm cún bị viêm da không ad?", "expected": "KNOWLEDGE"},
    {"query": "Mấy giờ thì bên shop đóng cửa không nhận khách gửi chó nữa vậy?", "expected": "KNOWLEDGE"},
    {"query": "Chó Poodle cạo lông máu lúc mấy tháng tuổi là đẹp nhất?", "expected": "KNOWLEDGE"},
    {"query": "Bảng giá tiêm phòng 7 bệnh với tẩy giun như thế nào?", "expected": "KNOWLEDGE"},
    {"query": "Pate cún loại nào giá rẻ mà tốt, tư vấn em với.", "expected": "KNOWLEDGE"}
]

# --- 2. HÀM KHỞI TẠO LLM ---
def get_router_llm(model_name: str) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Chưa thiết lập OPENROUTER_API_KEY. Kiểm tra lại file .env")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,    
        top_p=0.1,          
        max_tokens=10,     
        presence_penalty=0.0,
        frequency_penalty=0.0,
        timeout=15,
        extra_body={
            "reasoning": {"enabled": False}
        }
    )

# --- 3. HÀM CHẠY EVALUATE CHO TỪNG MODEL ---
def evaluate_model(model_name: str, test_cases: list) -> tuple:
    model_logs = []
    
    # Tạo tiêu đề block đúng định dạng 60 ký tự =
    border = "=" * 60
    header = f"{border}\n🤖 ĐANG KIỂM TRA MODEL: {model_name}\n{border}\n"
    print(header)
    model_logs.append(header)
    
    try:
        router_llm = get_router_llm(model_name)
        router = IntentRouter(router_llm)
    except Exception as e:
        err_msg = f"💥 Lỗi khởi tạo IntentRouter với model {model_name}: {e}\n"
        print(err_msg)
        model_logs.append(err_msg)
        return {"model": model_name, "score": "ERROR", "accuracy": "0.00%", "latency": "N/A"}, "".join(model_logs)

    passed_count = 0
    total_cases = len(test_cases)
    start_time = time.time()

    for idx, case in enumerate(test_cases, 1):
        query = case["query"]
        expected = case["expected"]
        
        try:
            predicted = router.classify(query)
            is_correct = predicted == expected
            status = "✅ ĐÚNG" if is_correct else f"❌ SAI (Predict: {predicted})"
            
            if is_correct:
                passed_count += 1
                
            line = f"Câu {idx:02d}: {status} | '{query}'"
            print(line)
            model_logs.append(line)
        except Exception as e:
            err_line = f"Câu {idx:02d}: 💥 LỖI KHI XỬ LÝ: {e} | '{query}'"
            print(err_line)
            model_logs.append(err_line)

    total_time = time.time() - start_time
    accuracy = (passed_count / total_cases) * 100
    
    summary_line = f"\n📊 Kết quả [{model_name}]: {passed_count}/{total_cases} câu đúng ({accuracy:.2f}%) | Time: {total_time:.2f}s\n"
    print(summary_line)
    model_logs.append(summary_line)
    
    metrics = {
        "model": model_name,
        "score": f"{passed_count}/{total_cases}",
        "accuracy": f"{accuracy:.2f}%",
        "latency": f"{total_time:.2f}s"
    }
    
    # Trả về cả dictionary chỉ số và toàn bộ đoạn text log của model này
    return metrics, "\n".join(model_logs)

# --- 4. LUỒNG CHẠY CHÍNH (MAIN PROCESS) ---
if __name__ == "__main__":
    load_dotenv()
    
    models = (
        "openai/gpt-4o-mini", #0,15 input
        "meta-llama/llama-3.1-8b-instruct", # 0.02 input/ 0.05 groq
        "google/gemma-3-27b-it", #0.08 input 
        "qwen/qwen3-8b",# 0,05 input
    )
    
    benchmark_results = []
    full_report_text = []  # Bộ nhớ lưu trữ toàn bộ chuỗi ký tự xuất ra file
    
    print("🚀 Bắt đầu khởi động chương trình kiểm thử Router...")
    
    for model in models:
        result, model_text_log = evaluate_model(model, TEST_CASES)
        benchmark_results.append(result)
        full_report_text.append(model_text_log)
        full_report_text.append("\n") # Tạo khoảng cách giữa các model

    # --- 5. TẠO BẢNG TỔNG HỢP KẾT QUẢ CUỐI CÙNG ---
    table_lines = []
    table_lines.append("🏆" + " BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK ".center(68, "=") + "🏆")
    table_lines.append(f"{'Model Name':<36} | {'Score':<8} | {'Accuracy':<10} | {'Latency':<8}")
    table_lines.append("-" * 72)
    for res in benchmark_results:
        table_lines.append(f"{res['model']:<36} | {res['score']:<8} | {res['accuracy']:<10} | {res['latency']:<8}")
    table_lines.append("=" * 72)
    
    final_table_text = "\n".join(table_lines)
    print("\n" + final_table_text)
    full_report_text.append(final_table_text)

    # --- 6. XUẤT THẲNG RA FILE TEXT BÁO CÁO ---
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file_path = os.path.join(output_dir, f"benchmark_raw_report_{timestamp}.txt")
    
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write("".join(full_report_text))
        
    print(f"\n🎉 Đã xuất nguyên trạng báo cáo thành công tại: {report_file_path}")