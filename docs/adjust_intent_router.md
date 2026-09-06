

# 📊 BÁO CÁO KẾT QUẢ TEST VÀ TỐI ƯU PROMPT — INTENT ROUTER

## 1. Prompt 
```python
  /no_think
  Classify the user message into exactly one category. Reply with ONLY one word.
  
  GREETING — chào hỏi, tạm biệt, tán gẫu, hỏi bạn là ai, vô nghĩa, lạc đề, không hỏi gì rõ ràng
  TOOL — hỏi giá/tiền/chi phí/bill của đúng 6 dịch vụ này: tắm, cạo lông, cắt/mài móng, lưu trú/gửi thú cưng, nặn tuyến hôi, vệ sinh/lấy ráy tai. Hỏi giá thứ KHÁC → KNOWLEDGE
  KNOWLEDGE — hỏi thông tin/tư vấn thú cưng, bệnh, dinh dưỡng, địa chỉ/vị trí/giờ mở cửa shop, cơ sở vật chất, dịch vụ ngoài danh sách TOOL, giá sản phẩm bán lẻ
  Nếu không rõ TOOL hay KNOWLEDGE → GREETING
  
  EXAMPLES:
  "Hello shop" → GREETING
  "aoqieuqiuw@!@@#!#$#!!4" → GREETING
  "Bạn là robot hay người?" → GREETING
  "Cho mình mượn tiền đi cạo lông cún" → GREETING
  "Tạm biệt nhé" → GREETING
  "Kể chuyện hài về mèo" → GREETING
  "Cún nhà mình nghịch quá haha" → GREETING
  "Tắm chó 10kg giá bao nhiêu?" → TOOL
  "Gửi mèo 3kg qua đêm tính sao?" → TOOL
  "Cạo lông poodle 2.5kg hết bao lúa?" → TOOL
  "Combo tắm cắt móng cho cún 5kg giá sao?" → TOOL
  "Khách sạn thú cưng 3 ngày giá nhiêu?" → TOOL
  "Shop báo bill cắt móng cho Poodle để mình đưa chồng coi thử." → TOOL
  "Khám chữa bệnh giá sao vậy shop?" → KNOWLEDGE
  "Tiêm phòng cho chó giá bao nhiêu?" → KNOWLEDGE
  "Chó mấy tháng tiêm phòng?" → KNOWLEDGE
  "Mèo nôn bọt trắng bị gì?" → KNOWLEDGE
  "Shop có những dịch vụ gì?" → KNOWLEDGE
  "Shop dời địa chỉ qua quận 7 rồi hả, cho xin lại định vị nha." → KNOWLEDGE
  "Khách sạn thú cưng chuồng có máy lạnh không?" → KNOWLEDGE
  "Túi vận chuyển phi hành gia cho mèo giá bao nhiêu?" → KNOWLEDGE
  "Mấy giờ shop đóng cửa không nhận gửi chó nữa?" → KNOWLEDGE

```
## 2. Kết quả 
```diff
🏆================= BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK ==================🏆
Model Name                           | Score    | Accuracy   | Latency 
------------------------------------------------------------------------
openai/gpt-4o-mini                   | 48/49    | 97.96%     | 69.56s  
meta-llama/llama-3.1-8b-instruct     | 37/49    | 75.51%     | 43.25s  
google/gemma-3-27b-it                | 46/49    | 93.88%     | 55.22s  
qwen/qwen3-8b                        | 45/49    | 91.84%     | 60.25s  
========================================================================
```
## 3. Prompt sau chỉnh sửa 
```python
"""/no_think
Classify the user message into exactly one category. Reply with ONLY one word.

GREETING — chào hỏi, tạm biệt, tán gẫu, hỏi bạn là ai, vô nghĩa, lạc đề, không hỏi gì rõ ràng
TOOL — hỏi giá/tiền/chi phí/bill của đúng 6 dịch vụ này: tắm, cạo lông, cắt/mài móng, lưu trú/gửi thú cưng, nặn tuyến hôi, vệ sinh/lấy ráy tai. Hỏi giá thứ KHÁC → KNOWLEDGE
KNOWLEDGE — hỏi thông tin/tư vấn thú cưng, bệnh, dinh dưỡng, địa chỉ/vị trí/giờ mở cửa shop, cơ sở vật chất, dịch vụ ngoài danh sách TOOL, giá sản phẩm bán lẻ

⚠️ QUAN TRỌNG: Người dùng thường kể lể, vòng vo ở đầu câu rồi mới đưa ra yêu cầu thật ở cuối. Hãy đọc TOÀN BỘ câu, xác định yêu cầu thật ở cuối, rồi mới classify.

Nếu không rõ TOOL hay KNOWLEDGE → GREETING

EXAMPLES:
"Hello shop" → GREETING
"aoqieuqiuw@!@@#!#$#!!4" → GREETING
"Bạn là robot hay người?" → GREETING
"Cho mình mượn tiền đi cạo lông cún" → GREETING
"Tạm biệt nhé" → GREETING
"Kể chuyện hài về mèo" → GREETING
"Cún nhà mình nghịch quá haha" → GREETING
"Tắm chó 10kg giá bao nhiêu?" → TOOL
"Gửi mèo 3kg qua đêm tính sao?" → TOOL
"Cạo lông poodle 2.5kg hết bao lúa?" → TOOL
"Combo tắm cắt móng cho cún 5kg giá sao?" → TOOL
"Khách sạn thú cưng 3 ngày giá nhiêu?" → TOOL
"Shop báo bill cắt móng cho Poodle để mình đưa chồng coi thử." → TOOL
"Mèo mình bị tiêu chảy mấy nay, dơ quá, đem qua tắm thì giá sao?" → TOOL
"Chồng mình bảo cắt móng Poodle tốn tiền lắm, shop báo bill cho mình xem thử." → TOOL
"Tuần sau đi du lịch, tính gửi 2 con 5kg ở shop 4 hôm, hết bao nhiêu lúa?" → TOOL
"Khám chữa bệnh giá sao vậy shop?" → KNOWLEDGE
"Tiêm phòng cho chó giá bao nhiêu?" → KNOWLEDGE
"Chó mấy tháng tiêm phòng?" → KNOWLEDGE
"Mèo nôn bọt trắng bị gì?" → KNOWLEDGE
"Shop có những dịch vụ gì?" → KNOWLEDGE
"Shop dời địa chỉ qua quận 7 rồi hả, cho xin lại định vị nha." → KNOWLEDGE
"Khách sạn thú cưng chuồng có máy lạnh không?" → KNOWLEDGE
"Túi vận chuyển phi hành gia cho mèo giá bao nhiêu?" → KNOWLEDGE
"Mấy giờ shop đóng cửa không nhận gửi chó nữa?" → KNOWLEDGE
"""
```
## 4. Kết quả sau chỉnh sửa Prompt
```diff
🏆================= BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK ==================🏆
Model Name                           | Score    | Accuracy   | Latency 
------------------------------------------------------------------------
openai/gpt-4o-mini                   | 49/49    | 100.00%    | 58.39s  
meta-llama/llama-3.1-8b-instruct     | 45/49    | 91.84%     | 45.71s  
google/gemma-3-27b-it                | 48/49    | 97.96%     | 57.93s  
qwen/qwen3-8b                        | 49/49    | 100.00%    | 67.06s  
========================================================================
```