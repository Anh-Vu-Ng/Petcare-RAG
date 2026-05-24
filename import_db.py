import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Đảm bảo import được các module từ src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.service_db import ServiceDB
from src.config import DATABASE_URL

def main():
    # Cấu hình encoding để in ký tự tiếng Việt trên Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    print("=== KHỞI CHẠY KHỞI TẠO DATABASE ===")
    print(f"DATABASE_URL hiện tại: {DATABASE_URL}")
    
    # Kiểm tra xem có đang dùng SQLite fallback hay Supabase
    if DATABASE_URL.startswith("sqlite"):
        print("⚠️ Cảnh báo: DATABASE_URL đang dùng SQLite fallback. Nếu muốn dùng Supabase, hãy cập nhật DATABASE_URL trong file .env trước.")
    else:
        print("📡 Đang kết nối tới PostgreSQL (Supabase/Production)...")
        
    try:
        db = ServiceDB()
        print("🔨 Đang khởi tạo các bảng (services, chat_history)...")
        db.init_db()
        print("📥 Đang import dữ liệu bảng giá từ file CSV...")
        db.import_from_csv(force=True)
        print("🎉 Hoàn thành khởi tạo dữ liệu database thành công!")
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi trong quá trình khởi tạo: {e}")

if __name__ == "__main__":
    main()
