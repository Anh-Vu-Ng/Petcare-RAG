FROM python:3.13-slim

# Cài đặt các công cụ hệ thống cần thiết và curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Astral uv cho việc quản lý packages cực nhanh
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Thêm uv vào PATH của hệ thống
ENV PATH="/root/.local/bin/:$PATH"

# Thiết lập thư mục làm việc chính trong container
WORKDIR /app

# Copy các file mô tả dependencies trước để tận dụng Docker cache layer
COPY pyproject.toml uv.lock ./

# Cài đặt dependencies hệ thống trước (không cài package chính dạng editable)
RUN uv sync --frozen --no-install-project

# Copy toàn bộ mã nguồn của dự án vào container
COPY src/ ./src/
COPY data/ ./data/

# Thiết lập biến môi trường bắt buộc
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1

# Cổng mặc định của FastAPI
EXPOSE 8000

# Lệnh khởi chạy server FastAPI sử dụng uv
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
