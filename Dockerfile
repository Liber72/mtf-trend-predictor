# Sử dụng Python 3.11 slim image
FROM python:3.11-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    PYTHONPATH=/app

# Cài đặt system dependencies cần thiết
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn pydantic python-multipart asyncpg

# Copy toàn bộ mã nguồn
COPY . .

# Mở port cho FastAPI
EXPOSE 8000

# Chạy FastAPI server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# LƯU Ý: MetaTrader5 library (mt5) chỉ hoạt động trên Windows.
# Khi chạy trong Docker (Linux), các tính năng gọi MT5 sẽ báo lỗi.
# Docker phù hợp để test DB, train models, và chạy Dashboard.
