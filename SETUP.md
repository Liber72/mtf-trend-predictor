# Setup Guide

Tài liệu này gom toàn bộ phần cài đặt và khởi động dự án. Nếu bạn chỉ cần overview nhanh, xem `README.md`.

## Yêu cầu hệ thống

- **Windows** nếu muốn dùng đầy đủ tính năng MT5.
- **MetaTrader 5 Desktop** đang chạy nếu cần crawl realtime, predict realtime hoặc giao dịch thật/demo.
- **Python 3.11** cho backend.
- **Node.js** hoặc **Bun** cho frontend.
- **Docker Desktop** nếu muốn chạy PostgreSQL bằng `docker compose`.

## Lưu ý quan trọng về Docker và MT5

Thư viện `MetaTrader5` chỉ hoạt động trên Windows. Vì vậy:

- Chạy **backend local trên Windows** là cách đầy đủ nhất để dùng MT5.
- `docker compose` phù hợp nhất để chạy **PostgreSQL**.
- API container trong Docker có thể dùng cho các phần không phụ thuộc MT5, nhưng các endpoint gọi MT5 sẽ không hoạt động đúng trong môi trường Linux container.

## Cấu hình môi trường

Tạo file `.env` từ `.env.example`:

```bash
copy .env.example .env
```

Mặc định `.env.example` dùng:

```env
DB_HOST=127.0.0.1
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=autotrader
```

Lưu ý:

- Nếu backend chạy **local** và database chạy bằng `docker compose`, hãy đổi `DB_PORT=5433` vì `docker-compose.yml` map host port `5433 -> 5432` trong container.
- Backend sẽ tự đọc `.env` ở root project hoặc `backend/.env`.

Frontend có thể cấu hình thêm qua `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

Nếu không cấu hình, frontend mặc định gọi backend ở `http://localhost:8000`.

## Chạy local

### 1. Chạy PostgreSQL

```bash
docker compose up -d db
docker compose ps db
```

Nếu backend chạy local, nhớ cập nhật `.env` thành `DB_PORT=5433`.

### 2. Cài backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Chạy migration

```bash
cd backend
alembic upgrade head
```

### 4. Khởi động backend

```bash
cd backend
uvicorn src.main:app --reload --port 8000
```

Sau khi chạy:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Health check: `http://127.0.0.1:8000/api/v1/health`

### 5. Cài và chạy frontend

Với npm:

```bash
cd frontend
npm install
npm run dev
```

Hoặc với Bun:

```bash
cd frontend
bun install
bun run dev
```

Frontend dev server chạy ở `http://localhost:5173`.

## Chạy bằng Docker

Chạy database:

```bash
docker compose up -d db
```

Chạy cả API container:

```bash
docker compose up -d api
```

Nhắc lại: chế độ này **không phù hợp cho các tính năng MT5** vì container Linux không truy cập được thư viện MT5 như khi chạy local trên Windows.

## Kiểm tra nhanh

Backend:

```bash
cd backend
python -m unittest discover -s tests
```

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

Nếu dùng Bun, bạn cũng có thể chạy test helper hiện có bằng:

```bash
cd frontend
bun test src/lib/backend-contract.test.ts
```
