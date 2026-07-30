# MTF Trend Predictor 📈🤖

**MTF Trend Predictor** (Multi-Timeframe Trend Predictor) là một hệ thống giao dịch tự động trên nền tảng **MetaTrader 5 (MT5)**, sử dụng mô hình học sâu **LSTM (Long Short-Term Memory)** để dự đoán xu hướng giá. Hệ thống phân tích đồng thời 2 khung thời gian (H1 và M5) cùng với các chỉ báo kỹ thuật để đưa ra quyết định giao dịch Buy/Sell một cách chính xác nhất. 

**Kết quả thử nghiệm mô hình tham khảo Test_reamtime.pdf**
Kết quả thử nghiệm mô hình: Winrate 64,36%(428/653), Average Profit: 57.96$, Netto P/L: 14 284.32

## 🌟 Tính năng nổi bật

- **Kiến trúc Multi-Timeframe**: Sử dụng hai mô hình học sâu LSTM riêng biệt dự đoán xu hướng cho khung thời gian H1 và M5. 
- **Hệ thống Đặc trưng (Features) phong phú**: Tích hợp các chỉ báo kỹ thuật tối ưu như: Open, High, Low, Close, ADX, MFI, RSI, SMA, CCI, Price Change, HL Range.
- **Giao dịch Tự động trên MT5**: Tự động kết nối, lấy dữ liệu realtime và vào lệnh trực tiếp thông qua thư viện `MetaTrader5`.
- **Quản lý rủi ro nâng cao**: Tích hợp tính năng Trailing Stop Loss linh hoạt bên cạnh Stop Loss (SL) và Take Profit (TP) cố định.
- **FastAPI Backend (Mới)**: Kiến trúc Modular Monolith với REST API chuẩn, PostgreSQL, và WebSockets cho khả năng mở rộng.
- **Bảng điều khiển trực quan (Dashboard)**: Quản lý toàn bộ hệ thống bằng giao diện người dùng **Streamlit**.

## ⚙️ Yêu cầu Hệ thống

Dự án bắt buộc phải chạy trên môi trường có thiết lập **TensorFlow GPU** và **Windows** để dùng MT5.

- **OS**: Windows (Bắt buộc do thư viện `MetaTrader5` trên Python chỉ hỗ trợ hệ điều hành Windows). *Lưu ý: Docker có thể dùng chạy DB, nhưng Backend gọi MT5 phải chạy trên Windows Host.*
- **Nền tảng giao dịch**: Cần có MetaTrader 5 Terminal bản Desktop chạy nền.
- **Python**: 3.11 

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Cài đặt môi trường

Sử dụng `uv` hoặc `pip` với Python 3.11:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Cấu hình Database (PostgreSQL)

Hệ thống sử dụng PostgreSQL để lưu lịch sử trade, prediction, và model version.
Bạn có thể dùng Docker để chạy DB nhanh chóng:
```bash
docker-compose up -d db
```

Sau đó tạo bảng bằng Alembic:
```bash
alembic upgrade head
```

### 3. Khởi động FastAPI Server

Mở Terminal 1 và chạy backend:
```bash
uvicorn src.main:app --reload --port 8000
```
- API Docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Các tính năng có sẵn: Crawl MT5, Import CSV, Train LSTM, Auto-Trade.

### 4. Khởi động Bảng điều khiển (Streamlit)

Mở Terminal 2 và chạy giao diện:
```bash
streamlit run src/apps/dashboard/app.py
```
- Dashboard: `http://localhost:8501`

## 📡 Danh sách API Endpoints chính

- `POST /api/v1/market-data/crawl`: Tải dữ liệu lịch sử từ MT5
- `POST /api/v1/market-data/import`: Import file CSV vào database PostgreSQL
- `POST /api/v1/models/train`: Huấn luyện mô hình LSTM (chạy ngầm)
- `GET /api/v1/models`: Danh sách các mô hình đã train
- `POST /api/v1/trading/auto/start`: Kích hoạt Bot giao dịch tự động
- `WS /api/v1/ws/trades`: WebSocket stream log giao dịch realtime

