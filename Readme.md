# MTF Trend Predictor 📈🤖

**MTF Trend Predictor** (Multi-Timeframe Trend Predictor) là một hệ thống giao dịch tự động trên nền tảng **MetaTrader 5 (MT5)**, sử dụng mô hình học sâu **LSTM (Long Short-Term Memory)** để dự đoán xu hướng giá. Hệ thống phân tích đồng thời 2 khung thời gian (H1 và M5) cùng với các chỉ báo kỹ thuật để đưa ra quyết định giao dịch Buy/Sell một cách chính xác nhất.

## 🌟 Tính năng nổi bật

- **Kiến trúc Multi-Timeframe**: Sử dụng hai mô hình học sâu LSTM riêng biệt dự đoán xu hướng cho khung thời gian H1 và M5. Hệ thống chỉ vào lệnh khi cả hai khung thời gian có chung dự đoán (VD: H1 dự đoán Tăng + M5 dự đoán Tăng -> Buy).
- **Hệ thống Đặc trưng (Features) phong phú**: Tích hợp các chỉ báo kỹ thuật tối ưu như: Open, High, Low, Close, ADX, MFI, RSI, SMA, CCI, Price Change, HL Range.
- **Giao dịch Tự động trên MT5**: Tự động kết nối, lấy dữ liệu realtime và vào lệnh trực tiếp thông qua thư viện `MetaTrader5`.
- **Quản lý rủi ro nâng cao**: Tích hợp tính năng Trailing Stop Loss linh hoạt (kéo SL linh động theo nhiều mức lời) bên cạnh Stop Loss (SL) và Take Profit (TP) cố định.
- **Bảng điều khiển trực quan (Dashboard)**: Quản lý toàn bộ hệ thống bằng giao diện người dùng **Streamlit**, cho phép theo dõi biểu đồ, tín hiệu, dữ liệu live và trạng thái mô hình.
- **Môi trường Tối ưu**: Hỗ trợ huấn luyện mô hình mạnh mẽ với kiến trúc tối ưu trên môi trường **TensorFlow GPU (`tf_gpu`)**.

## 📁 Cấu trúc Dự án

```text
├── app.py                  # Giao diện Streamlit quản lý toàn bộ hệ thống
├── backtester.py           # Công cụ Backtest chiến lược bằng dữ liệu lịch sử
├── config.py               # Chứa toàn bộ các tham số cấu hình (Magic numbers)
├── crawldata_MT5.py        # Kịch bản tải dữ liệu nến M5, H1 lịch sử từ MT5
├── data_processor.py       # Tính toán các chỉ báo kỹ thuật, chuẩn hoá dữ liệu LSTM
├── lstm_model.py           # Định nghĩa kiến trúc mạng LSTM
├── mt5_trader.py           # Module giao tiếp trực tiếp với MT5 để vào lệnh, chốt lời/cắt lỗ
├── trainer.py              # Thành phần chuyên biệt dùng để huấn luyện và đánh giá mô hình
├── Readme.md               # File tài liệu dự án
└── requirements.txt        # Các thư viện phụ thuộc Python
```

## ⚙️ Yêu cầu Hệ thống

Dự án bắt buộc phải chạy trên môi trường có thiết lập **TensorFlow GPU** (`tf_gpu`) do giới hạn và yêu cầu tính toán lớn của mạng LSTM cho dữ liệu chuỗi thời gian.

- **OS**: Windows (Bắt buộc do thư viện `MetaTrader5` trên Python chỉ hỗ trợ hệ điều hành Windows)
- **Nền tảng giao dịch**: Cần có MetaTrader 5 Terminal bản Desktop chạy nền.
- **Python**: 3.8 - 3.10 

### Các gói Môi trường
Tham khảo file `requirements.txt`:
- `tensorflow` (Cần cài bản có GPU)
- `MetaTrader5`
- `streamlit`
- `pandas`, `numpy`, `scikit-learn`, `ta`

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Cài đặt thư viện

Khuyến nghị khởi tạo một môi trường ảo (Virtual Environment) sử dụng Anaconda hoặc venv trước khi thực hiện.
```bash
pip install -r requirements.txt
```

### 2. Thu thập dữ liệu

Trước khi huấn luyện mô hình, hệ thống cần được tải dữ liệu (ví dụ với `XAUUSD` khung H1, M5):
```bash
python crawldata_MT5.py
```
*Lưu ý: Bật sẵn phần mềm MetaTrader 5 Terminal trên máy tính, đăng nhập tài khoản rồi mới chạy script.*

### 3. Khởi động Bảng điều khiển (Streamlit Dashboard)

Đây là ứng dụng trọng tâm của quy trình. Khởi động Command Line/Terminal và chạy lệnh:
```bash
streamlit run app.py
```

Tại giao diện Web Dashboard hiển thị trên trình duyệt (thường là http://localhost:8501), bạn có thể:
1. **Huấn luyện (Train)**: Xây dựng và huấn luyện mô hình H1, M5 trên bộ dữ liệu vừa crawl.
2. **Backtest**: Kiểm thử mô hình trên dữ liệu quá khứ chưa từng thấy.
3. **Giao dịch Tự động (Auto Trade)**: Theo dõi nến realtime và uỷ quyền cho BOT thực hiện các tác vụ trade tự động lên tài khoản MT5 hiện hành.

## 📊 Tham số Kiến trúc Mô hình (Core Parameters)

Các chỉ số dưới đây được thiết lập trong thư mục `config.py`:
- **Loại Mạng**: 2 lớp LSTM sequence-to-vector (`LSTM_UNITS=(128, 64)`) kết hợp với lớp `Dense`, `Dropout(0.3)`.
- **Dữ liệu Chuỗi (Lookback)**: Mặc định nhìn lại quá khứ là **48 cây nến** gần nhất.
- **Scaler Window**: **300** cho kỹ thuật trượt MinMaxScaler chuyên biệt.
- **Huấn luyện**:
  - Epochs: 100
  - Batch Size: 32
  - Callbacks: Có `EarlyStopping` và `ReduceLROnPlateau`.

## ⚠️ Lưu ý Cảnh báo rủi ro (Disclaimer)

Dự án này là một công cụ nghiên cứu kiểm thử tín hiệu AI và hỗ trợ giao dịch, **KHÔNG ĐƯỢC XEM LÀ LỜI KHUYÊN HAY CHỈ ĐỊNH ĐẦU TƯ TÀI CHÍNH**. Giao dịch tài chính đòn bẩy cao (đặc biệt là Forex, Gold) phân bổ rủi ro cực lớn và bạn có toàn quyền quyết định về tài chính. **Hãy thử nghiệm kỹ lượng ở tài khoản DEMO** trước khi quyết định cấp bất cứ quyền giao dịch thực tới bot trên môi trường tiền thật (Real).
