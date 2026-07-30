.venv\Scripts\activate
# MTF Trend Predictor 📈🤖

**MTF Trend Predictor** (Multi-Timeframe Trend Predictor) là một hệ thống giao dịch tự động trên nền tảng **MetaTrader 5 (MT5)**, sử dụng mô hình học sâu **LSTM (Long Short-Term Memory)** để dự đoán xu hướng giá. Hệ thống phân tích đồng thời 2 khung thời gian (H1 và M5) cùng với các chỉ báo kỹ thuật để đưa ra quyết định giao dịch Buy/Sell một cách chính xác nhất.

## 🌟 Tính năng nổi bật

- **Kiến trúc Multi-Timeframe**: Sử dụng hai mô hình học sâu LSTM riêng biệt dự đoán xu hướng cho khung thời gian H1 và M5 theo cơ chế đồng thuận tuyệt đối.
- **Hệ thống Đặc trưng (Features)**: Tích hợp các chỉ báo kỹ thuật tối ưu như: Open, High, Low, Close, ADX, MFI, RSI, SMA, CCI, Price Change, HL Range.
- **Giao dịch Tự động trên MT5**: Tự động kết nối, lấy dữ liệu realtime và vào lệnh trực tiếp thông qua thư viện `MetaTrader5`.
- **Quản lý rủi ro nâng cao**: Tích hợp tính năng Trailing Stop Loss linh hoạt (kéo SL linh động theo nhiều mức lời) bên cạnh Stop Loss (SL) và Take Profit (TP) cố định.
- **Bảng điều khiển trực quan (Dashboard)**: Quản lý toàn bộ hệ thống bằng giao diện người dùng qua **Streamlit**, cho phép theo dõi biểu đồ, tín hiệu, dữ liệu live và trạng thái mô hình.
- **Môi trường Tối ưu**: Kiến trúc tối ưu trên môi trường **TensorFlow GPU**.

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

```bash
pip install -r requirements.txt
```
### 2. Thu thập dữ liệu
```bash
python crawldata_MT5.py
```
*Lưu ý: Bật sẵn phần mềm MetaTrader 5 Terminal trên máy tính, đăng nhập tài khoản rồi mới chạy script.*
### 3. Khởi động Bảng điều khiển (Streamlit Dashboard)
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
(Lưu ý đây chỉ là tham số kiến trúc mô hình tham khảo)

