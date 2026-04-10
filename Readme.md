- Mô hình AI:
    + Mô hình dùng nến H1 và nến M5 để dự đoán xu hướng của nến tiếp theo.
    + Hai mô hình dư đoán H1 và M5 có cùng một khoảng thời gian huấn luyện.
    + Hai mô hình đều được huấn luyện một bộ chỉ số:
        + Ví dụ : Open,High,Low,Close,ADX,MF,RSI,SMA, CCI.
    + Hai mô hình sẽ dự đoán xu hướng nến tiếp theo xem tăng hay giảm, nếu cùng dự đoán tăng thì buy, nếu cùng dự đoán giảm thì sell.
    + Dữ liệu dùng Crawldata/crawldata_MT5.py để crawl dữ liệu từ MetaTrader5.
    + Mô hình có cấu trúc mạng LSTM.
- Mô hình H1:
    + Chỉ nhìn dữ liệu H1
    + Lookback =48
    + Step = 1  
    + Epoch = 100
    + Batch_size = 32
- Mô hình M5:
    + Chỉ nhìn dữ liệu M5
    + Lookback =48
    + Step = 1  
    + Epoch = 100
    + Batch_size = 32
- Data processor
    + scaler window =300




