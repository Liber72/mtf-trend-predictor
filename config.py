"""
Config Module
Tập trung tất cả các hằng số (magic numbers) của dự án
"""

# ========== DATA PROCESSING ==========
LOOKBACK = 48                    # Số nến nhìn lại để dự đoán
STEP = 1                         # Bước nhảy giữa các sequences
SCALER_WINDOW = 300              # Kích thước cửa sổ trượt cho MinMaxScaler
TRAIN_RATIO = 0.8                # Tỉ lệ dữ liệu train (0.8 = 80% train, 20% val)

# ========== TECHNICAL INDICATORS ==========
ADX_WINDOW = 14                  # Chu kỳ ADX
MFI_WINDOW = 14                  # Chu kỳ MFI
MFI_DEFAULT_VALUE = 50           # Giá trị MFI mặc định khi không có volume
RSI_WINDOW = 14                  # Chu kỳ RSI
SMA_SHORT_WINDOW = 10            # Chu kỳ SMA ngắn
SMA_LONG_WINDOW = 20             # Chu kỳ SMA dài
CCI_WINDOW = 20                  # Chu kỳ CCI
CLOSE_POSITION_EPSILON = 1e-8   # Hệ số tránh chia cho 0 khi tính Close Position

# ========== FEATURE COLUMNS ==========
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close',
    'ADX', 'ADX_pos', 'ADX_neg',
    'MFI', 'RSI',
    'SMA_10', 'SMA_20',
    'CCI',
    'Price_Change', 'HL_Range', 'Close_Position'
]

# ========== LSTM MODEL ==========
LSTM_UNITS = (128, 64)           # Số units cho 2 lớp LSTM
DENSE_UNITS = 32                 # Số units cho lớp Dense ẩn
DROPOUT_RATE = 0.3               # Tỉ lệ dropout
LEARNING_RATE = 0.001            # Learning rate ban đầu
N_FEATURES = 15                  # Số features đầu vào
PREDICTION_THRESHOLD = 0.5       # Ngưỡng phân loại UP/DOWN

# ========== TRAINING CALLBACKS ==========
EPOCHS = 100                     # Số epochs mặc định
BATCH_SIZE = 32                  # Batch size mặc định
EARLY_STOPPING_PATIENCE = 15     # Patience cho EarlyStopping
REDUCE_LR_FACTOR = 0.5           # Hệ số giảm learning rate
REDUCE_LR_PATIENCE = 5           # Patience cho ReduceLROnPlateau
REDUCE_LR_MIN = 1e-6             # Learning rate tối thiểu

# ========== TRADING ==========
DEFAULT_SYMBOL = "XAUUSD"        # Symbol mặc định
ALTERNATIVE_SYMBOLS = ["XAUUSDm", "GOLD", "XAUUSD.a", "XAUUSD_i"]  # Symbols thay thế
DEFAULT_LOT = 0.1                # Lot size mặc định
DEFAULT_SL_PIPS = 500            # Stop Loss mặc định (pips)
DEFAULT_TP_PIPS = 500            # Take Profit mặc định (pips)
MAX_POSITIONS = 3                # Số lệnh tối đa được phép mở
MIN_CONFIDENCE = 0.5             # Độ tin cậy tối thiểu để vào lệnh
MAGIC_NUMBER = 12345             # Magic number nhận diện lệnh bot
ORDER_DEVIATION = 20             # Độ trượt giá cho phép (points)
PIP_MULTIPLIER = 10              # 1 pip = 10 points cho XAUUSD
XAUUSD_POINT = 0.01              # Point value cho XAUUSD

# ========== MODEL MODE ==========
MODEL_MODE_DUAL = "dual"              # Dùng cả M5 + H1 (đồng thuận)
MODEL_MODE_SINGLE_M5 = "single_m5"   # Chỉ dùng M5
DEFAULT_MODEL_MODE = MODEL_MODE_DUAL  # Chế độ mặc định

# ========== TRAILING STOP LOSS ==========
DEFAULT_TRAILING_SL_LEVELS = [
    (100, 5),                    # Khi lời 100 pips → SL = +5 pips
    (200, 100),                  # Khi lời 200 pips → SL = +100 pips
    (300, 200),                  # Khi lời 300 pips → SL = +200 pips
    (400, 300),                  # Khi lời 400 pips → SL = +300 pips
]
TRAILING_CHECK_INTERVAL = 1      # Kiểm tra trailing mỗi N giây

# ========== BACKTEST ==========
BACKTEST_INITIAL_EQUITY = 10000.0  # Vốn ban đầu cho backtest
BACKTEST_PROFIT_MULTIPLIER = 10    # Hệ số tính profit: profit_pips * lot * N

# ========== DATA CRAWL ==========
CRAWL_DEFAULT_LOOKBACK_DAYS = 180  # Số ngày mặc định khi crawl data

# ========== UI / APP ==========
TRADE_LOG_MAX_MESSAGES = 50       # Số messages tối đa trong trade log
REALTIME_DATA_COUNT = 350         # Số nến lấy realtime (>= SCALER_WINDOW)
MAX_TRADE_LOG_DISPLAY = 10        # Số messages hiển thị trên UI
AUTO_TRADE_INTERVAL = 0.5         # Khoảng thời gian giữa mỗi lần bot kiểm tra & vào lệnh (giây)
UI_REFRESH_INTERVAL = 3           # Khoảng thời gian giao diện tự refresh hiển thị (giây)

# ========== TCP COMMUNICATION ==========
TCP_HOST = "127.0.0.1"            # Host cho TCP server (Bot)
TCP_PORT = 5005                   # Port cho TCP server
TCP_BUFFER_SIZE = 65536           # Kích thước buffer nhận dữ liệu (bytes)

# ========== MODELS DIRECTORY ==========
MODELS_DIR = "models"             # Thư mục lưu models
