# 📊 Kế Hoạch Chuyển Đổi Sang Cơ Sở Dữ Liệu — LSTM Trading System

## 1. Phân Tích Hiện Trạng

### 1.1 Dữ liệu hiện tại đang lưu trữ bằng gì?

| Loại dữ liệu | Định dạng hiện tại | Kích thước | File / Vị trí |
|---|---|---|---|
| **Dữ liệu nến OHLCV (M5)** | CSV | ~39 MB | `XAUUSD_M5_*.csv` |
| **Dữ liệu nến OHLCV (H1)** | CSV | ~3.6 MB | `XAUUSD_H1_*.csv` |
| **Trained Models** | Keras (.keras) | ~1.5 MB/model | `models/*.keras` |
| **Scaler Config** | Pickle (.pkl) | ~190 bytes | `models/*.pkl` |
| **Validation Results** | CSV | ~9.3 MB (M5) | `models/*_validation_results.csv` |
| **Trade Log** | In-memory (list) | Runtime only | `MT5Trader.trade_log` |
| **Auto Trade Messages** | In-memory (list) | Runtime only | `MT5Trader.auto_trade_messages` |

### 1.2 Các điểm yếu của hệ thống hiện tại (CSV-based)

- ❌ **Không có persistence cho trade log** — Khi restart app, toàn bộ lịch sử giao dịch bị mất
- ❌ **Không truy vấn được** — Không thể filter/aggregate dữ liệu nến theo thời gian hiệu quả
- ❌ **File lock issues** — CSV không hỗ trợ concurrent read/write (Streamlit UI + Bot thread)
- ❌ **Duplicate data** — Mỗi lần crawl tạo file mới, không append/upsert
- ❌ **Không có indexing** — Scan toàn bộ file khi load (`pd.read_csv` đọc hết vào RAM)
- ❌ **Khó mở rộng** — Thêm symbol/timeframe mới = thêm file CSV mới
- ❌ **Không tracking model versions** — Không biết model nào train từ data nào, metrics ra sao qua thời gian

---

## 2. Các Loại Dữ Liệu Cần Lưu Trong Database

```mermaid
graph TD
    A[Database] --> B[📈 Market Data<br/>OHLCV candles]
    A --> C[🤖 Trade History<br/>Orders, P&L]
    A --> D[🧠 Model Registry<br/>Versions, Metrics]
    A --> E[📊 Predictions Log<br/>Signals, Confidence]
    A --> F[⚙️ System Config<br/>Parameters, Settings]
    A --> G[📝 Backtest Results<br/>Equity curves]
```

| Bảng (Table) | Mô tả | Ghi | Đọc | Tần suất |
|---|---|---|---|---|
| `candles` | Dữ liệu nến OHLCV | Crawl (batch) | Training, Prediction | Mỗi 5 phút (realtime) |
| `trades` | Lịch sử giao dịch | Mỗi khi mở/đóng lệnh | UI hiển thị | Real-time |
| `predictions` | Log dự đoán | Mỗi cycle auto-trade | Phân tích hiệu quả | Mỗi 0.5s |
| `model_versions` | Registry model | Sau mỗi lần train | Load model | Ít |
| `backtest_results` | Kết quả backtest | Sau mỗi lần backtest | UI hiển thị | Ít |
| `system_config` | Cấu hình hệ thống | Khi user thay đổi | Khi khởi động | Ít |

---

## 3. Tất Cả Các Phương Án Khả Thi

### Phương án 1: 🟢 SQLite

> **Embedded database, zero-config, single-file**

**Mô tả:** SQLite là RDBMS nhúng, toàn bộ DB nằm trong 1 file `.db`. Không cần server, không cần cài đặt.

**Cách tích hợp:**
```python
import sqlite3
conn = sqlite3.connect('trading_system.db')
# Hoặc dùng SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine('sqlite:///trading_system.db')
```

| Tiêu chí | Đánh giá |
|---|---|
| Hiệu năng đọc | ⭐⭐⭐⭐ Tốt cho dataset < 1GB |
| Hiệu năng ghi | ⭐⭐⭐ Chấp nhận được (single-writer lock) |
| Phức tạp triển khai | ⭐⭐⭐⭐⭐ Rất đơn giản, built-in Python |
| Khả năng mở rộng | ⭐⭐ Giới hạn (single machine, single writer) |
| Time-series support | ⭐⭐ Không có native, cần tự tạo index |
| Chi phí | ⭐⭐⭐⭐⭐ Miễn phí, 0 setup |
| Tích hợp Python | ⭐⭐⭐⭐⭐ Built-in `sqlite3` module |
| Phù hợp Windows | ⭐⭐⭐⭐⭐ Hoàn hảo |
| Real-time capability | ⭐⭐⭐ Chấp nhận được với WAL mode |

**✅ Ưu điểm so với các phương án khác:**
- **Zero deployment** — không cần cài gì thêm, Python có sẵn
- **Portable** — 1 file `.db`, dễ backup, dễ di chuyển
- **Đủ cho quy mô hiện tại** — 39MB M5 data + trade log rất nhẹ
- **WAL mode** cho phép concurrent reads khi 1 thread đang write
- **Tích hợp tuyệt vời với pandas** — `pd.read_sql()` / `df.to_sql()`

**❌ Nhược điểm:**
- Single writer lock — nếu bot thread ghi, UI thread phải đợi
- Không có time-series optimization
- Không scale nếu chạy nhiều bot instances

---

### Phương án 2: 🔵 PostgreSQL

> **Full-featured RDBMS, production-grade**

**Mô tả:** PostgreSQL là RDBMS mạnh nhất trong thế giới open-source. Cần chạy server riêng.

**Cách tích hợp:**
```python
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@localhost:5432/trading_db')
# Hoặc dùng psycopg2
import psycopg2
conn = psycopg2.connect("dbname=trading_db user=admin password=secret")
```

| Tiêu chí | Đánh giá |
|---|---|
| Hiệu năng đọc | ⭐⭐⭐⭐⭐ Rất tốt, advanced indexing |
| Hiệu năng ghi | ⭐⭐⭐⭐⭐ Concurrent writes, MVCC |
| Phức tạp triển khai | ⭐⭐ Cần cài & quản lý server |
| Khả năng mở rộng | ⭐⭐⭐⭐⭐ Rất cao |
| Time-series support | ⭐⭐⭐ Tốt với BRIN index + partitioning |
| Chi phí | ⭐⭐⭐⭐ Miễn phí, nhưng cần server resources |
| Tích hợp Python | ⭐⭐⭐⭐ psycopg2, SQLAlchemy |
| Phù hợp Windows | ⭐⭐⭐ Chạy được nhưng không phải native |
| Real-time capability | ⭐⭐⭐⭐⭐ LISTEN/NOTIFY, concurrent |

**✅ Ưu điểm so với các phương án khác:**
- **True concurrent access** — nhiều threads/processes đọc ghi cùng lúc
- **ACID compliance** — đảm bảo data integrity cho trade data
- **Advanced queries** — Window functions, CTEs, JSON support
- **Replication & Backup** — production-ready
- **Nền tảng cho TimescaleDB** — có thể nâng cấp sau

**❌ Nhược điểm:**
- **Overkill** cho 1 user, 1 machine, 1 symbol
- Cần cài đặt, cấu hình, maintain PostgreSQL server trên Windows
- Tốn RAM (tối thiểu ~256MB cho PostgreSQL server)

---

### Phương án 3: 🟣 TimescaleDB (PostgreSQL Extension)

> **Time-series database built on PostgreSQL — tối ưu cho dữ liệu nến**

**Mô tả:** TimescaleDB là extension của PostgreSQL, tối ưu hóa cho time-series data. Dữ liệu nến OHLCV chính là time-series data.

**Cách tích hợp:**
```python
# Giống PostgreSQL, thêm hypertable
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@localhost:5432/trading_db')

# SQL: CREATE TABLE candles (...);
# SQL: SELECT create_hypertable('candles', 'time');
```

| Tiêu chí | Đánh giá |
|---|---|
| Hiệu năng đọc | ⭐⭐⭐⭐⭐ Tối ưu cho time-range queries |
| Hiệu năng ghi | ⭐⭐⭐⭐⭐ Batch insert cực nhanh |
| Phức tạp triển khai | ⭐⭐ Cần PostgreSQL + Extension |
| Khả năng mở rộng | ⭐⭐⭐⭐⭐ Auto-partitioning theo time |
| Time-series support | ⭐⭐⭐⭐⭐ Native — sinh ra cho việc này |
| Chi phí | ⭐⭐⭐⭐ Miễn phí (Community Edition) |
| Tích hợp Python | ⭐⭐⭐⭐ Dùng psycopg2/SQLAlchemy |
| Phù hợp Windows | ⭐⭐⭐ Cần Docker hoặc WSL |
| Real-time capability | ⭐⭐⭐⭐⭐ Continuous aggregates, real-time |

**✅ Ưu điểm so với các phương án khác:**
- **Sinh ra cho time-series** — candle data là use case hoàn hảo
- **Auto-partitioning** — tự chia dữ liệu theo chunks thời gian
- **Continuous Aggregates** — tự động tính M5 → H1 → H4 → D1
- **Data retention policies** — tự xóa dữ liệu cũ
- **Compression** — giảm 90%+ storage cho dữ liệu cũ
- **Full PostgreSQL compatibility** — vẫn dùng SQL bình thường

**❌ Nhược điểm:**
- Phức tạp triển khai nhất trên Windows (recommend Docker)
- Overkill nếu chỉ có 1 symbol, 1 timeframe
- Learning curve cao hơn SQLite/DuckDB

---

### Phương án 4: 🟠 InfluxDB

> **Purpose-built time-series database**

**Mô tả:** InfluxDB là TSDB thuần túy, thiết kế cho metrics, IoT, monitoring data. Có thể dùng cho financial data.

**Cách tích hợp:**
```python
from influxdb_client import InfluxDBClient
client = InfluxDBClient(url="http://localhost:8086", token="my-token", org="my-org")
```

| Tiêu chí | Đánh giá |
|---|---|
| Hiệu năng đọc | ⭐⭐⭐⭐ Tốt cho time-range queries |
| Hiệu năng ghi | ⭐⭐⭐⭐⭐ Tối ưu cho high-frequency writes |
| Phức tạp triển khai | ⭐⭐ Cần cài server riêng |
| Khả năng mở rộng | ⭐⭐⭐⭐ Tốt (Enterprise) |
| Time-series support | ⭐⭐⭐⭐⭐ Native |
| Chi phí | ⭐⭐⭐ Miễn phí (OSS), trả phí (Cloud/Enterprise) |
| Tích hợp Python | ⭐⭐⭐ influxdb-client |
| Phù hợp Windows | ⭐⭐⭐ Có binary cho Windows |
| Real-time capability | ⭐⭐⭐⭐⭐ Flux queries, tasks |

**✅ Ưu điểm so với các phương án khác:**
- **Tối ưu cho write-heavy workloads** — ghi predictions mỗi 0.5s
- **Built-in downsampling** — tự aggregate dữ liệu
- **Flux query language** — mạnh cho time-series analytics
- **Retention policies** built-in

**❌ Nhược điểm:**
- **Không phải SQL** — dùng Flux/InfluxQL, learning curve riêng
- **Không lưu relational data tốt** — trade log, model registry cần DB riêng
- **Overkill** — thiết kế cho IoT/monitoring, không phải financial trading
- **Không tích hợp pandas native** — cần convert data format
- Cần chạy 2 DB (InfluxDB cho candles + SQLite/Postgres cho relational data)

---

### Phương án 5: 🟡 DuckDB

> **In-process OLAP database, "SQLite for Analytics"**

**Mô tả:** DuckDB là DB phân tích nhúng, cực nhanh cho analytical queries trên dữ liệu lớn. Đọc trực tiếp CSV/Parquet.

**Cách tích hợp:**
```python
import duckdb
conn = duckdb.connect('trading_system.duckdb')

# Đọc CSV trực tiếp
df = conn.execute("SELECT * FROM 'XAUUSD_M5_*.csv' WHERE Time > '2025-01-01'").fetchdf()

# Hoặc tạo table
conn.execute("CREATE TABLE candles AS SELECT * FROM 'XAUUSD_M5_*.csv'")
```

| Tiêu chí | Đánh giá |
|---|---|
| Hiệu năng đọc | ⭐⭐⭐⭐⭐ Cực nhanh cho analytics (columnar) |
| Hiệu năng ghi | ⭐⭐⭐ Tốt cho batch, chậm cho single-row |
| Phức tạp triển khai | ⭐⭐⭐⭐⭐ `pip install duckdb` |
| Khả năng mở rộng | ⭐⭐⭐ Single machine, but very efficient |
| Time-series support | ⭐⭐⭐ Tốt với window functions |
| Chi phí | ⭐⭐⭐⭐⭐ Miễn phí |
| Tích hợp Python | ⭐⭐⭐⭐⭐ Native Python API, pandas integration |
| Phù hợp Windows | ⭐⭐⭐⭐⭐ Hoàn hảo |
| Real-time capability | ⭐⭐ Thiết kế cho analytics, không phải OLTP |

**✅ Ưu điểm so với các phương án khác:**
- **Cực nhanh cho analytics** — tổng hợp, thống kê candle data
- **Đọc CSV/Parquet trực tiếp** — không cần import
- **Tích hợp pandas tuyệt vời** — `conn.execute().fetchdf()`
- **Zero-config** — `pip install duckdb`
- **Columnar storage** — nén tốt, scan nhanh

**❌ Nhược điểm:**
- **Không phù hợp OLTP** — ghi từng dòng trade log chậm
- **Single writer** — giống SQLite
- **Thiết kế cho batch analytics** — không phải real-time trading
- Concurrent access hạn chế

---

## 4. Bảng So Sánh Tổng Quan

| Tiêu chí | SQLite | PostgreSQL | TimescaleDB | InfluxDB | DuckDB |
|---|:---:|:---:|:---:|:---:|:---:|
| **Setup phức tạp** | ⭐ Rất dễ | ⭐⭐⭐⭐ Khó | ⭐⭐⭐⭐⭐ Rất khó | ⭐⭐⭐⭐ Khó | ⭐ Rất dễ |
| **Hiệu năng candles** | Tốt | Rất tốt | **Tốt nhất** | Rất tốt | Tốt nhất (đọc) |
| **Hiệu năng trade log** | Tốt | **Tốt nhất** | **Tốt nhất** | Trung bình | Kém |
| **Concurrent access** | Kém | **Tốt nhất** | **Tốt nhất** | Tốt | Kém |
| **Time-series native** | ❌ | ❌ | ✅ | ✅ | ❌ |
| **SQL support** | ✅ | ✅ | ✅ | ❌ (Flux) | ✅ |
| **Windows native** | ✅ | ⚠️ | ❌ (Docker) | ⚠️ | ✅ |
| **Cần server riêng** | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Chi phí vận hành** | $0 | Thấp | Thấp | Trung bình | $0 |
| **Phù hợp dự án này** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 5. Đề Xuất: Chiến Lược 2 Pha

### Phase 1: SQLite (Triển khai ngay — 1-2 ngày)

> **Mục tiêu:** Giải quyết các vấn đề cấp bách nhất mà không thay đổi kiến trúc lớn

```mermaid
graph LR
    subgraph "Phase 1 — SQLite"
        A[crawldata_MT5.py] -->|INSERT/UPSERT| B[(trading.db)]
        C[trainer.py] -->|READ candles| B
        C -->|INSERT model_versions| B
        D[mt5_trader.py] -->|INSERT trades| B
        D -->|INSERT predictions| B
        E[app.py / Streamlit] -->|READ all| B
    end
```

**Schema dự kiến:**

```sql
-- Bảng nến OHLCV (upsert by symbol + timeframe + time)
CREATE TABLE candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    time DATETIME NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    tick_volume INTEGER, spread INTEGER, real_volume INTEGER,
    UNIQUE(symbol, timeframe, time)
);
CREATE INDEX idx_candles_lookup ON candles(symbol, timeframe, time);

-- Bảng lịch sử giao dịch (persist trade log)
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time DATETIME NOT NULL,
    action TEXT NOT NULL,        -- OPEN, CLOSE, TRAILING_SL
    signal TEXT,                 -- BUY, SELL
    ticket INTEGER,
    price REAL,
    volume REAL,
    sl REAL, tp REAL,
    profit REAL,
    old_sl REAL, new_sl REAL,    -- cho trailing
    profit_pips REAL,
    level TEXT,                  -- trailing level
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Bảng log dự đoán
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time DATETIME NOT NULL,
    model_mode TEXT,             -- dual, single_m5
    h1_direction TEXT, h1_probability REAL,
    m5_direction TEXT, m5_probability REAL,
    combined_signal TEXT,        -- BUY, SELL, WAIT
    combined_confidence REAL,
    trade_executed INTEGER DEFAULT 0
);

-- Bảng model versions
CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timeframe TEXT NOT NULL,     -- H1, M5
    train_date DATETIME,
    data_file TEXT,
    train_ratio REAL,
    epochs INTEGER, batch_size INTEGER, lookback INTEGER,
    accuracy REAL, precision_val REAL, recall REAL, f1_score REAL,
    train_accuracy REAL, val_accuracy REAL,
    model_path TEXT,
    scaler_path TEXT,
    is_active INTEGER DEFAULT 1
);

-- Bảng backtest results
CREATE TABLE backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date DATETIME,
    model_mode TEXT,
    total_trades INTEGER,
    win_rate REAL,
    total_profit REAL,
    max_drawdown REAL,
    profit_factor REAL,
    config_json TEXT             -- JSON chứa SL/TP/confidence config
);
```

**Những gì cần thay đổi:**

| File | Thay đổi |
|---|---|
| `config.py` | Thêm `DATABASE_PATH = "trading_system.db"` |
| **Mới:** `database.py` | Module quản lý DB connection, init schema, CRUD operations |
| `crawldata_MT5.py` | Thêm `save_to_db()` bên cạnh `to_csv()` |
| `data_processor.py` | Thêm `load_from_db()` bên cạnh `load_data()` |
| `trainer.py` | Log model version vào DB sau khi train |
| `mt5_trader.py` | Persist trade_log và predictions vào DB |
| `backtester.py` | Lưu kết quả backtest vào DB |
| `app.py` | Đọc trade history, predictions từ DB |

**Effort:** ~1-2 ngày code, 0 setup infrastructure

---

### Phase 2: TimescaleDB (Khi cần scale — tương lai)

> **Nâng cấp khi:** Chạy nhiều symbols, nhiều bot instances, cần real-time dashboard

```mermaid
graph LR
    subgraph "Phase 2 — TimescaleDB"
        A[Crawlers x N] -->|BATCH INSERT| B[(TimescaleDB)]
        C[Bot Worker 1] -->|READ + WRITE| B
        D[Bot Worker 2] -->|READ + WRITE| B
        E[Streamlit Dashboard] -->|READ| B
        F[API Server] -->|READ + WRITE| B
        B -->|Continuous Aggregate| G[M5 → H1 → H4]
        B -->|Retention Policy| H[Auto cleanup > 2 years]
    end
```

**Điều kiện nâng cấp:**
- Chạy > 3 symbols đồng thời
- Cần nhiều bot instances chạy concurrent
- Data > 1GB
- Cần real-time aggregation (M5 → H1 tự động)

**Effort:** ~3-5 ngày, cần Docker trên Windows

---

## 6. Kết Luận & Khuyến Nghị

> [!IMPORTANT]
> **Khuyến nghị: Bắt đầu với SQLite (Phase 1)**
> 
> Với quy mô hiện tại (1 symbol XAUUSD, 2 timeframes, 1 bot instance, chạy local trên Windows), SQLite là lựa chọn tối ưu nhất. Nó giải quyết ngay 100% các vấn đề hiện tại mà không cần cài đặt hay quản lý infrastructure nào.

**Lý do chính:**
1. **Built-in Python** — `import sqlite3`, xong
2. **Đủ hiệu năng** — 39MB data + trade log = quá nhẹ cho SQLite
3. **WAL mode** — cho phép Streamlit đọc trong khi bot thread ghi
4. **Tích hợp pandas hoàn hảo** — `pd.read_sql()` thay cho `pd.read_csv()`
5. **Migration path rõ ràng** — Schema SQL giống nhau, dễ chuyển sang PostgreSQL/TimescaleDB sau

> [!TIP]
> **Nếu muốn thêm DuckDB cho analytics:**
> Có thể dùng kết hợp SQLite (OLTP: trade log, predictions) + DuckDB (OLAP: phân tích candle data, backtest). DuckDB có thể đọc trực tiếp file SQLite.

---

## 7. Bước Tiếp Theo

Nếu bạn đồng ý với Phase 1 (SQLite), tôi sẽ:

1. ✅ Tạo module `database.py` với schema và CRUD operations
2. ✅ Cập nhật `crawldata_MT5.py` để lưu vào DB (giữ nguyên CSV cho backward compatible)
3. ✅ Cập nhật `data_processor.py` để đọc từ DB
4. ✅ Cập nhật `mt5_trader.py` để persist trade log & predictions
5. ✅ Cập nhật `trainer.py` để log model versions
6. ✅ Cập nhật `app.py` để đọc history từ DB

**Bạn muốn chọn phương án nào? Hoặc bạn có câu hỏi gì thêm về bất kỳ phương án nào?**
