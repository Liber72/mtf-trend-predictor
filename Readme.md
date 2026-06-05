# MTF Trend Predictor

MTF Trend Predictor là một hệ thống giao dịch và giám sát cho chiến lược dự đoán xu hướng đa khung thời gian trên **MetaTrader 5**. Dự án hiện tại gồm:

- **Backend FastAPI** để crawl dữ liệu, train model, dự đoán, giao dịch và lưu lịch sử.
- **Frontend Operator Console** xây bằng **TanStack Start + React** để vận hành hệ thống qua giao diện web.
- **PostgreSQL** để lưu candles, model versions, predictions và trades.
- **Mô hình LSTM** cho hai khung thời gian chính: **H1** và **M5**.

## Tài liệu

- Setup và chạy dự án: `SETUP.md`
- Overview kỹ thuật nhanh: tài liệu này

## Hiện trạng dự án

- Hỗ trợ crawl dữ liệu từ MT5 vào database.
- Hỗ trợ import CSV từ thư mục `data/` hoặc đường dẫn tuyệt đối.
- Hỗ trợ train model LSTM theo timeframe và quản lý version model.
- Hỗ trợ realtime prediction ở 2 chế độ: `dual` và `single_m5`.
- Hỗ trợ kết nối MT5, giao dịch tay, auto trading và xem positions đang mở.
- Hỗ trợ WebSocket để monitor trade log và trạng thái model realtime.
- Đã có frontend nội bộ để vận hành, không còn là backend-only project.

## Kiến trúc

```text
backend/                   FastAPI app, Alembic, tests, ML/trading modules
  alembic/
  src/
    apps/api/endpoints/    REST + WebSocket endpoints
    core/                  settings, constants, logging, dependencies
    infrastructure/        DB session, repositories, middleware
    modules/               market_data, ml, trading, backtesting
    utils/
  tests/

frontend/                  TanStack Start + React operator console
  src/
    routes/                Dashboard, Market Data, Models, Predictions, Trading, Monitor
    components/
    hooks/
    lib/

data/                      CSV market data
models/                    Trained `.keras` models và scaler files
artifacts/                 Runtime artifacts được backend tạo nếu cần
logs/                      Application logs
docker-compose.yml         PostgreSQL + API container
```

## Các màn hình frontend

- `/`: Dashboard tổng quan hệ thống.
- `/market-data`: crawl MT5, import CSV, xem file dữ liệu.
- `/models`: train model, xem versions, activate model.
- `/models/$modelId`: xem chi tiết model.
- `/predictions`: chạy inference và xem lịch sử prediction.
- `/trading`: kết nối MT5, manual trade, auto trade, positions, trade history.
- `/monitor`: theo dõi WebSocket stream của trades và predictions.

## Backend API chính

### System

- `GET /api/v1/health`

### Market data

- `POST /api/v1/market-data/crawl`
- `POST /api/v1/market-data/import`
- `GET /api/v1/market-data/files`

### Models

- `POST /api/v1/models/train`
- `GET /api/v1/models`
- `GET /api/v1/models/{model_id}`
- `PATCH /api/v1/models/{model_id}/activate`

### Predictions

- `POST /api/v1/predictions/predict`
- `GET /api/v1/predictions`

### Trading / MT5

- `POST /api/v1/mt5/connect`
- `POST /api/v1/mt5/disconnect`
- `GET /api/v1/mt5/status`
- `GET /api/v1/mt5/positions`
- `POST /api/v1/trading/execute`
- `POST /api/v1/trading/auto/start`
- `POST /api/v1/trading/auto/stop`
- `GET /api/v1/trading/auto/status`
- `GET /api/v1/trades`
- `GET /api/v1/trades/{trade_id}`

### WebSocket

- `WS /api/v1/ws/trades`
- `WS /api/v1/ws/predictions`

## Dữ liệu được lưu ở đâu

- `candles`: dữ liệu nến crawl/import.
- `model_versions`: metadata và version model đã train.
- `predictions`: lịch sử suy luận.
- `trades`: lịch sử giao dịch.
- `models/`: file model và scaler được train ra.
- `data/`: file CSV đầu vào.
- `logs/`: log runtime.

## Rủi ro và disclaimer

Đây là dự án nghiên cứu, thử nghiệm và vận hành tín hiệu giao dịch. Nội dung repo này **không phải lời khuyên đầu tư tài chính**. Hãy kiểm thử kỹ trên tài khoản demo trước khi kết nối bot với tài khoản thật.
