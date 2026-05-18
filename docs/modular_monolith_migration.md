# Modular Monolith Migration

## Target shape

```text
src/
  main.py
  core/
  utils/
  infrastructure/
    db/
  modules/
    market_data/
    feature_engineering/
    ml/
    backtesting/
    trading/
    monitoring/
  apps/
    api/
```

## Why this shape

- Keeps one deployable app, which fits a local ML trading system.
- Separates feature boundaries so training, inference, backtesting, and execution can evolve independently.
- Avoids the current root-level script pileup.
- Makes FastAPI, SQLAlchemy 2 async, asyncpg, and PostgreSQL fit naturally.
- Lets model artifacts stay on disk while metadata and history live in PostgreSQL.

## What is already started

- `src/core/constants.py` now centralizes legacy trading and ML defaults.
- `src/core/settings.py` holds runtime config and filesystem paths.
- `src/infrastructure/db/` now contains async SQLAlchemy engine/session wiring.
- ORM models now exist for candles, model versions, predictions, trades, and backtest runs.
- `src/main.py` is the FastAPI bootstrap.
- `alembic/` is scaffolded for schema migrations.

## Next migration steps

1. Move `DataProcessor` into `src/modules/feature_engineering/`.
2. Move `LSTMModel` and `Trainer` into `src/modules/ml/`.
3. Move `Backtester` into `src/modules/backtesting/`.
4. Move `MT5Trader` into `src/modules/trading/`.
5. Replace CSV-only flows with repository/service calls to PostgreSQL.
6. Add API endpoints for training, backtesting, prediction, and trade history.
