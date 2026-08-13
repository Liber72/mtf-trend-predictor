"""Client to communicate with the FastAPI backend."""

import requests
from typing import Any

API_URL = "http://127.0.0.1:8000/api/v1"

class APIClient:
    
    @staticmethod
    def get_system_health() -> dict[str, Any]:
        resp = requests.get(f"{API_URL}/health")
        return resp.json() if resp.status_code == 200 else {}

    @staticmethod
    def crawl_data(symbol: str, start_date: str, end_date: str, timeframe: str) -> dict[str, Any]:
        resp = requests.post(f"{API_URL}/market-data/crawl", json={
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "timeframe": timeframe
        })
        return resp.json()

    @staticmethod
    def get_csv_files() -> list[str]:
        resp = requests.get(f"{API_URL}/market-data/files")
        if resp.status_code == 200:
            return [item["filename"] for item in resp.json()]
        return []

    @staticmethod
    def train_model(timeframe: str, data_file: str, lookback: int, epochs: int, batch_size: int, train_ratio: float) -> dict[str, Any]:
        resp = requests.post(f"{API_URL}/models/train", json={
            "timeframe": timeframe,
            "data_file": data_file,
            "lookback": lookback,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_ratio": train_ratio
        })
        return resp.json()

    @staticmethod
    def get_models() -> dict[str, Any]:
        resp = requests.get(f"{API_URL}/models")
        return resp.json()

    @staticmethod
    def activate_model(model_id: int) -> dict[str, Any]:
        resp = requests.patch(f"{API_URL}/models/{model_id}/activate")
        return resp.json()

    @staticmethod
    def mt5_connect() -> dict[str, Any]:
        resp = requests.post(f"{API_URL}/mt5/connect")
        return resp.json()

    @staticmethod
    def mt5_disconnect() -> dict[str, Any]:
        resp = requests.post(f"{API_URL}/mt5/disconnect")
        return resp.json()

    @staticmethod
    def mt5_status() -> dict[str, Any]:
        resp = requests.get(f"{API_URL}/mt5/status")
        return resp.json()

    @staticmethod
    def mt5_positions() -> list[dict[str, Any]]:
        resp = requests.get(f"{API_URL}/mt5/positions")
        if resp.status_code == 200:
            return resp.json()
        return []

    @staticmethod
    def execute_trade(signal: str, confidence: float) -> dict[str, Any]:
        resp = requests.post(f"{API_URL}/trading/execute", json={
            "signal": signal,
            "confidence": confidence
        })
        return resp.json()

    @staticmethod
    def start_auto_trade(interval: float, model_mode: str) -> dict[str, Any]:
        resp = requests.post(f"{API_URL}/trading/auto/start", json={
            "interval": interval,
            "model_mode": model_mode
        })
        return resp.json()

    @staticmethod
    def stop_auto_trade() -> dict[str, Any]:
        resp = requests.post(f"{API_URL}/trading/auto/stop")
        return resp.json()

    @staticmethod
    def get_auto_trade_status() -> dict[str, Any]:
        resp = requests.get(f"{API_URL}/trading/auto/status")
        return resp.json()

    @staticmethod
    def get_trades(limit: int = 20) -> list[dict[str, Any]]:
        resp = requests.get(f"{API_URL}/trades?size={limit}")
        if resp.status_code == 200:
            return resp.json().get("items", [])
        return []

    @staticmethod
    def predict_realtime(model_mode: str) -> dict[str, Any]:
        resp = requests.post(f"{API_URL}/predictions/predict", json={
            "model_mode": model_mode
        })
        return resp.json()

    @staticmethod
    def start_trailing(levels: list[dict] | None = None) -> dict[str, Any]:
        payload = {}
        if levels:
            payload["levels"] = levels
        resp = requests.post(f"{API_URL}/trading/trailing/start", json=payload if payload else None)
        return resp.json()

    @staticmethod
    def stop_trailing() -> dict[str, Any]:
        resp = requests.post(f"{API_URL}/trading/trailing/stop")
        return resp.json()

    @staticmethod
    def get_trailing_status() -> dict[str, Any]:
        resp = requests.get(f"{API_URL}/trading/trailing/status")
        return resp.json()
