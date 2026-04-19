"""
TCP Interface Module
Giao tiếp giữa Bot (Server) và Streamlit UI (Client) qua TCP
"""

import socket
import json
import threading
import time
from typing import Optional, Dict, List, Callable
from datetime import datetime

from config import TCP_HOST, TCP_PORT, TCP_BUFFER_SIZE


class BotServer:
    """
    TCP Server chạy trên tiến trình Bot.
    Lắng nghe lệnh từ UI (Client) và trả về trạng thái.
    Hỗ trợ nhiều Client kết nối đồng thời (multi-tab).
    """

    def __init__(self, host: str = TCP_HOST, port: int = TCP_PORT):
        self.host = host
        self.port = port
        self._server_socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Trạng thái bot (được cập nhật bởi bot.py)
        self.state: Dict = {
            "is_running": False,
            "auto_trade_enabled": False,
            "trailing_sl_enabled": False,
            "mt5_connected": False,
            "model_mode": "dual",
            "lot": 0.1,
            "sl_pips": 500,
            "tp_pips": 500,
            "max_positions": 3,
            "min_confidence": 0.5,
            "symbol": "XAUUSD",
            "auto_trade_interval": 0.5,
            "last_signal": None,
            "models_loaded": False,
        }

        # Log messages (circular buffer)
        self._log_messages: List[Dict] = []
        self._max_log = 200

        # Callback khi nhận lệnh từ UI
        self._command_handler: Optional[Callable] = None

    def set_command_handler(self, handler: Callable):
        """Đặt callback xử lý lệnh từ Client"""
        self._command_handler = handler

    def add_log(self, message: str, msg_type: str = "info"):
        """Thêm log message (thread-safe)"""
        with self._lock:
            self._log_messages.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": message,
                "type": msg_type,
            })
            if len(self._log_messages) > self._max_log:
                self._log_messages = self._log_messages[-self._max_log:]

    def get_logs(self, limit: int = 50) -> List[Dict]:
        """Lấy log messages"""
        with self._lock:
            return list(self._log_messages[-limit:])

    def update_state(self, key: str, value):
        """Cập nhật một trường state"""
        with self._lock:
            self.state[key] = value

    def get_state(self) -> Dict:
        """Lấy bản sao state hiện tại"""
        with self._lock:
            return dict(self.state)

    def start(self):
        """Khởi động TCP Server trong background thread"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print(f"🌐 TCP Server started on {self.host}:{self.port}")

    def stop(self):
        """Dừng TCP Server"""
        self._running = False
        # Đóng server socket để unblock accept()
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        print("🌐 TCP Server stopped")

    def _listen_loop(self):
        """Vòng lặp chính lắng nghe kết nối"""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)  # Timeout để kiểm tra _running

        try:
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(5)
        except OSError as e:
            print(f"❌ TCP Server bind error: {e}")
            self._running = False
            return

        while self._running:
            try:
                client_socket, addr = self._server_socket.accept()
                # Xử lý mỗi client trong thread riêng
                t = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, addr),
                    daemon=True,
                )
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break  # Socket đã đóng

        try:
            self._server_socket.close()
        except Exception:
            pass

    def _handle_client(self, client_socket: socket.socket, addr):
        """Xử lý một kết nối client"""
        try:
            client_socket.settimeout(5.0)
            data = client_socket.recv(TCP_BUFFER_SIZE)
            if not data:
                return

            request = json.loads(data.decode("utf-8"))
            response = self._process_request(request)

            response_bytes = json.dumps(response, default=str).encode("utf-8")
            client_socket.sendall(response_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            error_resp = json.dumps({"status": "error", "message": f"Invalid request: {e}"})
            try:
                client_socket.sendall(error_resp.encode("utf-8"))
            except Exception:
                pass
        except socket.timeout:
            pass
        except Exception as e:
            print(f"⚠️ Client handler error: {e}")
        finally:
            try:
                client_socket.close()
            except Exception:
                pass

    def _process_request(self, request: Dict) -> Dict:
        """Xử lý request từ Client và trả response"""
        action = request.get("action", "").upper()

        if action == "GET_STATUS":
            return {
                "status": "ok",
                "data": self.get_state(),
            }

        elif action == "GET_LOGS":
            limit = request.get("limit", 50)
            return {
                "status": "ok",
                "logs": self.get_logs(limit),
            }

        elif action == "UPDATE_CONFIG":
            # Cập nhật cấu hình từ UI
            config_data = request.get("data", {})
            with self._lock:
                for key, value in config_data.items():
                    if key in self.state:
                        self.state[key] = value
            # Gọi callback nếu có
            if self._command_handler:
                self._command_handler("UPDATE_CONFIG", config_data)
            return {"status": "ok", "message": "Config updated"}

        elif action == "START":
            if self._command_handler:
                self._command_handler("START", request.get("data", {}))
            return {"status": "ok", "message": "Start command sent"}

        elif action == "STOP":
            if self._command_handler:
                self._command_handler("STOP", request.get("data", {}))
            return {"status": "ok", "message": "Stop command sent"}

        elif action == "RELOAD_MODELS":
            if self._command_handler:
                self._command_handler("RELOAD_MODELS", {})
            return {"status": "ok", "message": "Reload command sent"}

        elif action == "PING":
            return {"status": "ok", "message": "pong"}

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class BotClient:
    """
    TCP Client chạy trên Streamlit UI.
    Gửi lệnh điều khiển và đọc trạng thái từ Bot Server.
    Mỗi lệnh sử dụng một kết nối ngắn (short-lived connection).
    """

    def __init__(self, host: str = TCP_HOST, port: int = TCP_PORT, timeout: float = 3.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def _send_request(self, request: Dict) -> Optional[Dict]:
        """Gửi request và nhận response"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
                data = json.dumps(request).encode("utf-8")
                sock.sendall(data)
                response = sock.recv(TCP_BUFFER_SIZE)
                return json.loads(response.decode("utf-8"))
        except (ConnectionRefusedError, ConnectionResetError):
            return None
        except socket.timeout:
            return None
        except Exception as e:
            print(f"⚠️ TCP Client error: {e}")
            return None

    # ── Convenience methods ──

    def ping(self) -> bool:
        """Kiểm tra Bot Server có đang chạy không"""
        resp = self._send_request({"action": "PING"})
        return resp is not None and resp.get("status") == "ok"

    def get_status(self) -> Optional[Dict]:
        """Lấy trạng thái bot"""
        resp = self._send_request({"action": "GET_STATUS"})
        if resp and resp.get("status") == "ok":
            return resp.get("data")
        return None

    def get_logs(self, limit: int = 50) -> List[Dict]:
        """Lấy log từ bot"""
        resp = self._send_request({"action": "GET_LOGS", "limit": limit})
        if resp and resp.get("status") == "ok":
            return resp.get("logs", [])
        return []

    def update_config(self, **kwargs) -> bool:
        """Cập nhật config cho bot"""
        resp = self._send_request({"action": "UPDATE_CONFIG", "data": kwargs})
        return resp is not None and resp.get("status") == "ok"

    def start_bot(self, **kwargs) -> bool:
        """Gửi lệnh START cho bot"""
        resp = self._send_request({"action": "START", "data": kwargs})
        return resp is not None and resp.get("status") == "ok"

    def stop_bot(self) -> bool:
        """Gửi lệnh STOP cho bot"""
        resp = self._send_request({"action": "STOP"})
        return resp is not None and resp.get("status") == "ok"

    def reload_models(self) -> bool:
        """Gửi lệnh reload models"""
        resp = self._send_request({"action": "RELOAD_MODELS"})
        return resp is not None and resp.get("status") == "ok"
