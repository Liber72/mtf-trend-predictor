"""
Bot Worker — Tiến trình giao dịch tự động chạy độc lập
Chạy bằng: python bot.py
"""

import os
import sys
import time
import signal
from datetime import datetime
from typing import Optional

# Thêm thư mục hiện tại vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Cấu hình GPU memory growth trước khi import TensorFlow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU memory growth enabled cho {len(gpus)} GPU(s)")

from config import (
    DEFAULT_SYMBOL, DEFAULT_LOT, DEFAULT_SL_PIPS, DEFAULT_TP_PIPS,
    MAX_POSITIONS, MIN_CONFIDENCE,
    AUTO_TRADE_INTERVAL, REALTIME_DATA_COUNT,
    MODEL_MODE_DUAL, MODEL_MODE_SINGLE_M5, DEFAULT_MODEL_MODE,
    TRAILING_CHECK_INTERVAL,
)
from mt5_trader import MT5Trader
from trainer import Trainer
from tcp_interface import BotServer


class TradingBot:
    """
    Bot giao dịch tự động chạy độc lập.
    Nhận lệnh điều khiển từ UI qua TCP Server.
    """

    def __init__(self):
        self.server = BotServer()
        self.trader: Optional[MT5Trader] = None
        self.trainer: Optional[Trainer] = None

        # Trạng thái nội bộ
        self.auto_trade_enabled = False
        self.trailing_sl_enabled = False
        self.model_mode = DEFAULT_MODEL_MODE
        self.auto_trade_interval = AUTO_TRADE_INTERVAL

        # Shutdown flag
        self._shutdown = False

        # Đăng ký command handler
        self.server.set_command_handler(self._handle_command)

    # ── Khởi tạo ──

    def init_mt5(self) -> bool:
        """Khởi tạo kết nối MT5"""
        state = self.server.get_state()
        self.trader = MT5Trader(
            symbol=state.get("symbol", DEFAULT_SYMBOL),
            lot=state.get("lot", DEFAULT_LOT),
            sl_pips=state.get("sl_pips", DEFAULT_SL_PIPS),
            tp_pips=state.get("tp_pips", DEFAULT_TP_PIPS),
            max_positions=state.get("max_positions", MAX_POSITIONS),
            min_confidence=state.get("min_confidence", MIN_CONFIDENCE),
        )
        success, msg = self.trader.connect()
        if success:
            self.server.update_state("mt5_connected", True)
            self.server.add_log(f"✅ {msg}", "success")
            print(f"✅ MT5: {msg}")
        else:
            self.server.update_state("mt5_connected", False)
            self.server.add_log(f"❌ {msg}", "error")
            print(f"❌ MT5: {msg}")
        return success

    def load_models(self) -> bool:
        """Load models LSTM"""
        self.trainer = Trainer(model_mode=self.model_mode)
        h1_loaded, m5_loaded = self.trainer.load_models()

        if self.model_mode == MODEL_MODE_SINGLE_M5:
            loaded = m5_loaded
        else:
            loaded = h1_loaded and m5_loaded

        self.server.update_state("models_loaded", loaded)

        if loaded:
            self.server.add_log("✅ Models loaded", "success")
            print("✅ Models loaded")
        else:
            msg = f"⚠️ Models: H1={'✓' if h1_loaded else '✗'}, M5={'✓' if m5_loaded else '✗'}"
            self.server.add_log(msg, "warning")
            print(msg)
        return loaded

    # ── Xử lý lệnh từ UI ──

    def _handle_command(self, action: str, data: dict):
        """Callback xử lý lệnh TCP từ các Client (UI tabs)"""
        if action == "START":
            self.auto_trade_enabled = True
            self.server.update_state("auto_trade_enabled", True)
            self.server.add_log("🤖 Auto Trading: BẬT", "success")
            print("🤖 Auto Trading: BẬT")

        elif action == "STOP":
            self.auto_trade_enabled = False
            self.server.update_state("auto_trade_enabled", False)
            self.server.add_log("⏹️ Auto Trading: TẮT", "info")
            print("⏹️ Auto Trading: TẮT")

        elif action == "UPDATE_CONFIG":
            self._apply_config(data)

        elif action == "RELOAD_MODELS":
            self.server.add_log("🔄 Đang reload models...", "info")
            print("🔄 Reloading models...")
            self.load_models()

    def _apply_config(self, data: dict):
        """Áp dụng config mới từ UI vào trader"""
        if self.trader:
            if "lot" in data:
                self.trader.lot = data["lot"]
            if "sl_pips" in data:
                self.trader.sl_pips = data["sl_pips"]
            if "tp_pips" in data:
                self.trader.tp_pips = data["tp_pips"]
            if "max_positions" in data:
                self.trader.max_positions = data["max_positions"]
            if "min_confidence" in data:
                self.trader.min_confidence = data["min_confidence"]
            if "symbol" in data:
                self.trader.symbol = data["symbol"]

        if "model_mode" in data:
            self.model_mode = data["model_mode"]
            if self.trader:
                self.trader.model_mode = data["model_mode"]

        if "auto_trade_interval" in data:
            self.auto_trade_interval = data["auto_trade_interval"]

        if "trailing_sl_enabled" in data:
            self.trailing_sl_enabled = data["trailing_sl_enabled"]
            self.server.update_state("trailing_sl_enabled", data["trailing_sl_enabled"])
            status = "BẬT" if data["trailing_sl_enabled"] else "TẮT"
            self.server.add_log(f"📐 Trailing SL: {status}", "info")
            print(f"📐 Trailing SL: {status}")

        print(f"⚙️ Config updated: {data}")

    # ── Vòng lặp chính ──

    def _predict_and_trade(self):
        """Thực hiện 1 chu kỳ dự đoán và giao dịch"""
        if not self.trader or not self.trainer or not self.trader.connected:
            return

        # Lấy dữ liệu realtime
        m5_df = self.trader.get_realtime_data("M5", REALTIME_DATA_COUNT)
        if m5_df is None:
            return

        h1_df = None
        if self.model_mode == MODEL_MODE_DUAL:
            h1_df = self.trader.get_realtime_data("H1", REALTIME_DATA_COUNT)
            if h1_df is None:
                return

        # Dự đoán
        results = self.trainer.predict(h1_df, m5_df, model_mode=self.model_mode)

        if "combined" not in results:
            return

        signal = results["combined"].get("signal")
        confidence = results["combined"].get("confidence", 0)

        # Chi tiết
        h1_dir = results.get("H1", {}).get("direction", "N/A")
        h1_prob = results.get("H1", {}).get("probability", 0)
        m5_dir = results.get("M5", {}).get("direction", "N/A")
        m5_prob = results.get("M5", {}).get("probability", 0)

        # Cập nhật last_signal vào server state
        self.server.update_state("last_signal", {
            "signal": signal,
            "confidence": confidence,
            "h1_dir": h1_dir, "h1_prob": h1_prob,
            "m5_dir": m5_dir, "m5_prob": m5_prob,
            "time": datetime.now().strftime("%H:%M:%S"),
            "model_mode": self.model_mode,
        })

        # Terminal log
        print(f"\n{'='*50}")
        print(f"📊 PREDICTION @ {datetime.now().strftime('%H:%M:%S')} [{self.model_mode}]")
        if self.model_mode == MODEL_MODE_DUAL:
            print(f"   H1: {h1_dir} ({h1_prob*100:.1f}%)")
        print(f"   M5: {m5_dir} ({m5_prob*100:.1f}%)")
        if confidence:
            print(f"   Combined: {signal} (Confidence: {confidence*100:.1f}%)")
        else:
            print(f"   Combined: {signal}")

        # Thực thi lệnh
        if signal and signal != "WAIT":
            executed, msg = self.trader.execute_signal(signal, confidence)
            if executed:
                print(f"   ✅ VÀO LỆNH: {msg}")
                self.server.add_log(f"✅ {msg}", "success")
            else:
                print(f"   ℹ️ BỎ QUA: {msg}")
                self.server.add_log(f"ℹ️ {msg}", "info")
        else:
            print(f"   ⏸️ WAIT - không vào lệnh")

        print(f"{'='*50}")

    def _trailing_check(self):
        """Thực hiện 1 chu kỳ trailing stop loss"""
        if not self.trader or not self.trader.connected:
            return

        results = self.trader.trailing_stop_loss()
        for r in results:
            if r.get("modified"):
                msg = (f"📐 Trailing SL: #{r['ticket']} ({r['type']}) "
                       f"lời {r['profit_pips']} pips → SL={r['new_sl']:.2f} "
                       f"(level {r['level'][0]}→{r['level'][1]})")
                print(msg)
                self.server.add_log(msg, "info")

    def run(self):
        """Vòng lặp chính của bot"""
        print("=" * 60)
        print("🤖 LSTM Trading Bot — Starting...")
        print("=" * 60)

        # 1. Khởi động TCP Server
        self.server.start()
        self.server.update_state("is_running", True)

        # 2. Kết nối MT5
        if not self.init_mt5():
            print("⚠️ MT5 chưa kết nối. Bot sẽ chờ lệnh START từ UI.")

        # 3. Load models
        self.load_models()

        # 4. Vòng lặp chính
        print("\n🤖 Bot sẵn sàng. Chờ lệnh từ UI...\n")
        last_trade_time = 0
        last_trailing_time = 0

        try:
            while not self._shutdown:
                now = time.time()

                # Đồng bộ config vào trader nếu cần
                state = self.server.get_state()
                self.auto_trade_enabled = state.get("auto_trade_enabled", False)
                self.trailing_sl_enabled = state.get("trailing_sl_enabled", False)

                # Auto Trading
                if self.auto_trade_enabled and (now - last_trade_time) >= self.auto_trade_interval:
                    try:
                        self._predict_and_trade()
                    except Exception as e:
                        print(f"⚠️ Auto Trade error: {e}")
                        self.server.add_log(f"⚠️ Error: {e}", "error")
                    last_trade_time = now

                # Trailing SL
                if self.trailing_sl_enabled and (now - last_trailing_time) >= TRAILING_CHECK_INTERVAL:
                    try:
                        self._trailing_check()
                    except Exception as e:
                        print(f"⚠️ Trailing SL error: {e}")
                    last_trailing_time = now

                time.sleep(0.1)  # Giảm CPU usage

        except KeyboardInterrupt:
            print("\n⏹️ Bot dừng bởi Ctrl+C")

        # Cleanup
        self.server.update_state("is_running", False)
        self.server.stop()
        if self.trader and self.trader.connected:
            self.trader.disconnect()
            print("🔌 MT5 disconnected")
        print("🤖 Bot đã dừng.")


def main():
    bot = TradingBot()

    # Xử lý tín hiệu tắt (Ctrl+C, kill)
    def signal_handler(sig, frame):
        bot._shutdown = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bot.run()


if __name__ == "__main__":
    main()
