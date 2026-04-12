"""
Streamlit UI Application
Giao diện người dùng cho hệ thống dự đoán LSTM Dual-Timeframe với Auto Trading
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import sys
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_SYMBOL, DEFAULT_LOT, DEFAULT_SL_PIPS, DEFAULT_TP_PIPS,
    MAX_POSITIONS, MIN_CONFIDENCE, DEFAULT_TRAILING_SL_LEVELS,
    TRADE_LOG_MAX_MESSAGES, REALTIME_DATA_COUNT,
    MAX_TRADE_LOG_DISPLAY, AUTO_TRADE_INTERVAL, UI_REFRESH_INTERVAL,
    PREDICTION_THRESHOLD,
    MODEL_MODE_DUAL, MODEL_MODE_SINGLE_M5, DEFAULT_MODEL_MODE,
)
from data_processor import DataProcessor
from lstm_model import LSTMModel
from trainer import Trainer
from mt5_trader import MT5Trader, get_trader
from backtester import Backtester, BacktestResult
from crawldata_MT5 import download_xauusd_data, TIMEFRAME_MAP

# Page config
st.set_page_config(
    page_title="LSTM Dual-Timeframe Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        color: #00d4ff;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .buy-signal {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
    }
    .sell-signal {
        background: linear-gradient(135deg, #d63031 0%, #e17055 100%);
        color: white;
    }
    .wait-signal {
        background: linear-gradient(135deg, #636e72 0%, #b2bec3 100%);
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #00d4ff;
    }
    .connected-status {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        padding: 0.5rem 1rem;
        border-radius: 5px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .disconnected-status {
        background: linear-gradient(135deg, #636e72 0%, #b2bec3 100%);
        padding: 0.5rem 1rem;
        border-radius: 5px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .position-card {
        background: #f0f0f0;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #00b894;
    }
    .position-card.sell {
        border-left-color: #d63031;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Khởi tạo session state"""
    if 'trainer' not in st.session_state:
        st.session_state.trainer = Trainer()
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'h1_metrics' not in st.session_state:
        st.session_state.h1_metrics = None
    if 'm5_metrics' not in st.session_state:
        st.session_state.m5_metrics = None
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    # MT5 Trading states
    if 'mt5_connected' not in st.session_state:
        st.session_state.mt5_connected = False
    if 'auto_trading' not in st.session_state:
        st.session_state.auto_trading = False
    if 'trader' not in st.session_state:
        st.session_state.trader = None
    if 'last_signal' not in st.session_state:
        st.session_state.last_signal = None
    if 'trade_messages' not in st.session_state:
        st.session_state.trade_messages = []
    if 'model_mode' not in st.session_state:
        st.session_state.model_mode = DEFAULT_MODEL_MODE


def load_models():
    """Load các mô hình đã train"""
    trainer = st.session_state.trainer
    h1_loaded, m5_loaded = trainer.load_models()
    # models_loaded phụ thuộc vào mode
    if st.session_state.model_mode == MODEL_MODE_SINGLE_M5:
        st.session_state.models_loaded = m5_loaded
    else:
        st.session_state.models_loaded = h1_loaded and m5_loaded
    return h1_loaded, m5_loaded


def get_csv_files():
    """Lấy danh sách file CSV trong thư mục"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    return {f: os.path.join(base_dir, f) for f in csv_files}


def display_prediction_box(direction: str, probability: float, label: str):
    """Hiển thị box dự đoán"""
    if direction == "UP":
        color = "#00b894"
        icon = "📈"
    else:
        color = "#d63031"
        icon = "📉"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}99 0%, {color} 100%); 
                padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
        <h3 style="color: white; margin: 0;">{label}</h3>
        <h2 style="color: white; margin: 0.5rem 0;">{icon} {direction}</h2>
        <p style="color: white; margin: 0;">Confidence: {probability*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)


def display_combined_signal(signal: str, confidence: float = None, reason: str = None):
    """Hiển thị tín hiệu tổng hợp"""
    if signal == "BUY":
        css_class = "buy-signal"
        icon = "🚀 BUY"
    elif signal == "SELL":
        css_class = "sell-signal"
        icon = "🔻 SELL"
    else:
        css_class = "wait-signal"
        icon = "⏸️ WAIT"
    
    content = f"<h2 style='margin: 0;'>{icon}</h2>"
    if confidence:
        content += f"<p style='margin: 0.5rem 0 0 0;'>Confidence: {confidence*100:.1f}%</p>"
    if reason:
        content += f"<p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>{reason}</p>"
    
    st.markdown(f"""
    <div class="prediction-box {css_class}">
        {content}
    </div>
    """, unsafe_allow_html=True)


def add_trade_message(msg: str, msg_type: str = "info"):
    """Thêm message vào log"""
    st.session_state.trade_messages.append({
        'time': datetime.now().strftime("%H:%M:%S"),
        'message': msg,
        'type': msg_type
    })
    # Giữ tối đa 50 messages
    if len(st.session_state.trade_messages) > TRADE_LOG_MAX_MESSAGES:
        st.session_state.trade_messages = st.session_state.trade_messages[-TRADE_LOG_MAX_MESSAGES:]


def display_positions(positions):
    """Hiển thị danh sách vị thế"""
    if not positions:
        st.info("Không có lệnh đang mở")
        return
    
    for pos in positions:
        pos_type = pos['type']
        profit = pos['profit']
        profit_color = "#00b894" if profit >= 0 else "#d63031"
        card_class = "sell" if pos_type == "SELL" else ""
        
        st.markdown(f"""
        <div class="position-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>#{pos['ticket']}</strong> - {pos_type}
                    <br><small>Mở: {pos['price_open']:.2f} | Hiện tại: {pos['price_current']:.2f}</small>
                    <br><small>Lot: {pos['volume']} | SL: {pos['sl']:.2f} | TP: {pos['tp']:.2f}</small>
                </div>
                <div style="text-align: right;">
                    <span style="color: {profit_color}; font-size: 1.2rem; font-weight: bold;">
                        {profit:+.2f}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def realtime_predict_and_trade(trader: MT5Trader, auto_execute: bool = False):
    """Dự đoán realtime và thực hiện giao dịch"""
    results = {}
    model_mode = st.session_state.model_mode
    
    # Lấy dữ liệu realtime
    m5_df = trader.get_realtime_data("M5", REALTIME_DATA_COUNT)
    
    if model_mode == MODEL_MODE_DUAL:
        h1_df = trader.get_realtime_data("H1", REALTIME_DATA_COUNT)
    else:
        h1_df = None  # Single M5: không cần H1
    
    if m5_df is None:
        return None, "Không lấy được dữ liệu M5 từ MT5"
    
    if model_mode == MODEL_MODE_DUAL and h1_df is None:
        return None, "Không lấy được dữ liệu H1 từ MT5"
    
    # Dự đoán
    results = st.session_state.trainer.predict(h1_df, m5_df, model_mode=model_mode)
    
    # Thực hiện giao dịch nếu auto_execute
    if auto_execute and 'combined' in results:
        signal = results['combined'].get('signal')
        confidence = results['combined'].get('confidence', 0)
        
        # Lấy chi tiết từng prediction
        h1_dir = results.get('H1', {}).get('direction', 'N/A')
        h1_prob = results.get('H1', {}).get('probability', 0)
        m5_dir = results.get('M5', {}).get('direction', 'N/A')
        m5_prob = results.get('M5', {}).get('probability', 0)
        
        # In ra terminal chi tiết
        print(f"\n{'='*50}")
        print(f"📊 PREDICTION @ {datetime.now().strftime('%H:%M:%S')} [{model_mode}]")
        if model_mode == MODEL_MODE_DUAL:
            print(f"   H1: {h1_dir} ({h1_prob*100:.1f}%)")
        print(f"   M5: {m5_dir} ({m5_prob*100:.1f}%)")
        print(f"   Combined: {signal} (Confidence: {confidence*100:.1f}%)" if confidence else f"   Combined: {signal}")
        
        if signal and signal != "WAIT":
            executed, msg = trader.execute_signal(signal, confidence)
            results['trade_executed'] = executed
            results['trade_message'] = msg
            
            if executed:
                print(f"   ✅ VÀO LỆNH: {msg}")
                add_trade_message(f"✅ {msg}", "success")
            else:
                print(f"   ℹ️ BỎ QUA: {msg}")
                add_trade_message(f"ℹ️ {msg}", "info")
        else:
            print(f"   ⏸️ WAIT - không vào lệnh")
        print(f"{'='*50}")
    
    return results, None


def main():
    # Initialize
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">📊 LSTM Dual-Timeframe Predictor + Auto Trading</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Symbol input
        symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
        
        st.divider()
        
        # ========== MODEL MODE ==========
        st.subheader("🎯 Model Mode")
        
        model_mode = st.radio(
            "Chế độ dự đoán",
            options=[MODEL_MODE_DUAL, MODEL_MODE_SINGLE_M5],
            format_func=lambda x: "Dual (M5 + H1) — Đồng thuận" if x == MODEL_MODE_DUAL else "Single (M5 Only)",
            index=0 if st.session_state.model_mode == MODEL_MODE_DUAL else 1,
            help="Dual: cần cả 2 model cùng hướng mới vào lệnh. Single: chỉ dùng M5."
        )
        st.session_state.model_mode = model_mode
        
        # Cập nhật models_loaded theo mode mới
        trainer = st.session_state.trainer
        if model_mode == MODEL_MODE_SINGLE_M5:
            st.session_state.models_loaded = trainer.m5_model is not None
        else:
            st.session_state.models_loaded = (trainer.h1_model is not None and trainer.m5_model is not None)
        
        # Cập nhật trainer mode
        trainer.model_mode = model_mode

        st.divider()
        
        # ========== MT5 CONNECTION ==========
        st.subheader("🔌 MT5 Connection")
        
        col_con1, col_con2 = st.columns(2)
        with col_con1:
            connect_btn = st.button("Kết nối", width='stretch', disabled=st.session_state.mt5_connected)
        with col_con2:
            disconnect_btn = st.button("Ngắt", width='stretch', disabled=not st.session_state.mt5_connected)
        
        # Connection status
        if st.session_state.mt5_connected:
            st.markdown('<div class="connected-status">🟢 Đã kết nối MT5</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="disconnected-status">🔴 Chưa kết nối</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # ========== TRADING CONFIG ==========
        st.subheader("💰 Trading Config")
        
        lot_size = st.number_input("Lot Size", min_value=0.01, max_value=10.0, value=DEFAULT_LOT, step=0.01)
        sl_pips = st.number_input("Stop Loss (pips)", min_value=10, max_value=1000, value=DEFAULT_SL_PIPS, step=10)
        tp_pips = st.number_input("Take Profit (pips)", min_value=10, max_value=1000, value=DEFAULT_TP_PIPS, step=10)
        max_positions = st.number_input("Max Positions", min_value=1, max_value=20, value=MAX_POSITIONS, step=1)
        min_confidence = st.slider("Min Confidence", 0.5, 0.95, MIN_CONFIDENCE, 0.05)
        
        st.divider()
        
        # ========== TRAILING STOP LOSS ==========
        st.subheader("📐 Trailing Stop Loss")
        
        enable_trailing_sl = st.toggle(
            "Bật Trailing SL",
            value=st.session_state.get('trailing_sl_enabled', False),
            disabled=not st.session_state.mt5_connected,
            help="Tự động kéo SL theo lợi nhuận, kiểm tra mỗi 1 giây"
        )
        st.session_state.trailing_sl_enabled = enable_trailing_sl
        
        # Hiển thị bảng trailing levels
        if enable_trailing_sl:
            st.caption("Bảng trailing levels (pips):")
            trailing_data = pd.DataFrame({
                'Khi lời (pips)': [level[0] for level in DEFAULT_TRAILING_SL_LEVELS],
                'Kéo SL về (pips)': [level[1] for level in DEFAULT_TRAILING_SL_LEVELS]
            })
            st.dataframe(trailing_data, width='stretch', hide_index=True)
            
            # Trailing status indicator
            if st.session_state.mt5_connected and st.session_state.trader:
                if st.session_state.trader._trailing_thread_running:
                    st.markdown('✅ Trailing thread đang chạy (1s interval)')
                else:
                    st.markdown('⏸️ Trailing thread chưa bắt đầu')
        
        st.divider()
        
        # ========== AUTO TRADING ==========
        st.subheader("🤖 Auto Trading")
        
        auto_trading = st.toggle(
            "Bật Auto Trading",
            value=st.session_state.auto_trading,
            disabled=not st.session_state.mt5_connected or not st.session_state.models_loaded
        )
        st.session_state.auto_trading = auto_trading
        
        trade_interval = st.number_input(
            "Bot Interval (giây)", min_value=0.1, max_value=10.0,
            value=AUTO_TRADE_INTERVAL, step=0.1,
            help="Khoảng thời gian giữa mỗi lần bot kiểm tra & vào lệnh"
        )
        ui_refresh = st.number_input(
            "UI Refresh (giây)", min_value=1, max_value=60,
            value=UI_REFRESH_INTERVAL, step=1,
            help="Khoảng thời gian giao diện tự cập nhật hiển thị"
        )
        
        # Auto Trading thread status
        if st.session_state.mt5_connected and st.session_state.trader:
            if st.session_state.trader._auto_trade_thread_running:
                st.markdown(f'✅ Bot đang chạy ({st.session_state.trader._auto_trade_interval}s interval)')
            elif st.session_state.auto_trading:
                st.markdown('⏳ Đang khởi động...')
            else:
                st.markdown('⏸️ Bot chưa bật')
        
        if not st.session_state.models_loaded:
            st.warning("⚠️ Cần load models trước")
        
        st.divider()
        
        # ========== TRAINING CONFIG ==========
        st.subheader("📚 Huấn luyện")
        
        lookback = st.slider("Lookback", 12, 96, 48, help="Số nến nhìn lại")
        epochs = st.slider("Epochs", 10, 200, 100)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        train_ratio = st.slider("Train Ratio", 0.8, 1.0, 0.8, 0.05, help="0.8 = Val 20% đầu, Train 80% sau")
        
        # File selection
        csv_files = get_csv_files()
        
        st.markdown("**File dữ liệu H1:**")
        h1_files = {k: v for k, v in csv_files.items() if '_H1_' in k}
        if h1_files:
            h1_file = st.selectbox("H1 Data", list(h1_files.keys()), label_visibility="collapsed")
            h1_path = h1_files[h1_file]
        else:
            st.warning("Chưa có file H1")
            h1_path = None
        
        st.markdown("**File dữ liệu M5:**")
        m5_files = {k: v for k, v in csv_files.items() if '_M5_' in k}
        if m5_files:
            m5_file = st.selectbox("M5 Data", list(m5_files.keys()), label_visibility="collapsed")
            m5_path = m5_files[m5_file]
        else:
            st.warning("Chưa có file M5")
            m5_path = None
        
        st.divider()
        
        # Train buttons
        col1, col2 = st.columns(2)
        with col1:
            train_h1 = st.button("🔄 Train H1", width='stretch', disabled=h1_path is None)
        with col2:
            train_m5 = st.button("🔄 Train M5", width='stretch', disabled=m5_path is None)
        
        train_both = st.button(
            "🚀 Train Cả 2",
            type="primary",
            width='stretch',
            disabled=(h1_path is None or m5_path is None)
        )
        
        st.divider()
        
        # Load models button
        load_btn = st.button("📂 Load Models", width='stretch')
        
        st.divider()
        
        # ========== CRAWL DATA SECTION ==========
        st.subheader("📥 Crawl Data từ MT5")
        
        # Symbol input for crawl
        crawl_symbol = st.text_input("Symbol (Crawl)", value=DEFAULT_SYMBOL, key="crawl_symbol")
        
        # Timeframe selection - multiselect
        available_timeframes = ["M5", "H1", "M15", "M30", "H4", "D1"]
        selected_timeframes = st.multiselect(
            "Timeframe(s)",
            options=available_timeframes,
            default=["M5", "H1"],
            key="crawl_timeframes"
        )
        
        # Date range
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            crawl_start_date = st.date_input(
                "Từ ngày",
                value=datetime(2024, 1, 1),
                key="crawl_start"
            )
        with col_date2:
            crawl_end_date = st.date_input(
                "Đến ngày",
                value=datetime.now(),
                key="crawl_end"
            )
        
        # Validate dates
        date_valid = crawl_start_date < crawl_end_date
        if not date_valid:
            st.error("Ngày bắt đầu phải < ngày kết thúc")
        
        # Crawl button
        crawl_btn = st.button(
            "🚀 Bắt đầu Crawl",
            type="primary",
            width='stretch',
            disabled=(len(selected_timeframes) == 0 or not date_valid)
        )
        
        # Hiển thị danh sách files đã crawl
        st.divider()
        st.markdown("**📂 Files dữ liệu:**")
        for filename in sorted(csv_files.keys()):
            file_path = csv_files[filename]
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            st.caption(f"• {filename} ({file_size:.1f} MB)")
    
    # ========== MT5 CONNECTION HANDLERS ==========
    if connect_btn:
        trader = get_trader(
            symbol=symbol,
            lot=lot_size,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            max_positions=max_positions,
            min_confidence=min_confidence
        )
        success, msg = trader.connect()
        if success:
            st.session_state.trader = trader
            st.session_state.mt5_connected = True
            st.success(f"✅ {msg}")
            add_trade_message(f"Kết nối MT5: {msg}", "success")
        else:
            st.error(f"❌ {msg}")
            add_trade_message(f"Lỗi kết nối: {msg}", "error")
        st.rerun()
    
    if disconnect_btn:
        if st.session_state.trader:
            # Dừng auto trade thread trước khi disconnect
            if st.session_state.trader._auto_trade_thread_running:
                st.session_state.trader.stop_auto_trade_thread()
            # Dừng trailing thread trước khi disconnect
            if st.session_state.trader._trailing_thread_running:
                st.session_state.trader.stop_trailing_thread()
            st.session_state.trader.disconnect()
        st.session_state.mt5_connected = False
        st.session_state.auto_trading = False
        st.session_state.trailing_sl_enabled = False
        add_trade_message("Đã ngắt kết nối MT5", "info")
        st.rerun()
    
    # ========== CRAWL DATA HANDLER ==========
    if crawl_btn and selected_timeframes:
        crawl_results = []
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        for idx, tf in enumerate(selected_timeframes):
            status_text.text(f"Đang tải {crawl_symbol} {tf}...")
            
            try:
                # download_xauusd_data tự quản lý MT5 connection
                df = download_xauusd_data(
                    symbol=crawl_symbol,
                    start_date=crawl_start_date.strftime("%Y-%m-%d"),
                    end_date=crawl_end_date.strftime("%Y-%m-%d"),
                    timeframe=tf
                )
                
                if df is not None:
                    crawl_results.append({
                        'timeframe': tf,
                        'rows': len(df),
                        'status': 'success'
                    })
                else:
                    crawl_results.append({
                        'timeframe': tf,
                        'rows': 0,
                        'status': 'failed'
                    })
            except Exception as e:
                crawl_results.append({
                    'timeframe': tf,
                    'rows': 0,
                    'status': f'error: {str(e)}'
                })
            
            progress_bar.progress((idx + 1) / len(selected_timeframes))
        
        status_text.text("✅ Hoàn thành!")
        
        # Hiển thị kết quả
        for result in crawl_results:
            if result['status'] == 'success':
                st.sidebar.success(f"✓ {crawl_symbol}_{result['timeframe']}: {result['rows']:,} nến")
            else:
                st.sidebar.error(f"✗ {crawl_symbol}_{result['timeframe']}: {result['status']}")
        
        # Refresh file list
        time.sleep(1)
        st.rerun()
    
    # ========== SYNC TRADING CONFIG REALTIME ==========
    if st.session_state.mt5_connected and st.session_state.trader:
        trader = st.session_state.trader
        # Update config values in realtime
        trader.symbol = symbol
        trader.lot = lot_size
        trader.sl_pips = sl_pips
        trader.tp_pips = tp_pips
        trader.max_positions = max_positions
        trader.min_confidence = min_confidence
        # Sync model mode
        trader.model_mode = model_mode
        
        # ========== TRAILING SL THREAD CONTROL ==========
        if enable_trailing_sl and not trader._trailing_thread_running:
            trader.start_trailing_thread()
            add_trade_message("📐 Bật Trailing SL (kiểm tra mỗi 1s)", "info")
        elif not enable_trailing_sl and trader._trailing_thread_running:
            trader.stop_trailing_thread()
            add_trade_message("⏹️ Tắt Trailing SL", "info")
        
        # ========== AUTO TRADE THREAD CONTROL ==========
        if auto_trading and not trader._auto_trade_thread_running:
            # Bật auto-trade thread
            trader.start_auto_trade_thread(
                trainer=st.session_state.trainer,
                interval=trade_interval,
                model_mode=model_mode
            )
            mode_label = "Dual (M5+H1)" if model_mode == MODEL_MODE_DUAL else "Single (M5)"
            add_trade_message(f"🤖 Bật Auto Trading (interval: {trade_interval}s, mode: {mode_label})", "success")
        elif not auto_trading and trader._auto_trade_thread_running:
            # Tắt auto-trade thread
            trader.stop_auto_trade_thread()
            add_trade_message("⏹️ Tắt Auto Trading", "info")
    
    # ========== TRAINING HANDLERS ==========
    if train_h1 and h1_path:
        with st.spinner("Đang huấn luyện mô hình H1..."):
            try:
                _, metrics = st.session_state.trainer.train_model(
                    "H1", h1_path, lookback, epochs, batch_size, train_ratio
                )
                st.session_state.h1_metrics = metrics
                st.success("✅ Hoàn thành huấn luyện H1!")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
    
    if train_m5 and m5_path:
        with st.spinner("Đang huấn luyện mô hình M5..."):
            try:
                _, metrics = st.session_state.trainer.train_model(
                    "M5", m5_path, lookback, epochs, batch_size, train_ratio
                )
                st.session_state.m5_metrics = metrics
                st.success("✅ Hoàn thành huấn luyện M5!")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
    
    if train_both and h1_path and m5_path:
        progress = st.progress(0)
        status = st.empty()
        
        status.text("Đang huấn luyện mô hình H1...")
        try:
            _, h1_metrics = st.session_state.trainer.train_model(
                "H1", h1_path, lookback, epochs, batch_size, train_ratio
            )
            st.session_state.h1_metrics = h1_metrics
            progress.progress(50)
        except Exception as e:
            st.error(f"❌ Lỗi H1: {e}")
        
        status.text("Đang huấn luyện mô hình M5...")
        try:
            _, m5_metrics = st.session_state.trainer.train_model(
                "M5", m5_path, lookback, epochs, batch_size, train_ratio
            )
            st.session_state.m5_metrics = m5_metrics
            progress.progress(100)
        except Exception as e:
            st.error(f"❌ Lỗi M5: {e}")
        
        status.text("✅ Hoàn thành!")
        st.session_state.training_complete = True
    
    if load_btn:
        h1_loaded, m5_loaded = load_models()
        if h1_loaded:
            st.success("✅ Loaded H1 model")
        else:
            st.warning("⚠️ Chưa có H1 model")
        if m5_loaded:
            st.success("✅ Loaded M5 model")
        else:
            st.warning("⚠️ Chưa có M5 model")
    
    # ========== MAIN CONTENT ==========
    
    # Row 1: Account info & Prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Mô hình H1")
        if model_mode == MODEL_MODE_SINGLE_M5:
            st.info("⚠️ Chế độ Single M5 — không sử dụng H1")
        elif st.session_state.h1_metrics:
            metrics = st.session_state.h1_metrics
            st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            st.metric("Precision", f"{metrics['precision']*100:.2f}%")
            st.metric("Recall", f"{metrics['recall']*100:.2f}%")
            st.metric("F1 Score", f"{metrics['f1_score']*100:.2f}%")
        elif st.session_state.trainer.h1_model:
            st.info("Model đã load, chưa có metrics")
        else:
            st.info("Chưa có model H1")
    
    with col2:
        st.subheader("📊 Mô hình M5")
        if st.session_state.m5_metrics:
            metrics = st.session_state.m5_metrics
            st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            st.metric("Precision", f"{metrics['precision']*100:.2f}%")
            st.metric("Recall", f"{metrics['recall']*100:.2f}%")
            st.metric("F1 Score", f"{metrics['f1_score']*100:.2f}%")
        elif st.session_state.trainer.m5_model:
            st.info("Model đã load, chưa có metrics")
        else:
            st.info("Chưa có model M5")
    
    with col3:
        st.subheader("🎯 Tín hiệu & Trading")
        
        # Realtime prediction button
        if st.session_state.mt5_connected and st.session_state.models_loaded:
            predict_btn = st.button("🔮 Dự đoán Realtime", type="primary", width='stretch')
            
            if predict_btn:
                with st.spinner("Đang dự đoán..."):
                    results, error = realtime_predict_and_trade(
                        st.session_state.trader,
                        auto_execute=False
                    )
                    
                    if error:
                        st.error(error)
                    elif results:
                        if model_mode == MODEL_MODE_DUAL and 'H1' in results and 'direction' in results['H1']:
                            display_prediction_box(
                                results['H1']['direction'],
                                results['H1']['probability'],
                                "Mô hình H1"
                            )
                        
                        if 'M5' in results and 'direction' in results['M5']:
                            display_prediction_box(
                                results['M5']['direction'],
                                results['M5']['probability'],
                                "Mô hình M5"
                            )
                        
                        if 'combined' in results:
                            st.markdown("**Kết quả:**")
                            display_combined_signal(
                                results['combined']['signal'],
                                results['combined'].get('confidence'),
                                results['combined'].get('reason')
                            )
                            st.session_state.last_signal = results['combined']
        
        elif st.session_state.models_loaded:
            # Fallback to CSV prediction
            has_required_files = m5_path is not None
            if model_mode == MODEL_MODE_DUAL:
                has_required_files = has_required_files and h1_path is not None
            
            if has_required_files:
                if st.button("🔮 Dự đoán (CSV)", type="primary", width='stretch'):
                    try:
                        m5_df = pd.read_csv(m5_path)
                        h1_df = pd.read_csv(h1_path) if model_mode == MODEL_MODE_DUAL else None
                        results = st.session_state.trainer.predict(h1_df, m5_df, model_mode=model_mode)
                        
                        if model_mode == MODEL_MODE_DUAL and 'H1' in results and 'direction' in results['H1']:
                            display_prediction_box(
                                results['H1']['direction'],
                                results['H1']['probability'],
                                "Mô hình H1"
                            )
                        
                        if 'M5' in results and 'direction' in results['M5']:
                            display_prediction_box(
                                results['M5']['direction'],
                                results['M5']['probability'],
                                "Mô hình M5"
                            )
                        
                        if 'combined' in results:
                            st.markdown("**Kết quả:**")
                            display_combined_signal(
                                results['combined']['signal'],
                                results['combined'].get('confidence'),
                                results['combined'].get('reason')
                            )
                    except Exception as e:
                        st.error(f"❌ Lỗi dự đoán: {e}")
        else:
            if model_mode == MODEL_MODE_SINGLE_M5:
                st.info("Cần train hoặc load model M5")
            else:
                st.info("Cần train hoặc load cả 2 model")
    
    st.divider()
    
    # Row 2: Trading info
    if st.session_state.mt5_connected and st.session_state.trader:
        trader = st.session_state.trader
        
        col_acc, col_pos, col_log = st.columns([1, 1.5, 1.5])
        
        with col_acc:
            st.subheader("💳 Tài khoản")
            account = trader.get_account_info()
            if account:
                st.metric("Balance", f"${account['balance']:,.2f}")
                st.metric("Equity", f"${account['equity']:,.2f}")
                st.metric("Profit", f"${account['profit']:+,.2f}")
                st.metric("Margin Free", f"${account['margin_free']:,.2f}")
                st.caption(f"Login: {account['login']} | Server: {account['server']}")
            
            # Symbol info
            sym_info = trader.get_symbol_info()
            if sym_info:
                st.divider()
                st.metric("Bid", f"{sym_info['bid']:.2f}")
                st.metric("Ask", f"{sym_info['ask']:.2f}")
                st.caption(f"Spread: {sym_info['spread']} | Digits: {sym_info['digits']}")
        
        with col_pos:
            st.subheader("📋 Vị thế đang mở")
            
            positions = trader.get_open_positions()
            display_positions(positions)
            
            if positions:
                col_close1, col_close2 = st.columns(2)
                with col_close1:
                    if st.button("❌ Đóng tất cả", width='stretch', type="secondary"):
                        success, fail = trader.close_all_positions()
                        if success > 0:
                            st.success(f"Đóng {success} lệnh thành công")
                            add_trade_message(f"Đóng {success} lệnh thành công", "success")
                        if fail > 0:
                            st.warning(f"Không đóng được {fail} lệnh")
                        st.rerun()
            
            # Manual trade buttons
            st.divider()
            st.markdown("**Vào lệnh thủ công:**")
            col_buy, col_sell = st.columns(2)
            with col_buy:
                if st.button("🚀 BUY", width='stretch', type="primary"):
                    success, msg, ticket = trader.open_position("BUY")
                    if success:
                        st.success(msg)
                        add_trade_message(msg, "success")
                    else:
                        st.error(msg)
                        add_trade_message(msg, "error")
                    st.rerun()
            with col_sell:
                if st.button("🔻 SELL", width='stretch', type="primary"):
                    success, msg, ticket = trader.open_position("SELL")
                    if success:
                        st.success(msg)
                        add_trade_message(msg, "success")
                    else:
                        st.error(msg)
                        add_trade_message(msg, "error")
                    st.rerun()
        
        with col_log:
            st.subheader("📜 Trading Log")
            
            if st.session_state.trade_messages:
                for msg in reversed(st.session_state.trade_messages[-MAX_TRADE_LOG_DISPLAY:]):
                    icon = "✅" if msg['type'] == "success" else "ℹ️" if msg['type'] == "info" else "❌"
                    st.caption(f"[{msg['time']}] {icon} {msg['message']}")
            else:
                st.caption("Chưa có hoạt động nào")
    
    # Auto trading status display (thread chạy ngầm, UI chỉ hiển thị)
    if st.session_state.auto_trading and st.session_state.mt5_connected and st.session_state.trader:
        st.divider()
        
        auto_col1, auto_col2 = st.columns([3, 1])
        with auto_col1:
            st.markdown("### 🤖 Auto Trading đang chạy...")
            trader = st.session_state.trader
            mode_label = "Dual (M5+H1)" if trader.model_mode == MODEL_MODE_DUAL else "Single (M5)"
            st.caption(f"Mode: {mode_label} | Bot interval: {trader._auto_trade_interval if hasattr(trader, '_auto_trade_interval') else AUTO_TRADE_INTERVAL}s | UI refresh: {ui_refresh}s")
        
        with auto_col2:
            if st.button("⏹️ Dừng", type="secondary", width='stretch'):
                if st.session_state.trader._auto_trade_thread_running:
                    st.session_state.trader.stop_auto_trade_thread()
                st.session_state.auto_trading = False
                st.rerun()
        
        # Hiển thị tín hiệu mới nhất từ background thread
        last_signal = st.session_state.trader.get_last_signal()
        if last_signal:
            signal = last_signal.get('signal', 'WAIT')
            confidence = last_signal.get('confidence', 0)
            sig_time = last_signal.get('time', '')
            
            if signal == 'BUY':
                st.success(f"🚀 BUY | Confidence: {confidence*100:.1f}% | @ {sig_time}")
            elif signal == 'SELL':
                st.error(f"🔻 SELL | Confidence: {confidence*100:.1f}% | @ {sig_time}")
            else:
                if last_signal.get('model_mode') == MODEL_MODE_DUAL:
                    st.info(f"⏸️ WAIT | H1: {last_signal.get('h1_dir')} M5: {last_signal.get('m5_dir')} | @ {sig_time}")
                else:
                    st.info(f"⏸️ WAIT | M5: {last_signal.get('m5_dir')} | @ {sig_time}")
        
        # Hiển thị log từ auto-trade thread
        auto_messages = st.session_state.trader.get_auto_messages(MAX_TRADE_LOG_DISPLAY)
        if auto_messages:
            with st.expander("📜 Auto Trading Log", expanded=False):
                for msg in reversed(auto_messages):
                    icon = "✅" if msg['type'] == "success" else "ℹ️" if msg['type'] == "info" else "❌"
                    st.caption(f"[{msg['time']}] {icon} {msg['message']}")
        
        # UI refresh nhẹ nhàng (không chặn bot)
        time.sleep(ui_refresh)
        st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>LSTM Dual-Timeframe Predictor + Auto Trading v2.0</p>
        <p>Sử dụng nến H1 và M5 để dự đoán xu hướng giá và tự động giao dịch</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== BACKTEST SECTION ==========
    st.divider()
    st.subheader("📊 Backtest")
    
    if st.session_state.models_loaded:
        bt_col1, bt_col2 = st.columns([1, 3])
        
        with bt_col1:
            st.markdown("**Cấu hình Backtest:**")
            
            # Date range selection
            bt_mode = st.radio("Chế độ", ["Tùy chọn ngày", "Validation Set", "Toàn bộ"], key="bt_mode")
            
            if bt_mode == "Tùy chọn ngày":
                # Load data để lấy min/max date
                if h1_path and m5_path:
                    try:
                        temp_df = pd.read_csv(m5_path)
                        temp_df['Time'] = pd.to_datetime(temp_df['Time'])
                        min_date = temp_df['Time'].min().date()
                        max_date = temp_df['Time'].max().date()
                    except:
                        min_date = datetime(2025, 1, 1).date()
                        max_date = datetime.now().date()
                else:
                    min_date = datetime(2025, 1, 1).date()
                    max_date = datetime.now().date()
                
                bt_start_date = st.date_input("Từ ngày", value=min_date, min_value=min_date, max_value=max_date, key="bt_start")
                bt_end_date = st.date_input("Đến ngày", value=max_date, min_value=min_date, max_value=max_date, key="bt_end")
            else:
                bt_start_date = None
                bt_end_date = None
            
            st.divider()
            bt_lot = st.number_input("Lot Size (BT)", min_value=0.01, max_value=10.0, value=DEFAULT_LOT, step=0.01, key="bt_lot")
            bt_sl = st.number_input("SL (pips)", min_value=10, max_value=1000, value=DEFAULT_SL_PIPS, step=10, key="bt_sl")
            bt_tp = st.number_input("TP (pips)", min_value=10, max_value=1000, value=DEFAULT_TP_PIPS, step=10, key="bt_tp")
            bt_min_conf = st.slider("Min Confidence (BT)", 0.5, 0.95, MIN_CONFIDENCE, 0.05, key="bt_conf")
            
            run_backtest = st.button("🚀 Chạy Backtest", type="primary", width='stretch')
        
        with bt_col2:
            bt_has_files = m5_path is not None
            if model_mode == MODEL_MODE_DUAL:
                bt_has_files = bt_has_files and h1_path is not None
            
            if run_backtest and bt_has_files:
                with st.spinner("Đang chạy backtest..."):
                    try:
                        # Load data
                        m5_df = pd.read_csv(m5_path)
                        h1_df = pd.read_csv(h1_path) if model_mode == MODEL_MODE_DUAL else pd.DataFrame()
                        
                        # Determine parameters based on mode
                        if bt_mode == "Tùy chọn ngày":
                            start_dt = datetime.combine(bt_start_date, datetime.min.time())
                            end_dt = datetime.combine(bt_end_date, datetime.max.time())
                            validation_only = False
                        elif bt_mode == "Validation Set":
                            start_dt = None
                            end_dt = None
                            validation_only = True
                        else:  # Toàn bộ
                            start_dt = None
                            end_dt = None
                            validation_only = False
                        
                        # Run backtest
                        backtester = Backtester(
                            trainer=st.session_state.trainer,
                            lot=bt_lot,
                            sl_pips=bt_sl,
                            tp_pips=bt_tp,
                            min_confidence=bt_min_conf,
                            model_mode=model_mode
                        )
                        result = backtester.run_backtest(
                            h1_df, m5_df,
                            start_date=start_dt,
                            end_date=end_dt,
                            validation_only=validation_only
                        )
                        
                        # Thông tin data range
                        if bt_mode == "Tùy chọn ngày":
                            st.info(f"📅 Backtest từ {bt_start_date} đến {bt_end_date}")
                        elif bt_mode == "Validation Set":
                            val_start = int(len(m5_df) * 0.8)
                            st.info(f"📊 Backtest trên VALIDATION SET: {len(m5_df) - val_start} nến M5 (20% cuối)")
                        else:
                            st.info(f"📊 Backtest trên TOÀN BỘ dữ liệu: {len(m5_df)} nến M5")
                        
                        # Display results
                        st.markdown("### 📈 Kết quả Backtest")
                        
                        # Metrics row 1
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Tổng lệnh", result.total_trades)
                        with m2:
                            st.metric("Win Rate", f"{result.win_rate:.1f}%")
                        with m3:
                            st.metric("Profit Factor", f"{result.profit_factor:.2f}")
                        with m4:
                            profit_color = "normal" if result.total_profit >= 0 else "inverse"
                            st.metric("Tổng Profit", f"${result.total_profit:,.2f}", delta_color=profit_color)
                        
                        # Metrics row 2
                        m5, m6, m7, m8 = st.columns(4)
                        with m5:
                            st.metric("Thắng/Thua", f"{result.winning_trades}/{result.losing_trades}")
                        with m6:
                            st.metric("Max Drawdown", f"{result.max_drawdown:.1f}%")
                        with m7:
                            st.metric("Avg Win", f"${result.avg_win:.2f}")
                        with m8:
                            st.metric("Avg Loss", f"${result.avg_loss:.2f}")
                        
                        # Equity curve
                        if result.equity_curve:
                            st.markdown("**Equity Curve:**")
                            equity_df = pd.DataFrame({'Equity': result.equity_curve})
                            st.line_chart(equity_df)
                        
                        # Trade history
                        trades_df = backtester.get_trades_df(result)
                        if not trades_df.empty:
                            st.markdown("**Lịch sử giao dịch:**")
                            st.dataframe(trades_df, width='stretch', height=300)
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi backtest: {e}")
            elif not bt_has_files:
                if model_mode == MODEL_MODE_SINGLE_M5:
                    st.warning("Cần chọn file M5 để chạy backtest")
                else:
                    st.warning("Cần chọn file H1 và M5 để chạy backtest")
            else:
                st.info("Nhấn 'Chạy Backtest' để bắt đầu mô phỏng giao dịch trên dữ liệu lịch sử")
    else:
        st.warning("⚠️ Cần load models trước khi chạy backtest")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p>LSTM Dual-Timeframe Predictor + Auto Trading v2.0</p>
        <p>Sử dụng nến H1 và M5 để dự đoán xu hướng giá và tự động giao dịch</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
