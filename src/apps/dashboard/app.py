"""
Streamlit UI Application (API Client Version)
Giao diện người dùng cho hệ thống dự đoán LSTM kết nối qua FastAPI.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import time

from src.core.constants import (
    DEFAULT_SYMBOL, DEFAULT_LOT, DEFAULT_SL_PIPS, DEFAULT_TP_PIPS,
    MAX_POSITIONS, MIN_CONFIDENCE,
    AUTO_TRADE_INTERVAL, UI_REFRESH_INTERVAL,
    MODEL_MODE_DUAL, MODEL_MODE_SINGLE_M5, DEFAULT_MODEL_MODE,
)
from src.apps.dashboard.api_client import APIClient

# Page config
st.set_page_config(
    page_title="LSTM AutoTrader Dashboard",
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
    .buy-signal { background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); color: white; }
    .sell-signal { background: linear-gradient(135deg, #d63031 0%, #e17055 100%); color: white; }
    .wait-signal { background: linear-gradient(135deg, #636e72 0%, #b2bec3 100%); color: white; }
    .connected-status { background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); padding: 0.5rem; border-radius: 5px; color: white; text-align: center; }
    .disconnected-status { background: linear-gradient(135deg, #636e72 0%, #b2bec3 100%); padding: 0.5rem; border-radius: 5px; color: white; text-align: center; }
    .position-card { background: #f0f0f0; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #00b894; }
    .position-card.sell { border-left-color: #d63031; }
</style>
""", unsafe_allow_html=True)


def display_prediction_box(direction: str, probability: float, label: str):
    color = "#00b894" if direction == "UP" else "#d63031"
    icon = "📈" if direction == "UP" else "📉"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}99 0%, {color} 100%); 
                padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
        <h3 style="color: white; margin: 0;">{label}</h3>
        <h2 style="color: white; margin: 0.5rem 0;">{icon} {direction}</h2>
        <p style="color: white; margin: 0;">Confidence: {probability*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)


def display_combined_signal(signal: str, confidence: float = None, reason: str = None):
    css_class = "buy-signal" if signal == "BUY" else "sell-signal" if signal == "SELL" else "wait-signal"
    icon = "🚀 BUY" if signal == "BUY" else "🔻 SELL" if signal == "SELL" else "⏸️ WAIT"
    
    content = f"<h2 style='margin: 0;'>{icon}</h2>"
    if confidence: content += f"<p style='margin: 0.5rem 0 0 0;'>Confidence: {confidence*100:.1f}%</p>"
    if reason: content += f"<p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>{reason}</p>"
    
    st.markdown(f'<div class="prediction-box {css_class}">{content}</div>', unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">📊 LSTM AutoTrader Dashboard</div>', unsafe_allow_html=True)
    
    # Check Backend Health
    health = APIClient.get_system_health()
    backend_online = bool(health)
    
    if not backend_online:
        st.error("🚨Hiện Không thể kết nối tới Backend FastAPI. Hãy đảm bảo uvicorn đang chạy!")
        return

    # Lấy trạng thái hiện tại từ backend
    mt5_status = APIClient.mt5_status()
    auto_status = APIClient.get_auto_trade_status()
    db_models = APIClient.get_models().get("items", [])
    
    h1_active = any(m for m in db_models if m["timeframe"] == "H1" and m["is_active"])
    m5_active = any(m for m in db_models if m["timeframe"] == "M5" and m["is_active"])
    
    is_connected = mt5_status.get("connected", False)
    is_auto_trading = auto_status.get("running", False)

    # Sidebar
    with st.sidebar:

        
        # MT5 Connection
        st.subheader("🔌 MT5 Connection")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Kết nối", disabled=is_connected):
                res = APIClient.mt5_connect()
                st.rerun()
        with col_c2:
            if st.button("Ngắt", disabled=not is_connected):
                APIClient.mt5_disconnect()
                st.rerun()
                
        if is_connected:
            st.markdown('<div class="connected-status">🟢 Đã kết nối MT5</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="disconnected-status">🔴 Chưa kết nối</div>', unsafe_allow_html=True)
            
        st.divider()
        
        # Model Mode
        st.subheader("🎯 Model Mode")
        model_mode = st.radio(
            "Chế độ dự đoán",
            options=[MODEL_MODE_DUAL, MODEL_MODE_SINGLE_M5],
            index=0 if auto_status.get("model_mode") == MODEL_MODE_DUAL else 1
        )
        
        # Auto Trading
        st.subheader("🤖 Auto Trading")
        auto_interval = st.number_input("Bot Interval (s)", 0.5, 10.0, float(auto_status.get("interval") or AUTO_TRADE_INTERVAL))
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            if st.button("Bật Bot", type="primary", disabled=is_auto_trading or not is_connected):
                APIClient.start_auto_trade(auto_interval, model_mode)
                st.rerun()
        with col_a2:
            if st.button("Tắt Bot", disabled=not is_auto_trading):
                APIClient.stop_auto_trade()
                st.rerun()
                
        if is_auto_trading:
            st.success("Bot đang chạy...")
            
        st.divider()
        
        # Training
        st.subheader("📚 Huấn luyện Model")
        lookback = st.slider("Lookback", 12, 96, 48)
        epochs = st.slider("Epochs", 10, 200, 100)
        train_tf = st.selectbox("Timeframe", ["H1", "M5"])
        
        st.caption("Dữ liệu train sẽ được lấy tự động từ Database (ưu tiên crawl trước)")
        if st.button(f"Train {train_tf}", width="stretch", type="primary"):
            with st.spinner("Đang lấy data từ DB và train (background)..."):
                # Gửi request với data_file rỗng, server sẽ tự hiểu là query DB
                res = APIClient.train_model(train_tf, "", lookback, epochs, 32, 0.8)
                if "detail" in res:
                    st.error(res["detail"])
                else:
                    st.success(res.get("message", "Đã gửi request!"))
                time.sleep(1)
                st.rerun()

        st.divider()
        
        # Crawl Data
        st.subheader("📥 Crawl Data")
        crawl_tf = st.selectbox("TF Crawl", ["M5", "H1"])
        if st.button("Crawl từ MT5", disabled=not is_connected):
            with st.spinner("Đang lấy data..."):
                res = APIClient.crawl_data("XAUUSD", "2024-01-01", datetime.now().strftime("%Y-%m-%d"), crawl_tf)
                st.success(res.get("message"))

    # Main Content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Model H1 (Active)")
        h1_model = next((m for m in db_models if m["timeframe"] == "H1" and m["is_active"]), None)
        if h1_model:
            metrics = h1_model.get("metrics", {})
            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
            st.caption(f"Phiên bản: {h1_model['version']}")
        else:
            st.warning("Chưa có H1 model active")

    with col2:
        st.subheader("📊 Model M5 (Active)")
        m5_model = next((m for m in db_models if m["timeframe"] == "M5" and m["is_active"]), None)
        if m5_model:
            metrics = m5_model.get("metrics", {})
            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
            st.caption(f"Phiên bản: {m5_model['version']}")
        else:
            st.warning("Chưa có M5 model active")
            
    with col3:
        st.subheader("🎯 Dự đoán Realtime")
        if st.button("🔮 Predict Now ", type="primary", width="stretch"):
            if not is_connected:
                st.error("Cần kết nối MT5 để lấy giá realtime")
            else:
                with st.spinner("Calling API..."):
                    res = APIClient.predict_realtime(model_mode)
                    if "detail" in res:
                        st.error(res["detail"])
                    else:
                        if res.get("h1"):
                            display_prediction_box(res["h1"]["direction"], res["h1"]["probability"], "H1")
                        if res.get("m5"):
                            display_prediction_box(res["m5"]["direction"], res["m5"]["probability"], "M5")
                        if res.get("combined"):
                            c = res["combined"]
                            display_combined_signal(c["signal"], c.get("confidence"), c.get("reason"))

    st.divider()
    
    # Row 2: Trading Info
    col_acc, col_pos, col_hist = st.columns([1, 1.5, 1.5])
    
    with col_acc:
        st.subheader("💳 Tài khoản (MT5)")
        if is_connected and mt5_status.get("account_info"):
            acc = mt5_status["account_info"]
            st.metric("Balance", f"${acc['balance']:,.2f}")
            st.metric("Equity", f"${acc['equity']:,.2f}")
            st.metric("Profit", f"${acc['profit']:+,.2f}")
        else:
            st.caption("Chưa kết nối")
            
    with col_pos:
        st.subheader("📋 Vị thế đang mở")
        if is_connected:
            positions = APIClient.mt5_positions()
            if not positions:
                st.info("Không có lệnh mở")
            else:
                for pos in positions:
                    profit_color = "#00b894" if pos['profit'] >= 0 else "#d63031"
                    card_class = "sell" if pos['type'] == "SELL" else ""
                    st.markdown(f"""
                    <div class="position-card {card_class}">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <strong>#{pos['ticket']}</strong> - {pos['type']}<br>
                                <small>{pos['price_open']} → {pos['price_current']}</small>
                            </div>
                            <strong style="color: {profit_color}; font-size: 1.2rem;">{pos['profit']:+.2f}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("**Test Vào Lệnh:**")
            col_b, col_s = st.columns(2)
            with col_b:
                if st.button("Buy", width="stretch"):
                    APIClient.execute_trade("BUY", 0.99)
                    st.rerun()
            with col_s:
                if st.button("Sell", width="stretch"):
                    APIClient.execute_trade("SELL", 0.99)
                    st.rerun()

    with col_hist:
        st.subheader("📜 Lịch sử Trade")
        trades = APIClient.get_trades(10)
        if trades:
            for t in trades:
                icon = "🟢" if t['direction'] == "BUY" else "🔴"
                st.caption(f"{icon} #{t['id']} | {t['direction']} | PnL: {t.get('pnl', 0)} | {t['entry_time'][:16]}")
        else:
            st.info("Chưa có trade nào trong DB")

    if is_auto_trading:
        time.sleep(UI_REFRESH_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main()
