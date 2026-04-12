"""
Backtester Module
Mô phỏng giao dịch trên dữ liệu lịch sử để đánh giá hiệu quả model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from config import (
    DEFAULT_LOT, DEFAULT_SL_PIPS, DEFAULT_TP_PIPS,
    MIN_CONFIDENCE, XAUUSD_POINT, PIP_MULTIPLIER,
    LOOKBACK, TRAIN_RATIO,
    BACKTEST_INITIAL_EQUITY, BACKTEST_PROFIT_MULTIPLIER,
    MODEL_MODE_DUAL, MODEL_MODE_SINGLE_M5, DEFAULT_MODEL_MODE,
)


@dataclass
class Trade:
    """Thông tin một giao dịch"""
    entry_time: datetime
    entry_price: float
    signal: str  # BUY or SELL
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  
    profit: float = 0.0
    profit_pips: float = 0.0


@dataclass
class BacktestResult:
    """Kết quả backtest"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    total_profit_pips: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_profit_per_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    signals_history: List[Dict] = field(default_factory=list)


class Backtester:
    """
    Backtest trading signals on historical data
    """
    
    def __init__(
        self,
        trainer,
        lot: float = DEFAULT_LOT,
        sl_pips: float = DEFAULT_SL_PIPS,
        tp_pips: float = DEFAULT_TP_PIPS,
        min_confidence: float = MIN_CONFIDENCE,
        point: float = XAUUSD_POINT,
        model_mode: str = DEFAULT_MODEL_MODE
    ):
        """
        Args:
            trainer: Trainer instance với models đã load
            lot: Lot size
            sl_pips: Stop loss in pips
            tp_pips: Take profit in pips
            min_confidence: Minimum confidence để vào lệnh
            point: Point value của symbol (0.01 for gold)
            model_mode: Chế độ model ("dual" hoặc "single_m5")
        """
        self.trainer = trainer
        self.lot = lot
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.min_confidence = min_confidence
        self.point = point
        self.pip_value = PIP_MULTIPLIER * point
        self.model_mode = model_mode
    
    def run_backtest(
        self,
        h1_df: pd.DataFrame,
        m5_df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        validation_only: bool = False,
        train_ratio: float = TRAIN_RATIO,
        use_m5_for_timing: bool = True
    ) -> BacktestResult:
        """
        Chạy backtest trên dữ liệu lịch sử
        
        Args:
            h1_df: DataFrame dữ liệu H1
            m5_df: DataFrame dữ liệu M5
            start_date: Ngày bắt đầu backtest (None = từ đầu)
            end_date: Ngày kết thúc backtest (None = đến cuối)
            validation_only: True = chỉ backtest trên phần validation (20% cuối)
            train_ratio: Tỉ lệ dữ liệu đã train (mặc định 0.8)
            use_m5_for_timing: Dùng M5 để timing entry/exit
            
        Returns:
            BacktestResult với chi tiết giao dịch
        """
        result = BacktestResult()
        current_trade: Optional[Trade] = None
        equity = BACKTEST_INITIAL_EQUITY
        peak_equity = equity
        result.equity_curve.append(equity)
        
        # Sync H1 and M5 data by time
        h1_df = h1_df.copy()
        m5_df = m5_df.copy()
        
        # Ensure Time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(h1_df['Time']):
            h1_df['Time'] = pd.to_datetime(h1_df['Time'])
        if not pd.api.types.is_datetime64_any_dtype(m5_df['Time']):
            m5_df['Time'] = pd.to_datetime(m5_df['Time'])
        
        # Use M5 timeframe for iteration
        lookback = LOOKBACK
        
        # Xác định điểm bắt đầu và kết thúc backtest
        if start_date is not None or end_date is not None:
            # Lọc theo khoảng thời gian chọn
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
                m5_mask = m5_df['Time'] >= start_date
                h1_mask = h1_df['Time'] >= start_date
            else:
                m5_mask = pd.Series([True] * len(m5_df))
                h1_mask = pd.Series([True] * len(h1_df))
            
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
                m5_mask = m5_mask & (m5_df['Time'] <= end_date)
                h1_mask = h1_mask & (h1_df['Time'] <= end_date)
            
            # Tìm start_idx trong m5_df gốc
            first_valid_idx = m5_mask.idxmax() if m5_mask.any() else 0
            last_valid_idx = m5_mask[::-1].idxmax() if m5_mask.any() else len(m5_df) - 1
            
            start_idx = max(first_valid_idx, lookback + 10)
            end_idx = last_valid_idx
            
            print(f"📅 Backtest từ {start_date} đến {end_date}")
            print(f"   M5: {m5_mask.sum()} nến | H1: {h1_mask.sum()} nến")
            
        elif validation_only:
            # Tìm thời gian bắt đầu validation (20% cuối theo thời gian của M5)
            m5_train_size = int(len(m5_df) * train_ratio)
            val_start_time = m5_df.iloc[m5_train_size]['Time']
            
            # Tìm H1 validation tương ứng
            h1_val_mask = h1_df['Time'] >= val_start_time
            h1_train_size = len(h1_df) - h1_val_mask.sum()
            
            start_idx = max(m5_train_size, lookback + 10)
            end_idx = len(m5_df) - 1
            
            print(f"📊 Backtest chỉ trên VALIDATION SET:")
            print(f"   Thời gian bắt đầu val: {val_start_time}")
            print(f"   M5: Train {m5_train_size} nến | Val {len(m5_df) - m5_train_size} nến")
            print(f"   H1: Train {h1_train_size} nến | Val {h1_val_mask.sum()} nến")
        else:
            # Backtest trên toàn bộ dữ liệu
            start_idx = lookback + 10
            end_idx = len(m5_df) - 1
        
        for i in range(start_idx, end_idx):
            current_time = m5_df.iloc[i]['Time']
            current_price = m5_df.iloc[i]['Close']
            next_bar = m5_df.iloc[i + 1]
            
            # Get corresponding H1 data up to current time
            # Chỉ lấy H1 trong phạm vi validation nếu validation_only
            h1_subset = h1_df[h1_df['Time'] <= current_time].tail(lookback + 10)
            m5_subset = m5_df.iloc[max(0, i - lookback - 10):i + 1]
            
            if len(h1_subset) < lookback and self.model_mode == MODEL_MODE_DUAL:
                continue
            if len(m5_subset) < lookback:
                continue
            
            # Check existing trade for TP/SL
            if current_trade is not None:
                high_price = next_bar['High']
                low_price = next_bar['Low']
                close_price = next_bar['Close']
                
                if current_trade.signal == "BUY":
                    # Check SL
                    sl_price = current_trade.entry_price - self.sl_pips * self.pip_value
                    tp_price = current_trade.entry_price + self.tp_pips * self.pip_value
                    
                    if low_price <= sl_price:
                        current_trade.exit_time = next_bar['Time']
                        current_trade.exit_price = sl_price
                        current_trade.exit_reason = "SL"
                        current_trade.profit_pips = -self.sl_pips
                        current_trade.profit = current_trade.profit_pips * self.lot * BACKTEST_PROFIT_MULTIPLIER
                        result.trades.append(current_trade)
                        equity += current_trade.profit
                        current_trade = None
                    elif high_price >= tp_price:
                        current_trade.exit_time = next_bar['Time']
                        current_trade.exit_price = tp_price
                        current_trade.exit_reason = "TP"
                        current_trade.profit_pips = self.tp_pips
                        current_trade.profit = current_trade.profit_pips * self.lot * 10
                        result.trades.append(current_trade)
                        equity += current_trade.profit
                        current_trade = None
                else:  # SELL
                    sl_price = current_trade.entry_price + self.sl_pips * self.pip_value
                    tp_price = current_trade.entry_price - self.tp_pips * self.pip_value
                    
                    if high_price >= sl_price:
                        current_trade.exit_time = next_bar['Time']
                        current_trade.exit_price = sl_price
                        current_trade.exit_reason = "SL"
                        current_trade.profit_pips = -self.sl_pips
                        current_trade.profit = current_trade.profit_pips * self.lot * 10
                        result.trades.append(current_trade)
                        equity += current_trade.profit
                        current_trade = None
                    elif low_price <= tp_price:
                        current_trade.exit_time = next_bar['Time']
                        current_trade.exit_price = tp_price
                        current_trade.exit_reason = "TP"
                        current_trade.profit_pips = self.tp_pips
                        current_trade.profit = current_trade.profit_pips * self.lot * 10
                        result.trades.append(current_trade)
                        equity += current_trade.profit
                        current_trade = None
            
            # Generate signal if no open trade
            if current_trade is None:
                try:
                    prediction = self.trainer.predict(
                        h1_subset if self.model_mode == MODEL_MODE_DUAL else None,
                        m5_subset,
                        model_mode=self.model_mode
                    )
                    
                    if 'combined' in prediction:
                        signal = prediction['combined'].get('signal')
                        confidence = prediction['combined'].get('confidence', 0)
                        
                        result.signals_history.append({
                            'time': current_time,
                            'signal': signal,
                            'confidence': confidence if confidence else 0,
                            'h1_dir': prediction.get('H1', {}).get('direction'),
                            'm5_dir': prediction.get('M5', {}).get('direction'),
                            'price': current_price
                        })
                        
                        if signal in ['BUY', 'SELL'] and confidence and confidence >= self.min_confidence:
                            current_trade = Trade(
                                entry_time=current_time,
                                entry_price=current_price,
                                signal=signal
                            )
                except Exception as e:
                    # Log error for debugging (only first 5)
                    if len(result.signals_history) < 5:
                        print(f"⚠️ Prediction error at {current_time}: {e}")
                    continue
            
            # Update equity curve
            result.equity_curve.append(equity)
            
            # Update max drawdown
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity * 100
            if drawdown > result.max_drawdown:
                result.max_drawdown = drawdown
        
        # Close any remaining trade at last price
        if current_trade is not None:
            last_price = m5_df.iloc[-1]['Close']
            if current_trade.signal == "BUY":
                current_trade.profit_pips = (last_price - current_trade.entry_price) / self.pip_value
            else:
                current_trade.profit_pips = (current_trade.entry_price - last_price) / self.pip_value
            current_trade.profit = current_trade.profit_pips * self.lot * BACKTEST_PROFIT_MULTIPLIER
            current_trade.exit_time = m5_df.iloc[-1]['Time']
            current_trade.exit_price = last_price
            current_trade.exit_reason = "END"
            result.trades.append(current_trade)
            equity += current_trade.profit
            result.equity_curve.append(equity)
        
        # Calculate statistics
        self._calculate_statistics(result)
        
        # Debug summary
        buy_signals = sum(1 for s in result.signals_history if s.get('signal') == 'BUY')
        sell_signals = sum(1 for s in result.signals_history if s.get('signal') == 'SELL')
        wait_signals = sum(1 for s in result.signals_history if s.get('signal') == 'WAIT')
        print(f"\n📊 Backtest Summary:")
        print(f"   Signals: BUY={buy_signals}, SELL={sell_signals}, WAIT={wait_signals}")
        print(f"   Trades executed: {len(result.trades)}")
        print(f"   Final equity: ${equity:,.2f}")
        
        return result
    
    def _calculate_statistics(self, result: BacktestResult):
        """Tính toán các thống kê từ danh sách trades"""
        if not result.trades:
            return
        
        result.total_trades = len(result.trades)
        
        wins = [t for t in result.trades if t.profit > 0]
        losses = [t for t in result.trades if t.profit <= 0]
        
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = (result.winning_trades / result.total_trades * 100) if result.total_trades > 0 else 0
        
        result.total_profit = sum(t.profit for t in result.trades)
        result.total_profit_pips = sum(t.profit_pips for t in result.trades)
        
        result.avg_profit_per_trade = result.total_profit / result.total_trades if result.total_trades > 0 else 0
        
        if wins:
            result.avg_win = sum(t.profit for t in wins) / len(wins)
            result.largest_win = max(t.profit for t in wins)
        
        if losses:
            result.avg_loss = sum(t.profit for t in losses) / len(losses)
            result.largest_loss = min(t.profit for t in losses)
        
        # Profit factor
        total_wins = sum(t.profit for t in wins) if wins else 0
        total_losses = abs(sum(t.profit for t in losses)) if losses else 1
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    def get_trades_df(self, result: BacktestResult) -> pd.DataFrame:
        """Convert trades to DataFrame for display"""
        if not result.trades:
            return pd.DataFrame()
        
        trades_data = []
        for t in result.trades:
            trades_data.append({
                'Entry Time': t.entry_time,
                'Signal': t.signal,
                'Entry Price': t.entry_price,
                'Exit Time': t.exit_time,
                'Exit Price': t.exit_price,
                'Exit Reason': t.exit_reason,
                'Profit ($)': round(t.profit, 2),
                'Profit (pips)': round(t.profit_pips, 1)
            })
        
        return pd.DataFrame(trades_data)
    
    def get_signals_df(self, result: BacktestResult) -> pd.DataFrame:
        """Convert signals history to DataFrame"""
        if not result.signals_history:
            return pd.DataFrame()
        
        return pd.DataFrame(result.signals_history)
