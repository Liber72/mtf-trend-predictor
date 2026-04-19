"""
MT5 Trader Module
Tự động đặt lệnh giao dịch trong MT5 dựa trên tín hiệu dự đoán
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import time

from config import (
    DEFAULT_SYMBOL, ALTERNATIVE_SYMBOLS,
    DEFAULT_LOT, DEFAULT_SL_PIPS, DEFAULT_TP_PIPS,
    MAX_POSITIONS, MIN_CONFIDENCE, MAGIC_NUMBER,
    ORDER_DEVIATION, PIP_MULTIPLIER,
    DEFAULT_TRAILING_SL_LEVELS,
    MODEL_MODE_DUAL, MODEL_MODE_SINGLE_M5, DEFAULT_MODEL_MODE,
)


class MT5Trader:
    """
    Quản lý giao dịch tự động với MT5
    """
    
    TIMEFRAME_MAP = {
        "M5": mt5.TIMEFRAME_M5,
        "H1": mt5.TIMEFRAME_H1,
    }
    
    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        lot: float = DEFAULT_LOT,
        sl_pips: float = DEFAULT_SL_PIPS,
        tp_pips: float = DEFAULT_TP_PIPS,
        max_positions: int = MAX_POSITIONS,
        min_confidence: float = MIN_CONFIDENCE,
        magic_number: int = MAGIC_NUMBER,
        trailing_sl_levels: list = None,
        trailing_sl_enabled: bool = False
    ):
        """
        Args:
            symbol: Symbol giao dịch
            lot: Khối lượng lệnh
            sl_pips: Số pips stop loss
            tp_pips: Số pips take profit
            max_positions: Số lệnh tối đa được phép mở
            min_confidence: Độ tin cậy tối thiểu để vào lệnh
            magic_number: Magic number để nhận diện lệnh của bot
            trailing_sl_levels: List[(trigger_pips, sl_pips)] — các mốc trailing
            trailing_sl_enabled: Bật/tắt trailing SL
        """
        self.symbol = symbol
        self.lot = lot
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.max_positions = max_positions
        self.min_confidence = min_confidence
        self.magic_number = magic_number
        self.connected = False
        self.trade_log = []
        self.last_order_candle_time = None  # Track candle time of last order
        
        # Trailing Stop Loss config
        self.trailing_sl_levels = trailing_sl_levels or list(DEFAULT_TRAILING_SL_LEVELS)
        self.trailing_sl_enabled = trailing_sl_enabled
        self.model_mode = DEFAULT_MODEL_MODE  # Chế độ model
    
    def connect(self) -> Tuple[bool, str]:
        """
        Kết nối MT5
        
        Returns:
            Tuple (success, message)
        """
        if not mt5.initialize():
            error = mt5.last_error()
            return False, f"Lỗi khởi tạo MT5: {error}"
        
        # Kiểm tra symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            # Thử các symbol thay thế
            alternatives = ALTERNATIVE_SYMBOLS
            for alt in alternatives:
                if mt5.symbol_info(alt) is not None:
                    self.symbol = alt
                    symbol_info = mt5.symbol_info(alt)
                    break
        
        if symbol_info is None:
            mt5.shutdown()
            return False, f"Không tìm thấy symbol {self.symbol}"
        
        # Enable symbol nếu cần
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                mt5.shutdown()
                return False, f"Không thể chọn symbol {self.symbol}"
        
        self.connected = True
        version = mt5.version()
        return True, f"Kết nối thành công! MT5 v{version[0]}.{version[1]}"
    
    def disconnect(self):
        """Ngắt kết nối MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Lấy thông tin tài khoản
        
        Returns:
            Dict thông tin tài khoản hoặc None
        """
        if not self.connected:
            return None
        
        account = mt5.account_info()
        if account is None:
            return None
        
        return {
            'login': account.login,
            'server': account.server,
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'margin_free': account.margin_free,
            'margin_level': account.margin_level,
            'profit': account.profit,
            'currency': account.currency,
            'leverage': account.leverage
        }
    
    def get_symbol_info(self) -> Optional[Dict]:
        """
        Lấy thông tin symbol
        
        Returns:
            Dict thông tin symbol
        """
        if not self.connected:
            return None
        
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return None
        
        tick = mt5.symbol_info_tick(self.symbol)
        
        # Store filling mode for later use
        self._filling_mode = self._get_filling_mode(info)
        
        return {
            'symbol': self.symbol,
            'bid': tick.bid if tick else 0,
            'ask': tick.ask if tick else 0,
            'spread': info.spread,
            'point': info.point,
            'digits': info.digits,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'filling_mode': self._get_filling_mode(info)
        }
    
    def _get_filling_mode(self, symbol_info=None):
        """
        Tự động detect filling mode phù hợp với broker
        
        Returns:
            Filling mode phù hợp
        """
        if symbol_info is None:
            symbol_info = mt5.symbol_info(self.symbol)
        
        if symbol_info is None:
            return mt5.ORDER_FILLING_IOC
        
        filling = symbol_info.filling_mode
        
        # Filling mode values:
        # 1 = SYMBOL_FILLING_FOK (Fill or Kill)
        # 2 = SYMBOL_FILLING_IOC (Immediate or Cancel)
        # Check which filling modes are supported
        if filling & 1:  # FOK supported
            return mt5.ORDER_FILLING_FOK
        elif filling & 2:  # IOC supported
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN
    
    def get_realtime_data(self, timeframe: str = "M5", count: int = 100) -> Optional[pd.DataFrame]:
        """
        Lấy dữ liệu nến realtime từ MT5
        
        Args:
            timeframe: "M5" hoặc "H1"
            count: Số nến cần lấy
            
        Returns:
            DataFrame với dữ liệu nến
        """
        if not self.connected:
            return None
        
        if timeframe not in self.TIMEFRAME_MAP:
            return None
        
        mt5_tf = self.TIMEFRAME_MAP[timeframe]
        rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, count)
        
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'TickVolume', 'Spread', 'RealVolume']
        
        return df
    
    def get_open_positions(self) -> List[Dict]:
        """
        Lấy danh sách vị thế đang mở của bot này
        
        Returns:
            List các vị thế
        """
        if not self.connected:
            return []
        
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        
        result = []
        for pos in positions:
            # Chỉ lấy các lệnh của bot này
            if pos.magic == self.magic_number:
                result.append({
                    'ticket': pos.ticket,
                    'time': datetime.fromtimestamp(pos.time),
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'comment': pos.comment
                })
        
        return result
    
    def get_all_positions(self) -> List[Dict]:
        """
        Lấy tất cả vị thế đang mở trên symbol
        
        Returns:
            List các vị thế
        """
        if not self.connected:
            return []
        
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        
        return [{
            'ticket': pos.ticket,
            'time': datetime.fromtimestamp(pos.time),
            'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
            'volume': pos.volume,
            'price_open': pos.price_open,
            'price_current': pos.price_current,
            'sl': pos.sl,
            'tp': pos.tp,
            'profit': pos.profit,
            'magic': pos.magic
        } for pos in positions]
    
    def open_position(self, signal: str, lot: Optional[float] = None) -> Tuple[bool, str, Optional[int]]:
        """
        Mở vị thế mới
        
        Args:
            signal: "BUY" hoặc "SELL"
            lot: Khối lượng (None = dùng mặc định)
            
        Returns:
            Tuple (success, message, ticket)
        """
        if not self.connected:
            return False, "Chưa kết nối MT5", None
        
        # Kiểm tra số lệnh hiện tại
        current_positions = self.get_open_positions()
        if len(current_positions) >= self.max_positions:
            return False, f"Đã đạt giới hạn {self.max_positions} lệnh", None
        
        # Lấy thông tin symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return False, f"Không tìm thấy symbol {self.symbol}", None
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False, "Không lấy được giá", None
        
        point = symbol_info.point
        volume = lot if lot else self.lot
        
        # Tính SL/TP
        if signal == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = price - self.sl_pips * point * PIP_MULTIPLIER
            tp = price + self.tp_pips * point * PIP_MULTIPLIER
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = price + self.sl_pips * point * PIP_MULTIPLIER
            tp = price - self.tp_pips * point * PIP_MULTIPLIER
        
        # Tạo request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": ORDER_DEVIATION,
            "magic": self.magic_number,
            "comment": f"LSTM_Bot_{signal}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(),
        }
        
        # Gửi lệnh
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            return False, f"Lỗi gửi lệnh: {error}", None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Lỗi: {result.comment} (code: {result.retcode})", None
        
        # Log trade
        trade_info = {
            'time': datetime.now(),
            'action': 'OPEN',
            'signal': signal,
            'ticket': result.order,
            'price': price,
            'volume': volume,
            'sl': sl,
            'tp': tp
        }
        self.trade_log.append(trade_info)
        
        return True, f"Mở lệnh {signal} thành công! Ticket: {result.order}", result.order
    
    def close_position(self, ticket: int) -> Tuple[bool, str]:
        """
        Đóng vị thế theo ticket
        
        Args:
            ticket: Ticket của lệnh
            
        Returns:
            Tuple (success, message)
        """
        if not self.connected:
            return False, "Chưa kết nối MT5"
        
        # Lấy thông tin vị thế
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return False, f"Không tìm thấy lệnh {ticket}"
        
        pos = position[0]
        tick = mt5.symbol_info_tick(self.symbol)
        
        # Xác định loại lệnh đóng
        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": ORDER_DEVIATION,
            "magic": self.magic_number,
            "comment": "LSTM_Bot_Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(),
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            return False, f"Lỗi: {mt5.last_error()}"
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Lỗi: {result.comment} (code: {result.retcode})"
        
        # Log trade
        self.trade_log.append({
            'time': datetime.now(),
            'action': 'CLOSE',
            'ticket': ticket,
            'price': price,
            'profit': pos.profit
        })
        
        return True, f"Đóng lệnh {ticket} thành công! Profit: {pos.profit:.2f}"
    
    def close_all_positions(self) -> Tuple[int, int]:
        """
        Đóng tất cả lệnh của bot
        
        Returns:
            Tuple (success_count, fail_count)
        """
        positions = self.get_open_positions()
        success = 0
        fail = 0
        
        for pos in positions:
            ok, _ = self.close_position(pos['ticket'])
            if ok:
                success += 1
            else:
                fail += 1
        
        return success, fail
    
    def modify_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> Tuple[bool, str]:
        """
        Sửa SL/TP của lệnh
        
        Args:
            ticket: Ticket của lệnh
            sl: Stop loss mới (None = giữ nguyên)
            tp: Take profit mới (None = giữ nguyên)
            
        Returns:
            Tuple (success, message)
        """
        if not self.connected:
            return False, "Chưa kết nối MT5"
        
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return False, f"Không tìm thấy lệnh {ticket}"
        
        pos = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": sl if sl else pos.sl,
            "tp": tp if tp else pos.tp,
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            return False, f"Lỗi: {mt5.last_error()}"
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Lỗi: {result.comment}"
        
        return True, "Sửa lệnh thành công!"
    
    def execute_signal(self, signal: str, confidence: float) -> Tuple[bool, str]:
        """
        Thực thi tín hiệu giao dịch
        
        Args:
            signal: "BUY", "SELL", hoặc "WAIT"
            confidence: Độ tin cậy (0-1)
            
        Returns:
            Tuple (executed, message)
        """
        if signal == "WAIT":
            return False, "Tín hiệu WAIT - không vào lệnh"
        
        if confidence < self.min_confidence:
            return False, f"Confidence {confidence:.1%} < {self.min_confidence:.1%} - không vào lệnh"
        
        # Kiểm tra giới hạn 1 lệnh mỗi nến M5
        current_candle_time = self._get_current_m5_candle_time()
        if current_candle_time and self.last_order_candle_time == current_candle_time:
            return False, f"Đã vào lệnh trong nến M5 hiện tại - chờ nến mới"
        
        # Đóng lệnh ngược chiều trước khi mở lệnh mới
        opposite_type = "SELL" if signal == "BUY" else "BUY"
        closed_count = self._close_opposite_positions(opposite_type)
        if closed_count > 0:
            print(f"🔄 Đã đóng {closed_count} lệnh {opposite_type} ngược chiều")
        
        # Thực thi lệnh
        success, message, ticket = self.open_position(signal)
        
        # Cập nhật candle time nếu thành công
        if success and current_candle_time:
            self.last_order_candle_time = current_candle_time
        
        return success, message
    
    def _close_opposite_positions(self, position_type: str) -> int:
        """
        Đóng tất cả lệnh theo loại (BUY hoặc SELL)
        
        Args:
            position_type: "BUY" hoặc "SELL"
            
        Returns:
            Số lệnh đã đóng
        """
        positions = self.get_open_positions()
        closed_count = 0
        
        for pos in positions:
            if pos['type'] == position_type:
                success, msg = self.close_position(pos['ticket'])
                if success:
                    closed_count += 1
                    print(f"   📝 Đóng lệnh ngược chiều: {msg}")
        
        return closed_count
    
    def _get_current_m5_candle_time(self) -> Optional[datetime]:
        """
        Lấy thời gian mở của nến M5 hiện tại
        
        Returns:
            Datetime của nến M5 hiện tại hoặc None
        """
        if not self.connected:
            return None
        
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 1)
        if rates is None or len(rates) == 0:
            return None
        
        return datetime.fromtimestamp(rates[0]['time'])
    
    def get_trade_log(self, limit: int = 20) -> List[Dict]:
        """
        Lấy lịch sử giao dịch gần đây
        
        Args:
            limit: Số lượng tối đa
            
        Returns:
            List các giao dịch
        """
        return self.trade_log[-limit:]
    
    # ========== TRAILING STOP LOSS ==========
    
    def trailing_stop_loss(self) -> List[Dict]:
        """
        Kiểm tra và cập nhật trailing SL cho tất cả positions đang mở.
        Được gọi mỗi 1 giây bởi background thread.
        
        Returns:
            List kết quả cho mỗi position đã xử lý
        """
        results = []
        
        if not self.connected:
            return results
        
        positions = self.get_open_positions()
        if not positions:
            return results
        
        # Lấy thông tin symbol để tính pip value
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return results
        
        point = symbol_info.point
        pip_value = point * PIP_MULTIPLIER  # 1 pip = PIP_MULTIPLIER points cho XAUUSD
        
        # Lấy giá hiện tại
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return results
        
        # Sort levels từ cao xuống thấp (check mức cao nhất trước)
        sorted_levels = sorted(self.trailing_sl_levels, key=lambda x: x[0], reverse=True)
        
        for pos in positions:
            entry_price = pos['price_open']
            current_sl = pos['sl']
            pos_type = pos['type']
            ticket = pos['ticket']
            
            # Tính profit hiện tại theo pips
            if pos_type == 'BUY':
                current_price = tick.bid  # Giá đóng BUY là bid
                current_profit_pips = (current_price - entry_price) / pip_value
            else:  # SELL
                current_price = tick.ask  # Giá đóng SELL là ask
                current_profit_pips = (entry_price - current_price) / pip_value
            
            # Tìm level phù hợp (từ cao xuống thấp)
            new_sl = None
            matched_level = None
            for trigger_pips, sl_pips_level in sorted_levels:
                if current_profit_pips >= trigger_pips:
                    # Tính SL mới
                    if pos_type == 'BUY':
                        new_sl = entry_price + sl_pips_level * pip_value
                    else:  # SELL
                        new_sl = entry_price - sl_pips_level * pip_value
                    matched_level = (trigger_pips, sl_pips_level)
                    break  # Dùng mức cao nhất phù hợp
            
            result_info = {
                'ticket': ticket,
                'type': pos_type,
                'profit_pips': round(current_profit_pips, 1),
                'current_sl': current_sl,
                'modified': False
            }
            
            if new_sl is not None:
                # Kiểm tra: chỉ modify nếu SL mới TỐT HƠN SL hiện tại
                should_modify = False
                if pos_type == 'BUY':
                    # BUY: SL tốt hơn = SL cao hơn (gần giá hơn)
                    should_modify = new_sl > current_sl
                else:  # SELL
                    # SELL: SL tốt hơn = SL thấp hơn (gần giá hơn)
                    should_modify = new_sl < current_sl
                
                if should_modify:
                    # Round SL theo digits của symbol
                    new_sl = round(new_sl, symbol_info.digits)
                    success, msg = self.modify_position(ticket, sl=new_sl)
                    
                    if success:
                        result_info['modified'] = True
                        result_info['new_sl'] = new_sl
                        result_info['level'] = matched_level
                        
                        # Log vào trade_log
                        self.trade_log.append({
                            'time': datetime.now(),
                            'action': 'TRAILING_SL',
                            'ticket': ticket,
                            'old_sl': current_sl,
                            'new_sl': new_sl,
                            'profit_pips': round(current_profit_pips, 1),
                            'level': f"{matched_level[0]}→{matched_level[1]}"
                        })
                    else:
                        result_info['error'] = msg
            
            results.append(result_info)
        
        return results



if __name__ == "__main__":
    # Test
    trader = MT5Trader()
    success, msg = trader.connect()
    print(msg)
    
    if success:
        print("\n--- Account Info ---")
        print(trader.get_account_info())
        
        print("\n--- Symbol Info ---")
        print(trader.get_symbol_info())
        
        print("\n--- M5 Data (last 5) ---")
        df = trader.get_realtime_data("M5", 5)
        if df is not None:
            print(df)
        
        trader.disconnect()
