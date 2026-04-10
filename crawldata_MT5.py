
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, timezone
import os

from config import (
    DEFAULT_SYMBOL, ALTERNATIVE_SYMBOLS,
    CRAWL_DEFAULT_LOOKBACK_DAYS,
)

TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M2":  mt5.TIMEFRAME_M2,
    "M3":  mt5.TIMEFRAME_M3,
    "M4":  mt5.TIMEFRAME_M4,
    "M5":  mt5.TIMEFRAME_M5,
    "M6":  mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,

    "H1":  mt5.TIMEFRAME_H1,
    "H2":  mt5.TIMEFRAME_H2,
    "H3":  mt5.TIMEFRAME_H3,
    "H4":  mt5.TIMEFRAME_H4,
    "H6":  mt5.TIMEFRAME_H6,
    "H8":  mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,

    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

def download_xauusd_data(symbol: str, start_date: datetime, end_date: datetime, timeframe: str = "M5"):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    if timeframe not in TIMEFRAME_MAP:
        print(f"Timeframe không hợp lệ: {timeframe}. Các timeframe hợp lệ: {list(TIMEFRAME_MAP.keys())}")
        return None
    
    mt5_timeframe = TIMEFRAME_MAP[timeframe]
    
    if not mt5.initialize():
        print(f"Lỗi khởi tạo MT5: {mt5.last_error()}")
        return None
    print(f"Kết nối MT5 thành công!, Phiên bản MT5: {mt5.version()}")
    if end_date is None:
        end_date = datetime.now()
        print(datetime.now())
    if start_date is None:
        start_date = end_date - timedelta(days=CRAWL_DEFAULT_LOOKBACK_DAYS) 
    
    print(f"\nTải dữ liệu từ: {start_date.strftime('%Y-%m-%d')} đến {end_date.strftime('%Y-%m-%d')}")
    if symbol is None:
        symbol = DEFAULT_SYMBOL
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        alternative_symbols = ALTERNATIVE_SYMBOLS
        for alt_symbol in alternative_symbols:
            symbol_info = mt5.symbol_info(alt_symbol)
            if symbol_info is not None:
                symbol = alt_symbol
                print(f"Sử dụng symbol: {symbol}")
                break
        
        if symbol_info is None:
            print(f"Không tìm thấy symbol XAUUSD. Các symbol có sẵn:")
            symbols = mt5.symbols_get()
            gold_symbols = [s.name for s in symbols if 'XAU' in s.name or 'GOLD' in s.name.upper()]
            print(gold_symbols[:10] if gold_symbols else "Không tìm thấy symbol vàng")
            mt5.shutdown()
            return None
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"Không thể chọn symbol {symbol}")
            mt5.shutdown()
            return None
    print(f"\nĐang tải dữ liệu {symbol} {timeframe}...")
    print(f"Debug: copy_rates_range(symbol='{symbol}', timeframe={mt5_timeframe}, start={start_date}, end={end_date})")
    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
         print(f"Failed to select symbol {symbol}")
         
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"Không có dữ liệu. Lỗi: {mt5.last_error()}")
        mt5.shutdown()
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'TickVolume', 'Spread', 'RealVolume']
    df = df.sort_values('Time').reset_index(drop=True)
    print(f"\n✓ Đã tải thành công {len(df)} nến {timeframe}")
    print(f"  - Từ: {df['Time'].iloc[0]}")
    print(f"  - Đến: {df['Time'].iloc[-1]}")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✓ Đã lưu file: {output_file}")
    print("\n--- Thống kê dữ liệu ---")
    print(f"Số lượng nến: {len(df)}")
    print(f"Giá cao nhất: {df['High'].max():.2f}")
    print(f"Giá thấp nhất: {df['Low'].min():.2f}")
    print(f"Giá mở đầu kỳ: {df['Open'].iloc[0]:.2f}")
    print(f"Giá đóng cửa: {df['Close'].iloc[-1]:.2f}")
    print("\n--- 5 dòng đầu ---")
    print(df.head().to_string(index=False))
    print("\n--- 5 dòng cuối ---")
    print(df.tail().to_string(index=False))
    mt5.shutdown()
    print("\n✓ Đã ngắt kết nối MT5")
    return df

if __name__ == "__main__": 
    df = download_xauusd_data(symbol=DEFAULT_SYMBOL, start_date="2016-01-25", end_date="2026-01-25", timeframe="M5")


