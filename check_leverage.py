"""
Kiểm tra đòn bẩy (leverage) của tài khoản MT5.
Chạy: python check_leverage.py
"""

import MetaTrader5 as mt5


def check_leverage():
    # Khởi tạo kết nối MT5
    if not mt5.initialize():
        print(f"❌ Lỗi khởi tạo MT5: {mt5.last_error()}")
        return

    print(f"✅ Kết nối MT5 thành công! Phiên bản: {mt5.version()}")
    print("=" * 50)

    # Lấy thông tin tài khoản
    account_info = mt5.account_info()
    if account_info is None:
        print(f"❌ Không lấy được thông tin tài khoản: {mt5.last_error()}")
        mt5.shutdown()
        return

    # Hiển thị thông tin đòn bẩy và tài khoản
    print("📊 THÔNG TIN TÀI KHOẢN")
    print("=" * 50)
    print(f"  🔑 Login:          {account_info.login}")
    print(f"  👤 Tên:            {account_info.name}")
    print(f"  🏦 Server:         {account_info.server}")
    print(f"  💰 Loại tài khoản: {'Demo' if account_info.trade_mode == 0 else 'Contest' if account_info.trade_mode == 1 else 'Real'}")
    print(f"  💵 Tiền tệ:        {account_info.currency}")
    print()
    print("📈 THÔNG TIN ĐÒN BẨY")
    print("=" * 50)
    print(f"  ⚡ Đòn bẩy:        1:{account_info.leverage}")
    print()
    print("💰 THÔNG TIN SỐ DƯ")
    print("=" * 50)
    print(f"  💵 Số dư (Balance):    {account_info.balance:.2f} {account_info.currency}")
    print(f"  📊 Equity:             {account_info.equity:.2f} {account_info.currency}")
    print(f"  🔒 Margin đã dùng:     {account_info.margin:.2f} {account_info.currency}")
    print(f"  🔓 Margin còn lại:     {account_info.margin_free:.2f} {account_info.currency}")
    print(f"  📈 Margin Level:       {account_info.margin_level:.2f}%" if account_info.margin_level else "  📈 Margin Level:       N/A (không có lệnh)")
    print(f"  📊 Profit hiện tại:    {account_info.profit:.2f} {account_info.currency}")
    print()

    # Giải thích ý nghĩa đòn bẩy
    leverage = account_info.leverage
    print("📝 Ý NGHĨA ĐÒN BẨY")
    print("=" * 50)
    print(f"  Với đòn bẩy 1:{leverage}, bạn chỉ cần {100/leverage:.2f}% giá trị")
    print(f"  giao dịch làm ký quỹ (margin).")
    print(f"  Ví dụ: Để mở lệnh 1 lot XAUUSD (~$100,000),")
    print(f"         bạn chỉ cần ký quỹ ~${100000/leverage:,.2f}")
    print()

    # Ngắt kết nối
    mt5.shutdown()
    print("✅ Đã ngắt kết nối MT5")


if __name__ == "__main__":
    check_leverage()
