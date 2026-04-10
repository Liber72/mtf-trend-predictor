"""
Data Processor Module
Xử lý dữ liệu và tính toán các chỉ số kỹ thuật cho mô hình LSTM
"""

import pandas as pd
import numpy as np
import ta
from typing import Tuple, Optional
import pickle
import os

from config import (
    LOOKBACK, STEP, SCALER_WINDOW, TRAIN_RATIO,
    ADX_WINDOW, MFI_WINDOW, MFI_DEFAULT_VALUE, RSI_WINDOW,
    SMA_SHORT_WINDOW, SMA_LONG_WINDOW, CCI_WINDOW,
    CLOSE_POSITION_EPSILON, FEATURE_COLUMNS,
)


class DataProcessor:
    """
    Xử lý dữ liệu nến và tính các chỉ số kỹ thuật
    Sử dụng Sliding Window MinMaxScaler để chuẩn hóa dữ liệu
    """
    
    def __init__(self, lookback: int = LOOKBACK, step: int = STEP, scaler_window: int = SCALER_WINDOW):
        """
        Args:
            lookback: Số nến nhìn lại để dự đoán (mặc định LOOKBACK)
            step: Bước nhảy giữa các sequence (mặc định STEP)
            scaler_window: Kích thước cửa sổ trượt cho MinMaxScaler (mặc định SCALER_WINDOW)
        """
        self.lookback = lookback
        self.step = step
        self.scaler_window = scaler_window
        self.feature_columns = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load dữ liệu từ file CSV
        
        Args:
            file_path: Đường dẫn đến file CSV
            
        Returns:
            DataFrame chứa dữ liệu nến
        """
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm các chỉ số kỹ thuật vào DataFrame
        
        Các chỉ số: ADX, MFI, RSI, SMA, CCI
        
        Args:
            df: DataFrame chứa OHLCV data
            
        Returns:
            DataFrame với các chỉ số kỹ thuật đã thêm
        """
        df = df.copy()
        
        # ADX - Average Directional Index (xu hướng mạnh/yếu)
        adx_indicator = ta.trend.ADXIndicator(
            high=df['High'], 
            low=df['Low'], 
            close=df['Close'], 
            window=ADX_WINDOW
        )
        df['ADX'] = adx_indicator.adx()
        df['ADX_pos'] = adx_indicator.adx_pos()  # +DI
        df['ADX_neg'] = adx_indicator.adx_neg()  # -DI
        
        # MFI - Money Flow Index (dòng tiền)
        # Sử dụng TickVolume thay vì Volume
        if 'TickVolume' in df.columns:
            df['MFI'] = ta.volume.MFIIndicator(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                volume=df['TickVolume'],
                window=MFI_WINDOW
            ).money_flow_index()
        else:
            df['MFI'] = MFI_DEFAULT_VALUE  # Giá trị mặc định nếu không có volume
        
        # RSI - Relative Strength Index (quá mua/quá bán)
        df['RSI'] = ta.momentum.RSIIndicator(
            close=df['Close'], 
            window=RSI_WINDOW
        ).rsi()
        
        # SMA - Simple Moving Average
        df['SMA_10'] = ta.trend.SMAIndicator(
            close=df['Close'], 
            window=SMA_SHORT_WINDOW
        ).sma_indicator()
        
        df['SMA_20'] = ta.trend.SMAIndicator(
            close=df['Close'], 
            window=SMA_LONG_WINDOW
        ).sma_indicator()
        
        # CCI - Commodity Channel Index
        df['CCI'] = ta.trend.CCIIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=CCI_WINDOW
        ).cci()
        
        # Thêm các features bổ sung
        # Price change
        df['Price_Change'] = df['Close'].pct_change()
        
        # High-Low range
        df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Close position trong range
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + CLOSE_POSITION_EPSILON)
        
        return df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo label cho dự đoán xu hướng
        
        Label = 1 nếu Close[t+1] > Close[t] (tăng)
        Label = 0 nếu Close[t+1] <= Close[t] (giảm)
        
        Args:
            df: DataFrame chứa dữ liệu
            
        Returns:
            DataFrame với cột 'Label' đã thêm
        """
        df = df.copy()
        df['Label'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị các features cho training
        
        Args:
            df: DataFrame đã có các chỉ số kỹ thuật
            
        Returns:
            DataFrame chỉ chứa các features cần thiết
        """
        self.feature_columns = list(FEATURE_COLUMNS)
        df_clean = df.dropna(subset=self.feature_columns + ['Label'])
        
        return df_clean
    def normalize_sliding_window(self, features: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa dữ liệu bằng Sliding Window MinMaxScaler
        
        Mỗi row tại vị trí i được scale dựa trên window [i-scaler_window+1 : i+1].
        Sử dụng vectorized numpy operations để tối ưu tốc độ.
        
        Args:
            features: Array (n_samples, n_features) chứa raw features
            
        Returns:
            Array dữ liệu đã chuẩn hóa (cùng shape)
        """
        n_samples, n_features = features.shape
        scaled = np.zeros_like(features, dtype=np.float32)
        
        for i in range(n_samples):
            # Window: từ max(0, i - scaler_window + 1) đến i + 1
            start = max(0, i - self.scaler_window + 1)
            window = features[start:i + 1]
            
            # Tính min/max trên window (vectorized)
            w_min = window.min(axis=0)
            w_max = window.max(axis=0)
            w_range = w_max - w_min
            
            # Tránh chia cho 0
            w_range[w_range == 0] = 1.0
            
            # Scale row hiện tại theo min/max của window
            scaled[i] = (features[i] - w_min) / w_range
        
        return scaled
    
    def create_sequences(
        self, 
        features: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        
        for i in range(0, len(features) - self.lookback, self.step):
            X.append(features[i:i + self.lookback].astype(np.float32))
            y.append(labels[i + self.lookback - 1])  
        
        return np.array(X), np.array(y)
    
    def process_data(
        self, 
        file_path: str,
        train_ratio: float = TRAIN_RATIO,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pipeline xử lý dữ liệu đầy đủ với Sliding Window MinMaxScaler
        
        Args:
            file_path: Đường dẫn file CSV
            train_ratio: Tỉ lệ dữ liệu train (mặc định 0.8)
            fit_scaler: Không sử dụng nữa (giữ lại để tương thích)
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        df = self.load_data(file_path)
        print(f"✓ Loaded {len(df)} rows from {file_path}")
        df = self.add_technical_indicators(df)
        print(f"✓ Added technical indicators")
        df = self.create_labels(df)
        df = self.prepare_features(df)
        print(f"✓ Prepared {len(df)} samples after removing NaN")
        val_size = int(len(df) * (1 - train_ratio))
        if train_ratio >= 1.0:
            val_size = 0
        df_test = df.iloc[:val_size] if val_size > 0 else df.iloc[:0]
        df_train = df.iloc[val_size:]
        
        print(f"📊 Data split: Val={len(df_test)} nến (đầu) | Train={len(df_train)} nến (sau)")
        print(f"📐 Sliding Window Scaler: window={self.scaler_window}")
        
        # Lưu df_test để export kết quả validation
        self.df_test = df_test.copy() if len(df_test) > 0 else None
        self.df_test_start_idx = 0  
        
        # === SLIDING WINDOW NORMALIZE ===
        # Normalize toàn bộ data liên tục (val + train) để window có đủ context
        # Thứ tự trong df: [val_data (đầu)] [train_data (sau)]
        all_features = df[self.feature_columns].values
        print(f"⏳ Đang normalize {len(all_features)} samples với sliding window={self.scaler_window}...")
        all_scaled = self.normalize_sliding_window(all_features)
        print(f"✓ Normalize hoàn tất")
        
        # Tách lại train/test từ dữ liệu đã scale
        scaled_test = all_scaled[:val_size] if val_size > 0 else np.array([]).reshape(0, all_scaled.shape[1])
        scaled_train = all_scaled[val_size:]
        
        labels_train = df_train['Label'].values
        
        # Create train sequences
        X_train, y_train = self.create_sequences(scaled_train, labels_train)
        
        # Handle test/val data (có thể rỗng nếu train_ratio = 1.0)
        if len(df_test) > self.lookback:
            labels_test = df_test['Label'].values
            X_test, y_test = self.create_sequences(scaled_test, labels_test)
        else:
            # Không có test data (train 100%)
            X_test = np.array([]).reshape(0, self.lookback, X_train.shape[2])
            y_test = np.array([])
            print(f"⚠️ Train ratio = 100% - không có validation set")
        
        print(f"✓ Created sequences:")
        print(f"  - Train: {X_train.shape[0]} samples")
        print(f"  - Val/Test: {X_test.shape[0]} samples")
        print(f"  - Features: {X_train.shape[2]}")
        print(f"  - Lookback: {self.lookback}")
        print(f"  - Scaler Window: {self.scaler_window}")
        
        return X_train, X_test, y_train, y_test
    
    def save_scaler(self, file_path: str):
        """
        Lưu cấu hình scaler vào file (scaler_window + feature_columns)
        
        Args:
            file_path: Đường dẫn file pickle
        """
        with open(file_path, 'wb') as f:
            pickle.dump({
                'scaler_window': self.scaler_window,
                'feature_columns': self.feature_columns
            }, f)
        print(f"✓ Saved scaler config (window={self.scaler_window}) to {file_path}")
    
    def load_scaler(self, file_path: str):
        """
        Load cấu hình scaler từ file
        
        Args:
            file_path: Đường dẫn file pickle
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            # Hỗ trợ cả format cũ (có 'scaler') và format mới (có 'scaler_window')
            if 'scaler_window' in data:
                self.scaler_window = data['scaler_window']
            else:
                print(f"⚠️ File scaler cũ (global), sử dụng scaler_window={self.scaler_window}")
            self.feature_columns = data['feature_columns']
        print(f"✓ Loaded scaler config (window={self.scaler_window}) from {file_path}")
    
    def get_latest_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """
        Lấy sequence mới nhất để dự đoán, sử dụng sliding window scaler
        
        Cần ít nhất scaler_window nến để scale chính xác.
        Nếu không đủ scaler_window, sử dụng tất cả nến có sẵn.
        
        Args:
            df: DataFrame dữ liệu mới nhất (nên có >= scaler_window nến)
            
        Returns:
            Array shape (1, lookback, features) cho prediction
        """
        # Add indicators
        df = self.add_technical_indicators(df)
        df = df.dropna()
        
        min_required = self.lookback
        if len(df) < min_required:
            raise ValueError(
                f"Không đủ dữ liệu. Cần ít nhất {min_required} nến, hiện có {len(df)}. "
                f"(Khuyến nghị >= {self.scaler_window} nến để scale chính xác)"
            )
        
        # Lấy tối đa scaler_window nến cuối để có đủ context cho sliding window
        n_context = min(len(df), self.scaler_window)
        df_context = df.iloc[-n_context:]
        
        # Normalize toàn bộ context window
        features_raw = df_context[self.feature_columns].values
        features_scaled = self.normalize_sliding_window(features_raw)
        
        # Lấy lookback nến cuối từ dữ liệu đã scale
        features = features_scaled[-self.lookback:]
        
        # Reshape cho LSTM
        return features.reshape(1, self.lookback, -1)


if __name__ == "__main__":
    # Test module
    import os
    
    # Tìm file CSV trong thư mục
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    
    if csv_files:
        print(f"Tìm thấy {len(csv_files)} file CSV:")
        for f in csv_files:
            print(f"  - {f}")
        
        test_file = os.path.join(current_dir, csv_files[0])
        
        processor = DataProcessor(lookback=LOOKBACK, step=STEP, scaler_window=SCALER_WINDOW)
        X_train, X_test, y_train, y_test = processor.process_data(test_file)
        
        print(f"\n✓ Test thành công!")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train distribution: {np.bincount(y_train.astype(int))}")
    else:
        print("Không tìm thấy file CSV. Vui lòng chạy crawldata_MT5.py trước.")
