"""
Trainer Module
Huấn luyện và quản lý các mô hình LSTM cho H1 và M5
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

# Cấu hình GPU memory growth trước khi import TensorFlow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU memory growth enabled cho {len(gpus)} GPU(s)")

from config import (
    LOOKBACK, SCALER_WINDOW, EPOCHS, BATCH_SIZE,
    TRAIN_RATIO, MODELS_DIR,
    MODEL_MODE_DUAL, MODEL_MODE_SINGLE_M5, DEFAULT_MODEL_MODE,
)
from data_processor import DataProcessor
from lstm_model import LSTMModel


class Trainer:
    """
    Quản lý việc huấn luyện mô hình cho các timeframe khác nhau
    """
    
    def __init__(self, models_dir: str = MODELS_DIR, model_mode: str = DEFAULT_MODEL_MODE):
        """
        Args:
            models_dir: Thư mục lưu models
            model_mode: Chế độ model ("dual" hoặc "single_m5")
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, models_dir)
        self.model_mode = model_mode
        
        # Tạo thư mục nếu chưa tồn tại (fix Windows compatibility)
        if not os.path.exists(self.models_dir):
            try:
                os.makedirs(self.models_dir)
            except OSError:
                pass  # Thư mục đã tồn tại hoặc không thể tạo
        
        # Các processors và models
        self.h1_processor = None
        self.m5_processor = None
        self.h1_model = None
        self.m5_model = None
        
        # Metrics
        self.h1_metrics = None
        self.m5_metrics = None
    
    def find_data_file(self, timeframe: str) -> Optional[str]:
        """
        Tìm file dữ liệu cho timeframe
        
        Args:
            timeframe: "H1" hoặc "M5"
            
        Returns:
            Đường dẫn file hoặc None
        """
        csv_files = [f for f in os.listdir(self.base_dir) if f.endswith('.csv')]
        
        for f in csv_files:
            if f"_{timeframe}_" in f:
                return os.path.join(self.base_dir, f)
        
        return None
    
    def train_model(
        self,
        timeframe: str,
        data_file: Optional[str] = None,
        lookback: int = LOOKBACK,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        train_ratio: float = TRAIN_RATIO
    ) -> Tuple[LSTMModel, Dict]:
        """
        Huấn luyện một mô hình
        
        Args:
            timeframe: "H1" hoặc "M5"
            data_file: Đường dẫn file dữ liệu (tùy chọn)
            lookback: Số nến lookback
            epochs: Số epochs
            batch_size: Batch size
            train_ratio: Tỉ lệ dữ liệu train (1.0 = train 100%)
            
        Returns:
            Tuple (model, metrics)
        """
        print(f"\n{'='*60}")
        print(f"HUẤN LUYỆN MÔ HÌNH {timeframe}")
        print(f"Train ratio: {train_ratio*100:.0f}%")
        print(f"{'='*60}")
        
        # Tìm file dữ liệu
        if data_file is None:
            data_file = self.find_data_file(timeframe)
        
        if data_file is None or not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu cho {timeframe}. "
                f"Vui lòng chạy crawldata_MT5.py với timeframe={timeframe}"
            )
        
        print(f"✓ Sử dụng dữ liệu: {data_file}")
        
        # Xử lý dữ liệu
        processor = DataProcessor(lookback=lookback, step=1, scaler_window=SCALER_WINDOW)
        X_train, X_test, y_train, y_test = processor.process_data(data_file, train_ratio=train_ratio)
        
        # Lưu scaler
        scaler_path = os.path.join(self.models_dir, f"{timeframe.lower()}_scaler.pkl")
        processor.save_scaler(scaler_path)
        
        # Build model
        n_features = X_train.shape[2]
        model = LSTMModel(lookback=lookback, n_features=n_features)
        model.build_model()
        
        # Đường dẫn lưu model
        model_path = os.path.join(self.models_dir, f"{timeframe.lower()}_model.keras")
        
        # Train
        train_results = model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=epochs,
            batch_size=batch_size,
            model_path=model_path
        )
        
        # Evaluate (chỉ khi có test data)
        if len(X_test) > 0:
            metrics = model.evaluate(X_test, y_test)
            metrics.update(train_results)
            
            print(f"\n📊 Kết quả đánh giá {timeframe}:")
            print(f"  - Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall: {metrics['recall']:.4f}")
            print(f"  - F1 Score: {metrics['f1_score']:.4f}")
            
            # Lưu kết quả validation ra CSV
            self.save_validation_results(
                timeframe=timeframe,
                X_test=X_test,
                y_test=y_test,
                processor=processor,
                model=model
            )
        else:
            # Không có test data (train 100%)
            metrics = train_results
            metrics['accuracy'] = 0
            metrics['precision'] = 0
            metrics['recall'] = 0
            metrics['f1_score'] = 0
            print(f"\n⚠️ Train 100% - không có validation metrics")
        
        # Lưu model
        model.save(model_path)
        
        # Lưu vào instance
        if timeframe == "H1":
            self.h1_processor = processor
            self.h1_model = model
            self.h1_metrics = metrics
        else:
            self.m5_processor = processor
            self.m5_model = model
            self.m5_metrics = metrics
        
        return model, metrics
    
    def save_validation_results(
        self,
        timeframe: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        processor: 'DataProcessor',
        model: 'LSTMModel',
        output_dir: Optional[str] = None
    ) -> str:
        """
        Lưu kết quả dự đoán trên validation set ra file CSV
        
        Args:
            timeframe: "H1" hoặc "M5"
            X_test: Dữ liệu test
            y_test: Labels thực tế
            processor: DataProcessor đã xử lý dữ liệu
            model: Model đã train
            output_dir: Thư mục output (mặc định = models_dir)
            
        Returns:
            Đường dẫn file CSV đã lưu
        """
        if output_dir is None:
            output_dir = self.models_dir
        
        # Dự đoán trên validation set
        predictions = model.model.predict(X_test, verbose=0)
        pred_labels = (predictions > 0.5).astype(int).flatten()
        pred_probs = predictions.flatten()
        
        # Lấy thông tin từ df_test
        if hasattr(processor, 'df_test') and processor.df_test is not None:
            # Lấy các dòng tương ứng với sequences
            # Mỗi sequence lấy label của nến cuối (lookback - 1)
            start_offset = processor.lookback - 1
            df_val = processor.df_test.iloc[start_offset:start_offset + len(y_test)].copy()
            df_val = df_val.reset_index(drop=True)
        else:
            # Tạo DataFrame cơ bản nếu không có df_test
            df_val = pd.DataFrame()
        
        # Tạo DataFrame kết quả
        results_df = pd.DataFrame({
            'Time': df_val['Time'].values if 'Time' in df_val.columns else range(len(y_test)),
            'Close': df_val['Close'].values if 'Close' in df_val.columns else 0,
            'Actual_Label': y_test,
            'Actual_Direction': ['UP' if y == 1 else 'DOWN' for y in y_test],
            'Predicted_Label': pred_labels,
            'Predicted_Direction': ['UP' if p == 1 else 'DOWN' for p in pred_labels],
            'Predicted_Prob': pred_probs,
            'Confidence': [p if p > 0.5 else 1 - p for p in pred_probs],
            'Correct': (pred_labels == y_test).astype(int)
        })
        
        # Lưu file
        output_file = os.path.join(output_dir, f"{timeframe.lower()}_validation_results.csv")
        results_df.to_csv(output_file, index=False)
        
        # Thống kê nhanh
        accuracy = (pred_labels == y_test).mean()
        print(f"\n📄 Đã lưu kết quả validation: {output_file}")
        print(f"   - Tổng mẫu: {len(y_test)}")
        print(f"   - Đúng: {(pred_labels == y_test).sum()} ({accuracy*100:.1f}%)")
        print(f"   - Sai: {(pred_labels != y_test).sum()} ({(1-accuracy)*100:.1f}%)")
        
        return output_file
    
    def train_both(
        self,
        h1_file: Optional[str] = None,
        m5_file: Optional[str] = None,
        lookback: int = LOOKBACK,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE
    ) -> Dict:
        """
        Huấn luyện cả 2 mô hình H1 và M5
        
        Returns:
            Dict chứa metrics của cả 2 mô hình
        """
        print("\n" + "="*60)
        print("BẮT ĐẦU HUẤN LUYỆN CẢ 2 MÔ HÌNH")
        print("="*60)
        
        results = {}
        
        # Train H1
        try:
            _, h1_metrics = self.train_model(
                "H1", h1_file, lookback, epochs, batch_size
            )
            results['H1'] = h1_metrics
        except Exception as e:
            print(f"❌ Lỗi train H1: {e}")
            results['H1'] = {'error': str(e)}
        
        # Train M5
        try:
            _, m5_metrics = self.train_model(
                "M5", m5_file, lookback, epochs, batch_size
            )
            results['M5'] = m5_metrics
        except Exception as e:
            print(f"❌ Lỗi train M5: {e}")
            results['M5'] = {'error': str(e)}
        
        print("\n" + "="*60)
        print("HOÀN THÀNH HUẤN LUYỆN")
        print("="*60)
        
        if 'accuracy' in results.get('H1', {}):
            print(f"✓ H1 Accuracy: {results['H1']['accuracy']*100:.2f}%")
        if 'accuracy' in results.get('M5', {}):
            print(f"✓ M5 Accuracy: {results['M5']['accuracy']*100:.2f}%")
        
        return results
    
    def load_models(self) -> Tuple[bool, bool]:
        """
        Load các mô hình đã train
        
        Returns:
            Tuple (h1_loaded, m5_loaded)
        """
        h1_loaded = False
        m5_loaded = False
        
        # Load H1
        h1_model_path = os.path.join(self.models_dir, "h1_model.keras")
        h1_scaler_path = os.path.join(self.models_dir, "h1_scaler.pkl")
        
        if os.path.exists(h1_model_path) and os.path.exists(h1_scaler_path):
            self.h1_processor = DataProcessor(lookback=LOOKBACK, scaler_window=SCALER_WINDOW)
            self.h1_processor.load_scaler(h1_scaler_path)
            self.h1_model = LSTMModel()
            self.h1_model.load(h1_model_path)
            h1_loaded = True
            print("✓ Loaded H1 model")
        
        # Load M5
        m5_model_path = os.path.join(self.models_dir, "m5_model.keras")
        m5_scaler_path = os.path.join(self.models_dir, "m5_scaler.pkl")
        
        if os.path.exists(m5_model_path) and os.path.exists(m5_scaler_path):
            self.m5_processor = DataProcessor(lookback=LOOKBACK, scaler_window=SCALER_WINDOW)
            self.m5_processor.load_scaler(m5_scaler_path)
            self.m5_model = LSTMModel()
            self.m5_model.load(m5_model_path)
            m5_loaded = True
            print("✓ Loaded M5 model")
        
        return h1_loaded, m5_loaded
    
    def predict(
        self,
        h1_data: Optional[np.ndarray] = None,
        m5_data: Optional[np.ndarray] = None,
        model_mode: Optional[str] = None
    ) -> Dict:
        """
        Dự đoán sử dụng mô hình theo chế độ đã chọn
        
        Args:
            h1_data: DataFrame dữ liệu H1 (cần ít nhất 48 nến)
            m5_data: DataFrame dữ liệu M5 (cần ít nhất 48 nến)
            model_mode: Chế độ model (None = dùng self.model_mode)
            
        Returns:
            Dict kết quả dự đoán
        """
        mode = model_mode or self.model_mode
        results = {}
        
        # Dự đoán H1 (chỉ khi mode dual)
        if mode == MODEL_MODE_DUAL and self.h1_model and h1_data is not None:
            try:
                X_h1 = self.h1_processor.get_latest_sequence(h1_data)
                direction, prob = self.h1_model.predict_single(X_h1)
                results['H1'] = {
                    'direction': direction,
                    'probability': prob
                }
            except Exception as e:
                results['H1'] = {'error': str(e)}
        
        # Dự đoán M5
        if self.m5_model and m5_data is not None:
            try:
                X_m5 = self.m5_processor.get_latest_sequence(m5_data)
                direction, prob = self.m5_model.predict_single(X_m5)
                results['M5'] = {
                    'direction': direction,
                    'probability': prob
                }
            except Exception as e:
                results['M5'] = {'error': str(e)}
        
        # Kết hợp dự đoán theo mode
        if mode == MODEL_MODE_SINGLE_M5:
            # Single M5: signal trực tiếp từ M5
            if 'M5' in results and 'direction' in results['M5']:
                m5_dir = results['M5']['direction']
                m5_prob = results['M5']['probability']
                results['combined'] = {
                    'signal': 'BUY' if m5_dir == 'UP' else 'SELL',
                    'confidence': m5_prob
                }
        else:
            # Dual: cần cả 2 model đồng thuận
            if 'H1' in results and 'M5' in results:
                h1_dir = results['H1'].get('direction')
                m5_dir = results['M5'].get('direction')
                
                if h1_dir == 'UP' and m5_dir == 'UP':
                    results['combined'] = {
                        'signal': 'BUY',
                        'confidence': (results['H1']['probability'] + results['M5']['probability']) / 2
                    }
                elif h1_dir == 'DOWN' and m5_dir == 'DOWN':
                    results['combined'] = {
                        'signal': 'SELL',
                        'confidence': (results['H1']['probability'] + results['M5']['probability']) / 2
                    }
                else:
                    results['combined'] = {
                        'signal': 'WAIT',
                        'reason': 'Hai mô hình không cùng xu hướng'
                    }
        
        return results


def main():
    """Main function với argument parsing"""
    parser = argparse.ArgumentParser(description='Train LSTM models for H1/M5')
    parser.add_argument(
        '--timeframe', '-t',
        choices=['H1', 'M5', 'both'],
        default='both',
        help='Timeframe để train (H1, M5, hoặc both)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=EPOCHS,
        help='Số epochs (mặc định: 100)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=BATCH_SIZE,
        help='Batch size (mặc định: 32)'
    )
    parser.add_argument(
        '--lookback', '-l',
        type=int,
        default=LOOKBACK,
        help='Lookback period (mặc định: 48)'
    )
    parser.add_argument(
        '--data-file', '-d',
        type=str,
        default=None,
        help='Đường dẫn file dữ liệu CSV (tùy chọn)'
    )
    
    args = parser.parse_args()
    
    trainer = Trainer()
    
    if args.timeframe == 'both':
        trainer.train_both(
            lookback=args.lookback,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        trainer.train_model(
            args.timeframe,
            data_file=args.data_file,
            lookback=args.lookback,
            epochs=args.epochs,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
