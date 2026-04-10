"""
LSTM Model Module
Xây dựng và huấn luyện mô hình LSTM cho dự đoán xu hướng giá
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional, Dict
import os
import h5py

from config import (
    LOOKBACK, N_FEATURES, LSTM_UNITS, DENSE_UNITS,
    DROPOUT_RATE, LEARNING_RATE, PREDICTION_THRESHOLD,
    EPOCHS, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE, REDUCE_LR_MIN,
)

class LSTMModel:
    """
    Mô hình LSTM cho dự đoán xu hướng giá
    """
    
    def __init__(
        self, 
        lookback: int = LOOKBACK, 
        n_features: int = N_FEATURES,
        lstm_units: Tuple[int, int] = LSTM_UNITS,
        dropout_rate: float = DROPOUT_RATE,
        learning_rate: float = LEARNING_RATE
    ):
        self.lookback = lookback
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self) -> Sequential:
        model = Sequential([

            LSTM(
                units=self.lstm_units[0],
                return_sequences=True,
                input_shape=(self.lookback, self.n_features)
            ),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Second LSTM layer
            LSTM(units=self.lstm_units[1], return_sequences=False),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(DENSE_UNITS, activation='relu'),
            Dropout(self.dropout_rate / 2),
            
            # Output layer - Binary classification
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_callbacks(self, model_path: Optional[str] = None) -> list:
        """
        Tạo callbacks cho training
        
        Args:
            model_path: Đường dẫn lưu model (nếu có)
            
        Returns:
            List các callbacks
        """
        callbacks = [
            # Early stopping nếu không improve
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Giảm learning rate khi plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=REDUCE_LR_MIN,
                verbose=1
            )
        ]
        
        # Checkpoint nếu có path
        if model_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        return callbacks
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        model_path: Optional[str] = None
    ) -> Dict:
        """
        Huấn luyện mô hình
        
        Args:
            X_train: Dữ liệu training (samples, lookback, features)
            y_train: Labels training
            X_val: Dữ liệu validation
            y_val: Labels validation
            epochs: Số epochs (mặc định 100)
            batch_size: Batch size (mặc định 32)
            model_path: Đường dẫn lưu model
            
        Returns:
            Dict kết quả training
        """
        if self.model is None:
            self.build_model()
        
        print(f"\n{'='*50}")
        print("Bắt đầu huấn luyện mô hình LSTM")
        print(f"{'='*50}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*50}\n")
        
        # Tạo tf.data.Dataset trên CPU để chỉ chuyển từng batch sang GPU (tránh OOM)
        with tf.device('/cpu:0'):
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Train
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=self.get_callbacks(model_path),
            verbose=1
        )
        
        # Evaluate (cũng dùng dataset để tránh OOM)
        train_loss, train_acc = self.model.evaluate(train_dataset, verbose=0)
        val_loss, val_acc = self.model.evaluate(val_dataset, verbose=0)
        
        results = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs_trained': len(self.history.history['loss'])
        }
        
        print(f"\n{'='*50}")
        print("Kết quả huấn luyện:")
        print(f"  - Train Accuracy: {train_acc:.4f}")
        print(f"  - Val Accuracy: {val_acc:.4f}")
        print(f"  - Epochs trained: {results['epochs_trained']}")
        print(f"{'='*50}\n")
        
        return results
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dự đoán xu hướng
        
        Args:
            X: Dữ liệu đầu vào (samples, lookback, features)
            
        Returns:
            Tuple (predictions, probabilities)
            - predictions: 1 = tăng, 0 = giảm
            - probabilities: Xác suất dự đoán
        """
        if self.model is None:
            raise ValueError("Model chưa được load hoặc train!")
        
        probabilities = self.model.predict(X, verbose=0)
        predictions = (probabilities > PREDICTION_THRESHOLD).astype(int)
        
        return predictions.flatten(), probabilities.flatten()
    
    def predict_single(self, X: np.ndarray) -> Tuple[str, float]:
        """
        Dự đoán cho một sequence duy nhất
        
        Args:
            X: Dữ liệu đầu vào shape (1, lookback, features)
            
        Returns:
            Tuple (direction, probability)
            - direction: "UP" hoặc "DOWN"
            - probability: Xác suất
        """
        _, probs = self.predict(X)
        prob = probs[0]
        
        if prob > PREDICTION_THRESHOLD:
            return "UP", prob
        else:
            return "DOWN", 1 - prob
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict:
        """
        Đánh giá mô hình trên test set
        
        Args:
            X_test: Dữ liệu test
            y_test: Labels test
            
        Returns:
            Dict metrics
        """
        predictions, probabilities = self.predict(X_test)
        
        # Accuracy
        accuracy = np.mean(predictions == y_test)
        
        # Confusion matrix values
        tp = np.sum((predictions == 1) & (y_test == 1))
        tn = np.sum((predictions == 0) & (y_test == 0))
        fp = np.sum((predictions == 1) & (y_test == 0))
        fn = np.sum((predictions == 0) & (y_test == 1))
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn)
            }
        }
    
    def save(self, file_path: str):
        """
        Lưu mô hình
        
        Args:
            file_path: Đường dẫn file .keras
        """
        if self.model is None:
            raise ValueError("Không có model để lưu!")
        
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        self.model.save(file_path)
        print(f"✓ Saved model to {file_path}")
    
    def load(self, file_path: str):
        """
        Load mô hình
        
        Args:
            file_path: Đường dẫn file .keras
        """
        self.model = load_model(file_path)
        print(f"✓ Loaded model from {file_path}")
    
    def summary(self):
        """In summary của model"""
        if self.model:
            self.model.summary()
        else:
            print("Model chưa được build!")


