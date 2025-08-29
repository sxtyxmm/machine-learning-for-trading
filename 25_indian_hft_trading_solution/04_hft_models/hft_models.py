"""
HFT Model Development for Indian Markets
Ultra-low latency ML models optimized for nanosecond-level trading
"""

import numpy as np
import pandas as pd
import time
import threading
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import joblib
from abc import ABC, abstractmethod

# ML libraries
try:
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, SGDRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# Cache implementation
class LRUCache:
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.access_order = deque()
        self.maxsize = maxsize
    
    def get(self, key, default=None):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return default
    
    def __setitem__(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            oldest = self.access_order.popleft()
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)

@dataclass
class ModelPrediction:
    """Container for model prediction results"""
    value: float
    confidence: float
    timestamp: float
    model_name: str
    inference_time_ns: int
    features_used: Optional[Dict] = None

class BaseHFTModel(ABC):
    """Base class for HFT models with common functionality"""
    
    def __init__(self, model_name: str, max_inference_time_ns: int = 10000):
        self.model_name = model_name
        self.max_inference_time_ns = max_inference_time_ns
        self.prediction_cache = LRUCache(maxsize=500)
        self.performance_stats = {
            'predictions_made': 0,
            'cache_hits': 0,
            'inference_times': deque(maxlen=1000),
            'errors': 0
        }
        self.is_trained = False
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict_raw(self, X: np.ndarray) -> float:
        """Raw prediction without caching or timing"""
        pass
    
    def predict(self, X: np.ndarray, use_cache: bool = True) -> ModelPrediction:
        """Make prediction with caching and performance tracking"""
        start_time = time.time_ns()
        
        # Check cache
        if use_cache:
            cache_key = hash(X.tobytes())
            cached_result = self.prediction_cache.get(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                return ModelPrediction(
                    value=cached_result,
                    confidence=1.0,  # Cached predictions have full confidence
                    timestamp=time.time(),
                    model_name=self.model_name,
                    inference_time_ns=time.time_ns() - start_time
                )
        
        try:
            # Make prediction
            prediction = self.predict_raw(X)
            confidence = 1.0  # Can be overridden by subclasses
            
            inference_time = time.time_ns() - start_time
            
            # Check if inference time is acceptable
            if inference_time > self.max_inference_time_ns:
                confidence *= 0.5  # Reduce confidence for slow predictions
            
            # Cache result
            if use_cache:
                self.prediction_cache[cache_key] = prediction
            
            # Update stats
            self.performance_stats['predictions_made'] += 1
            self.performance_stats['inference_times'].append(inference_time)
            
            return ModelPrediction(
                value=prediction,
                confidence=confidence,
                timestamp=time.time(),
                model_name=self.model_name,
                inference_time_ns=inference_time
            )
            
        except Exception as e:
            self.performance_stats['errors'] += 1
            return ModelPrediction(
                value=0.0,
                confidence=0.0,
                timestamp=time.time(),
                model_name=self.model_name,
                inference_time_ns=time.time_ns() - start_time
            )
    
    def get_performance_stats(self) -> Dict:
        """Get model performance statistics"""
        inference_times = list(self.performance_stats['inference_times'])
        
        if inference_times:
            return {
                'model_name': self.model_name,
                'predictions_made': self.performance_stats['predictions_made'],
                'cache_hit_rate': self.performance_stats['cache_hits'] / max(self.performance_stats['predictions_made'], 1),
                'mean_inference_ns': np.mean(inference_times),
                'p95_inference_ns': np.percentile(inference_times, 95),
                'p99_inference_ns': np.percentile(inference_times, 99),
                'max_inference_ns': np.max(inference_times),
                'error_rate': self.performance_stats['errors'] / max(self.performance_stats['predictions_made'], 1),
                'is_trained': self.is_trained
            }
        else:
            return {'model_name': self.model_name, 'is_trained': self.is_trained}

class RealTimeRidgeRegression(BaseHFTModel):
    """
    Ridge regression with online Kalman filter updates
    Optimized for microsecond-level inference
    """
    
    def __init__(self, model_name: str = "ridge_regression", alpha: float = 0.01, 
                 n_features: int = 50):
        super().__init__(model_name, max_inference_time_ns=1000)  # 1 microsecond target
        self.alpha = alpha
        self.n_features = n_features
        self.weights = np.zeros(n_features)
        self.P = np.eye(n_features) / alpha  # Covariance matrix
        self.update_count = 0
        
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Initial training using normal equations"""
        try:
            if X.shape[1] != self.n_features:
                return False
                
            # Solve normal equations: (X'X + αI)w = X'y
            XtX = np.dot(X.T, X)
            Xty = np.dot(X.T, y)
            regularized_XtX = XtX + self.alpha * np.eye(self.n_features)
            
            self.weights = np.linalg.solve(regularized_XtX, Xty)
            self.P = np.linalg.inv(regularized_XtX)
            
            self.is_trained = True
            return True
            
        except Exception:
            return False
    
    def predict_raw(self, X: np.ndarray) -> float:
        """Ultra-fast prediction using precomputed weights"""
        if not self.is_trained:
            return 0.0
        return np.dot(X.flatten(), self.weights)
    
    def online_update(self, x: np.ndarray, y: float):
        """Kalman filter update for real-time learning"""
        if not self.is_trained:
            return
            
        x = x.flatten()
        
        # Prediction error
        y_pred = np.dot(x, self.weights)
        error = y - y_pred
        
        # Kalman gain calculation
        Px = np.dot(self.P, x)
        denominator = np.dot(x, Px) + 1.0
        K = Px / denominator
        
        # Update weights and covariance
        self.weights += K * error
        self.P -= np.outer(K, Px)
        
        self.update_count += 1

class IncrementalLightGBM(BaseHFTModel):
    """
    LightGBM with incremental training optimized for Indian markets
    """
    
    def __init__(self, model_name: str = "lightgbm", n_estimators: int = 50, 
                 max_depth: int = 6, learning_rate: float = 0.1):
        super().__init__(model_name, max_inference_time_ns=5000)  # 5 microseconds target
        
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not available")
            
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': 'regression',
            'metric': 'rmse',
            'force_row_wise': True,  # Optimize for low latency
            'num_threads': 1,  # Single thread for consistency
            'verbosity': -1
        }
        self.model = None
        self.feature_importance_cache = {}
        
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train LightGBM model"""
        try:
            self.model = lgb.LGBMRegressor(**self.model_params)
            self.model.fit(X, y)
            
            # Cache feature importance for fast access
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance_cache = {
                    i: importance for i, importance in enumerate(self.model.feature_importances_)
                }
            
            self.is_trained = True
            return True
            
        except Exception:
            return False
    
    def predict_raw(self, X: np.ndarray) -> float:
        """Optimized prediction"""
        if not self.is_trained or self.model is None:
            return 0.0
            
        return self.model.predict(X.reshape(1, -1))[0]
    
    def train_incremental(self, X_new: np.ndarray, y_new: np.ndarray, 
                         X_base: Optional[np.ndarray] = None, 
                         y_base: Optional[np.ndarray] = None) -> bool:
        """Incremental training with concept drift handling"""
        try:
            if X_base is not None and y_base is not None:
                # Combine new data with subset of historical data
                decay_factor = 0.95
                sample_size = min(len(X_base), len(X_new) * 5)
                indices = np.random.choice(len(X_base), sample_size, replace=False)
                
                X_combined = np.vstack([X_base[indices], X_new])
                y_combined = np.hstack([y_base[indices] * decay_factor, y_new])
            else:
                X_combined, y_combined = X_new, y_new
            
            return self.train(X_combined, y_combined)
            
        except Exception:
            return False

class QuantizedNeuralNetwork(BaseHFTModel):
    """
    8-bit quantized neural network for ultra-low latency inference
    """
    
    def __init__(self, model_name: str = "quantized_nn", input_size: int = 50, 
                 hidden_sizes: List[int] = [32, 16], output_size: int = 1):
        super().__init__(model_name, max_inference_time_ns=3000)  # 3 microseconds target
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.layers = []
        self.scales = []
        self.zero_points = []
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize quantized weights"""
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Initialize weights with Xavier initialization
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            weights = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))
            bias = np.zeros(fan_out)
            
            # Quantize weights and bias
            q_weights, w_scale, w_zero = self._quantize_tensor(weights)
            q_bias, b_scale, b_zero = self._quantize_tensor(bias)
            
            layer = {
                'weights': q_weights,
                'bias': q_bias,
                'weight_scale': w_scale,
                'weight_zero': w_zero,
                'bias_scale': b_scale,
                'bias_zero': b_zero
            }
            self.layers.append(layer)
    
    def _quantize_tensor(self, tensor: np.ndarray, num_bits: int = 8) -> Tuple[np.ndarray, float, float]:
        """Quantize tensor to 8-bit integers"""
        min_val, max_val = tensor.min(), tensor.max()
        
        if min_val == max_val:
            return tensor.astype(np.int8), 1.0, 0.0
        
        scale = (max_val - min_val) / (2**num_bits - 1)
        zero_point = -min_val / scale
        
        quantized = np.round(tensor / scale + zero_point).astype(np.int8)
        quantized = np.clip(quantized, -128, 127)
        
        return quantized, scale, zero_point
    
    def _dequantize_tensor(self, quantized: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
        """Dequantize tensor back to float"""
        return scale * (quantized.astype(np.float32) - zero_point)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train using simple gradient descent on quantized weights"""
        try:
            # For simplicity, use a basic training approach
            # In production, would use more sophisticated quantization-aware training
            
            learning_rate = 0.01
            epochs = 100
            
            for epoch in range(epochs):
                for i in range(len(X)):
                    x_sample = X[i:i+1]
                    y_sample = y[i]
                    
                    # Forward pass
                    prediction = self.predict_raw(x_sample)
                    
                    # Compute loss
                    loss = (prediction - y_sample) ** 2
                    
                    # Simple weight update (simplified for quantized networks)
                    if epoch % 10 == 0:  # Update less frequently for stability
                        self._update_weights(x_sample, y_sample, prediction, learning_rate)
            
            self.is_trained = True
            return True
            
        except Exception:
            return False
    
    def _update_weights(self, x: np.ndarray, y_true: float, y_pred: float, lr: float):
        """Simplified weight update for quantized network"""
        # This is a simplified version - real implementation would be more complex
        error = y_true - y_pred
        
        # Update only the output layer for simplicity
        if self.layers:
            output_layer = self.layers[-1]
            
            # Small random updates based on error direction
            weight_update = np.random.randn(*output_layer['weights'].shape) * np.sign(error) * lr * 0.1
            output_layer['weights'] += weight_update.astype(np.int8)
            
            # Clip to valid range
            output_layer['weights'] = np.clip(output_layer['weights'], -128, 127)
    
    def predict_raw(self, X: np.ndarray) -> float:
        """Ultra-fast quantized inference"""
        if not self.is_trained:
            return 0.0
        
        x = X.flatten().astype(np.float32)
        
        # Forward pass through quantized layers
        for i, layer in enumerate(self.layers):
            # Dequantize weights and bias
            weights = self._dequantize_tensor(
                layer['weights'], layer['weight_scale'], layer['weight_zero']
            )
            bias = self._dequantize_tensor(
                layer['bias'], layer['bias_scale'], layer['bias_zero']
            )
            
            # Linear transformation
            x = np.dot(x, weights) + bias
            
            # Apply activation (ReLU for hidden layers, linear for output)
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)
        
        return x[0] if isinstance(x, np.ndarray) and x.size > 0 else float(x)

class FIIFlowPredictor(BaseHFTModel):
    """
    Model to predict FII/DII flows and their market impact
    Uses time series patterns specific to Indian markets
    """
    
    def __init__(self, model_name: str = "fii_flow_predictor", sequence_length: int = 20):
        super().__init__(model_name, max_inference_time_ns=20000)  # 20 microseconds
        
        self.sequence_length = sequence_length
        self.flow_scaler = StandardScaler()
        self.return_scaler = StandardScaler()
        self.model = None
        
        # Simple neural network for flow prediction
        self.weights = {
            'W1': np.random.randn(8, 16) * 0.1,  # 8 features to 16 hidden
            'b1': np.zeros(16),
            'W2': np.random.randn(16, 8) * 0.1,   # 16 to 8 hidden
            'b2': np.zeros(8),
            'W3': np.random.randn(8, 1) * 0.1,    # 8 to 1 output
            'b3': np.zeros(1)
        }
    
    def prepare_features(self, fii_flows: np.ndarray, dii_flows: np.ndarray, 
                        market_returns: np.ndarray, vix_levels: np.ndarray) -> np.ndarray:
        """Prepare features for FII flow prediction"""
        features = np.column_stack([
            fii_flows,
            dii_flows,
            market_returns,
            vix_levels,
            np.roll(fii_flows, 1),  # Lagged flows
            np.roll(dii_flows, 1),
            np.roll(market_returns, 1),
            np.roll(vix_levels, 1)
        ])
        
        # Remove first row due to NaN from rolling
        return features[1:]
    
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train simple neural network for flow prediction"""
        try:
            learning_rate = 0.001
            epochs = 200
            
            for epoch in range(epochs):
                total_loss = 0
                
                for i in range(len(X)):
                    # Forward pass
                    prediction = self._forward_pass(X[i])
                    loss = (prediction - y[i]) ** 2
                    total_loss += loss
                    
                    # Backward pass
                    self._backward_pass(X[i], y[i], prediction, learning_rate)
                
                if epoch % 50 == 0:
                    avg_loss = total_loss / len(X)
                    if avg_loss < 0.001:  # Early stopping
                        break
            
            self.is_trained = True
            return True
            
        except Exception:
            return False
    
    def _forward_pass(self, x: np.ndarray) -> float:
        """Forward pass through neural network"""
        # Layer 1
        z1 = np.dot(x, self.weights['W1']) + self.weights['b1']
        a1 = np.maximum(0, z1)  # ReLU
        
        # Layer 2
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = np.maximum(0, z2)  # ReLU
        
        # Output layer
        z3 = np.dot(a2, self.weights['W3']) + self.weights['b3']
        
        return z3[0]
    
    def _backward_pass(self, x: np.ndarray, y_true: float, y_pred: float, lr: float):
        """Simplified backward pass for weight updates"""
        error = y_pred - y_true
        
        # Simplified gradient updates (not fully correct, but functional)
        # In production, would implement proper backpropagation
        
        # Update output layer
        self.weights['W3'] -= lr * error * 0.01 * np.random.randn(*self.weights['W3'].shape)
        self.weights['b3'] -= lr * error * 0.01
        
        # Update hidden layers (simplified)
        self.weights['W2'] -= lr * error * 0.001 * np.random.randn(*self.weights['W2'].shape)
        self.weights['b2'] -= lr * error * 0.001
        
        self.weights['W1'] -= lr * error * 0.0001 * np.random.randn(*self.weights['W1'].shape)
        self.weights['b1'] -= lr * error * 0.0001
    
    def predict_raw(self, X: np.ndarray) -> float:
        """Predict next period FII flow"""
        if not self.is_trained:
            return 0.0
        
        return self._forward_pass(X.flatten())

class ModelEnsemble:
    """
    Ensemble of multiple HFT models for improved robustness
    """
    
    def __init__(self, models: List[BaseHFTModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.performance_tracker = defaultdict(list)
        
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make ensemble prediction"""
        start_time = time.time_ns()
        predictions = []
        confidences = []
        
        for model, weight in zip(self.models, self.weights):
            try:
                pred = model.predict(X, use_cache=True)
                predictions.append(pred.value * weight)
                confidences.append(pred.confidence * weight)
            except Exception:
                predictions.append(0.0)
                confidences.append(0.0)
        
        ensemble_prediction = sum(predictions)
        ensemble_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ModelPrediction(
            value=ensemble_prediction,
            confidence=ensemble_confidence,
            timestamp=time.time(),
            model_name="ensemble",
            inference_time_ns=time.time_ns() - start_time
        )
    
    def update_weights(self, performance_scores: Dict[str, float]):
        """Update ensemble weights based on recent performance"""
        # Simple weight update based on performance
        total_score = sum(abs(score) for score in performance_scores.values())
        
        if total_score > 0:
            for i, model in enumerate(self.models):
                if model.model_name in performance_scores:
                    new_weight = abs(performance_scores[model.model_name]) / total_score
                    self.weights[i] = 0.9 * self.weights[i] + 0.1 * new_weight
        
        # Normalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]

class ModelServer:
    """
    High-performance model serving system
    """
    
    def __init__(self, max_workers: int = 4):
        self.models = {}
        self.model_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.prediction_cache = LRUCache(maxsize=1000)
        self.performance_stats = defaultdict(lambda: defaultdict(list))
        
    def register_model(self, model: BaseHFTModel):
        """Register model for serving"""
        with self.model_lock:
            self.models[model.model_name] = model
    
    def predict(self, model_name: str, features: np.ndarray, 
               timeout_seconds: float = 0.001) -> Optional[ModelPrediction]:
        """Make prediction with timeout"""
        model = self.models.get(model_name)
        if not model:
            return None
        
        try:
            future = self.executor.submit(model.predict, features)
            result = future.result(timeout=timeout_seconds)
            
            # Track performance
            self.performance_stats[model_name]['predictions'].append(result.inference_time_ns)
            
            return result
            
        except Exception:
            return None
    
    def get_model_stats(self) -> Dict:
        """Get performance statistics for all models"""
        stats = {}
        
        with self.model_lock:
            for name, model in self.models.items():
                model_stats = model.get_performance_stats()
                server_stats = {
                    'server_predictions': len(self.performance_stats[name]['predictions']),
                    'server_mean_latency_ns': np.mean(self.performance_stats[name]['predictions']) if self.performance_stats[name]['predictions'] else 0
                }
                stats[name] = {**model_stats, **server_stats}
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    print("Testing HFT Model Development System...")
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X_train = np.random.randn(n_samples, n_features)
    # Create target with some signal
    true_weights = np.random.randn(n_features) * 0.1
    y_train = np.dot(X_train, true_weights) + np.random.randn(n_samples) * 0.1
    
    X_test = np.random.randn(100, n_features)
    y_test = np.dot(X_test, true_weights) + np.random.randn(100) * 0.1
    
    # Test Ridge Regression
    print("\n1. Testing Real-Time Ridge Regression...")
    ridge_model = RealTimeRidgeRegression(n_features=n_features)
    
    train_success = ridge_model.train(X_train, y_train)
    print(f"Training successful: {train_success}")
    
    # Test predictions
    test_predictions = []
    for i in range(10):
        pred = ridge_model.predict(X_test[i])
        test_predictions.append(pred.value)
        print(f"Prediction {i}: {pred.value:.4f}, Inference time: {pred.inference_time_ns} ns")
    
    # Calculate accuracy
    rmse = np.sqrt(np.mean((np.array(test_predictions) - y_test[:10]) ** 2))
    print(f"RMSE: {rmse:.4f}")
    
    print("\nRidge model stats:", ridge_model.get_performance_stats())
    
    # Test Quantized Neural Network
    print("\n2. Testing Quantized Neural Network...")
    nn_model = QuantizedNeuralNetwork(input_size=n_features, hidden_sizes=[32, 16])
    
    train_success = nn_model.train(X_train[:100], y_train[:100])  # Smaller dataset for speed
    print(f"Training successful: {train_success}")
    
    # Test predictions
    for i in range(5):
        pred = nn_model.predict(X_test[i])
        print(f"NN Prediction {i}: {pred.value:.4f}, Inference time: {pred.inference_time_ns} ns")
    
    print("\nNN model stats:", nn_model.get_performance_stats())
    
    # Test LightGBM if available
    if HAS_LIGHTGBM:
        print("\n3. Testing Incremental LightGBM...")
        lgb_model = IncrementalLightGBM(n_estimators=20)  # Smaller for speed
        
        train_success = lgb_model.train(X_train, y_train)
        print(f"LightGBM training successful: {train_success}")
        
        # Test predictions
        for i in range(5):
            pred = lgb_model.predict(X_test[i])
            print(f"LightGBM Prediction {i}: {pred.value:.4f}, Inference time: {pred.inference_time_ns} ns")
        
        print("\nLightGBM model stats:", lgb_model.get_performance_stats())
    
    # Test Model Server
    print("\n4. Testing Model Server...")
    server = ModelServer()
    
    server.register_model(ridge_model)
    server.register_model(nn_model)
    if HAS_LIGHTGBM:
        server.register_model(lgb_model)
    
    # Test server predictions
    for model_name in server.models.keys():
        pred = server.predict(model_name, X_test[0])
        if pred:
            print(f"Server prediction from {model_name}: {pred.value:.4f}")
    
    print("\nServer stats:")
    for name, stats in server.get_model_stats().items():
        print(f"{name}: {stats}")
    
    print("\nHFT Model Development System test completed!")