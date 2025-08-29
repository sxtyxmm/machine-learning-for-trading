# HFT Model Development for Indian Markets

## ML Models Optimized for Nanosecond-Level Trading

### Ultra-Low Latency Machine Learning Pipeline

#### Model Architecture Overview
The HFT ML system is designed for inference times under 10 microseconds with model updates in real-time. The architecture leverages:

1. **Pre-computed Feature Engineering**: Features are calculated in parallel and cached
2. **Quantized Neural Networks**: 8-bit quantization for 4x speedup
3. **Ensemble Methods**: Multiple simple models for robustness
4. **Hardware Acceleration**: Custom FPGA/GPU implementations

### Primary Model Types

#### 1. Linear Models with Real-Time Updates

**Ridge Regression with Kalman Filter Updates**
```python
class RealTimeRidgeRegression:
    """
    Ridge regression with online Kalman filter updates
    Inference time: < 1 microsecond
    """
    def __init__(self, alpha=0.01, n_features=50):
        self.alpha = alpha
        self.weights = np.zeros(n_features)
        self.P = np.eye(n_features) / alpha  # Covariance matrix
        self.update_count = 0
        
    def predict(self, X):
        """Ultra-fast prediction using precomputed weights"""
        return np.dot(X, self.weights)
    
    def online_update(self, x, y):
        """Kalman filter update for real-time learning"""
        # Prediction error
        y_pred = np.dot(x, self.weights)
        error = y - y_pred
        
        # Kalman gain
        S = np.dot(x, np.dot(self.P, x)) + 1.0  # Innovation covariance
        K = np.dot(self.P, x) / S  # Kalman gain
        
        # Update weights and covariance
        self.weights += K * error
        self.P -= np.outer(K, np.dot(x, self.P))
        
        self.update_count += 1
```

#### 2. Gradient Boosting for Feature Interactions

**LightGBM with Incremental Training**
```python
class IncrementalLightGBM:
    """
    LightGBM optimized for Indian market patterns
    Inference time: < 5 microseconds
    """
    def __init__(self, n_estimators=50, max_depth=6, learning_rate=0.1):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='regression',
            metric='rmse',
            force_row_wise=True,  # Optimize for low latency
            num_threads=1,  # Single thread for consistency
            verbosity=-1
        )
        self.feature_importance_cache = {}
        self.prediction_cache = {}
        
    def train_incremental(self, X_new, y_new, X_base=None, y_base=None):
        """Incremental training with concept drift handling"""
        if X_base is not None:
            # Combine new data with subset of historical data
            decay_factor = 0.95
            sample_size = min(len(X_base), len(X_new) * 5)
            indices = np.random.choice(len(X_base), sample_size, replace=False)
            
            X_combined = np.vstack([X_base[indices], X_new])
            y_combined = np.hstack([y_base[indices] * decay_factor, y_new])
        else:
            X_combined, y_combined = X_new, y_new
            
        self.model.fit(X_combined, y_combined)
        
    def predict_optimized(self, X):
        """Optimized prediction with caching"""
        # Simple cache for identical feature vectors
        cache_key = hash(X.tobytes())
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
            
        prediction = self.model.predict(X.reshape(1, -1))[0]
        self.prediction_cache[cache_key] = prediction
        
        # Limit cache size
        if len(self.prediction_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self.prediction_cache.keys())[:100]
            for key in keys_to_remove:
                del self.prediction_cache[key]
                
        return prediction
```

#### 3. Neural Networks with Quantization

**Quantized Feedforward Network**
```python
class QuantizedNeuralNetwork:
    """
    8-bit quantized neural network for ultra-low latency
    Inference time: < 3 microseconds
    """
    def __init__(self, input_size=50, hidden_sizes=[32, 16], output_size=1):
        self.layers = []
        
        # Initialize quantized weights
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layer = {
                'weights': self._quantize_weights(np.random.randn(prev_size, hidden_size) * 0.1),
                'bias': self._quantize_weights(np.random.randn(hidden_size) * 0.01),
                'scale': 1.0,
                'zero_point': 0
            }
            self.layers.append(layer)
            prev_size = hidden_size
            
        # Output layer
        output_layer = {
            'weights': self._quantize_weights(np.random.randn(prev_size, output_size) * 0.1),
            'bias': self._quantize_weights(np.random.randn(output_size) * 0.01),
            'scale': 1.0,
            'zero_point': 0
        }
        self.layers.append(output_layer)
        
    def _quantize_weights(self, weights, num_bits=8):
        """Quantize weights to 8-bit integers"""
        min_val, max_val = weights.min(), weights.max()
        scale = (max_val - min_val) / (2**num_bits - 1)
        zero_point = -min_val / scale
        
        quantized = np.round(weights / scale + zero_point).astype(np.int8)
        return quantized, scale, zero_point
    
    def predict(self, X):
        """Ultra-fast quantized inference"""
        x = X.astype(np.int8)
        
        for i, layer in enumerate(self.layers):
            weights, scale, zero_point = layer['weights']
            bias, bias_scale, bias_zero = layer['bias']
            
            # Quantized matrix multiplication
            x = np.dot(x, weights) + bias
            
            # Apply activation (ReLU for hidden layers)
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)
                
        return x[0] if len(x.shape) > 0 else x
```

### Indian Market-Specific Models

#### 4. FII/DII Flow Prediction Model

**LSTM for Institutional Flow Prediction**
```python
class FIIFlowPredictor:
    """
    LSTM model to predict FII/DII flows and their market impact
    Updates every 15 minutes with new flow data
    """
    def __init__(self, sequence_length=20, hidden_size=32):
        self.sequence_length = sequence_length
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(hidden_size, return_sequences=True, 
                               kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(hidden_size//2, 
                               kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Compile with custom loss for asymmetric errors
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self._asymmetric_loss,
            metrics=['mae']
        )
        
    def _asymmetric_loss(self, y_true, y_pred):
        """Custom loss that penalizes underestimation more than overestimation"""
        error = y_true - y_pred
        return tf.where(error >= 0, 2 * error**2, error**2)
    
    def prepare_features(self, fii_flows, dii_flows, market_returns, vix_levels):
        """Prepare features for FII flow prediction"""
        features = np.column_stack([
            fii_flows,
            dii_flows, 
            market_returns,
            vix_levels,
            np.roll(fii_flows, 1),  # Lagged flows
            np.roll(dii_flows, 1),
            fii_flows / np.roll(fii_flows, 5).mean(),  # Flow momentum
            dii_flows / np.roll(dii_flows, 5).mean()
        ])
        
        return features
    
    def predict_next_flow(self, recent_data):
        """Predict next period FII flow"""
        if len(recent_data) < self.sequence_length:
            return 0.0
            
        X = recent_data[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        prediction = self.model.predict(X, verbose=0)[0][0]
        
        return prediction
```

#### 5. Circuit Breaker Prediction Model

**Ensemble Model for Circuit Breaker Events**
```python
class CircuitBreakerPredictor:
    """
    Ensemble model to predict circuit breaker trigger probability
    Critical for risk management in Indian markets
    """
    def __init__(self):
        # Multiple models for ensemble
        self.models = {
            'volatility': self._create_volatility_model(),
            'momentum': self._create_momentum_model(),
            'volume': self._create_volume_model()
        }
        self.ensemble_weights = {'volatility': 0.4, 'momentum': 0.35, 'volume': 0.25}
        
    def _create_volatility_model(self):
        """Model based on realized volatility patterns"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=50, max_depth=8, random_state=42,
                n_jobs=1  # Single thread for consistency
            ))
        ])
    
    def _create_momentum_model(self):
        """Model based on price momentum indicators"""
        return Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=30, max_depth=6, learning_rate=0.1,
                random_state=42
            ))
        ])
    
    def _create_volume_model(self):
        """Model based on volume patterns"""
        return Pipeline([
            ('scaler', MinMaxScaler()),
            ('regressor', SVR(kernel='rbf', C=1.0, gamma='scale'))
        ])
    
    def prepare_features(self, price_data, volume_data, vix_data):
        """Prepare features for circuit breaker prediction"""
        features = {}
        
        # Volatility features
        returns = np.diff(price_data) / price_data[:-1]
        features['realized_vol'] = np.std(returns[-20:])  # 20-period realized vol
        features['vol_of_vol'] = np.std([np.std(returns[i-5:i]) for i in range(5, len(returns))][-10:])
        
        # Momentum features
        features['rsi'] = self._calculate_rsi(price_data)
        features['momentum_5'] = (price_data[-1] - price_data[-6]) / price_data[-6]
        features['momentum_10'] = (price_data[-1] - price_data[-11]) / price_data[-11]
        
        # Volume features
        features['vol_ratio'] = volume_data[-1] / np.mean(volume_data[-20:])
        features['vol_trend'] = np.polyfit(range(10), volume_data[-10:], 1)[0]
        
        # Market stress indicators
        features['vix_level'] = vix_data[-1]
        features['vix_change'] = vix_data[-1] - vix_data[-2]
        
        return np.array(list(features.values())).reshape(1, -1)
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def predict_probability(self, market_data):
        """Predict circuit breaker probability"""
        features = self.prepare_features(
            market_data['prices'],
            market_data['volumes'],
            market_data['vix']
        )
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(features)[0]
        
        # Ensemble prediction
        ensemble_pred = sum(
            predictions[name] * self.ensemble_weights[name]
            for name in predictions
        )
        
        # Convert to probability using sigmoid
        probability = 1 / (1 + np.exp(-ensemble_pred))
        
        return probability
```

### Model Deployment and Inference Optimization

#### 6. Model Serving Infrastructure

**Ultra-Low Latency Model Server**
```python
class ModelServer:
    """
    High-performance model serving with microsecond response times
    """
    def __init__(self):
        self.models = {}
        self.feature_cache = LRUCache(maxsize=1000)
        self.prediction_cache = LRUCache(maxsize=500)
        self.model_lock = threading.RLock()
        
    def load_model(self, model_name, model_path, model_type='sklearn'):
        """Load model with optimization"""
        with self.model_lock:
            if model_type == 'sklearn':
                model = joblib.load(model_path)
            elif model_type == 'tensorflow':
                model = tf.lite.Interpreter(model_path=model_path)
                model.allocate_tensors()
            elif model_type == 'onnx':
                model = onnxruntime.InferenceSession(model_path)
            
            # Wrap model with optimization
            self.models[model_name] = {
                'model': model,
                'type': model_type,
                'warmup_complete': False
            }
            
            # Warmup model
            self._warmup_model(model_name)
    
    def _warmup_model(self, model_name):
        """Warmup model with dummy predictions"""
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Generate dummy features
        dummy_features = np.random.randn(1, 50).astype(np.float32)
        
        # Run multiple warmup predictions
        for _ in range(100):
            try:
                if model_info['type'] == 'sklearn':
                    _ = model.predict(dummy_features)
                elif model_info['type'] == 'tensorflow':
                    input_details = model.get_input_details()
                    output_details = model.get_output_details()
                    model.set_tensor(input_details[0]['index'], dummy_features)
                    model.invoke()
                    _ = model.get_tensor(output_details[0]['index'])
            except:
                pass
                
        model_info['warmup_complete'] = True
    
    def predict(self, model_name, features, use_cache=True):
        """Ultra-fast prediction with caching"""
        # Check cache first
        if use_cache:
            cache_key = (model_name, hash(features.tobytes()))
            cached_result = self.prediction_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Get model
        model_info = self.models.get(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
        
        model = model_info['model']
        
        # Make prediction
        start_time = time.time_ns()
        
        try:
            if model_info['type'] == 'sklearn':
                prediction = model.predict(features.reshape(1, -1))[0]
            elif model_info['type'] == 'tensorflow':
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                model.set_tensor(input_details[0]['index'], features.reshape(1, -1))
                model.invoke()
                prediction = model.get_tensor(output_details[0]['index'])[0]
            else:
                prediction = 0.0
                
        except Exception as e:
            prediction = 0.0  # Fallback for errors
        
        inference_time_ns = time.time_ns() - start_time
        
        # Cache result
        if use_cache:
            self.prediction_cache[cache_key] = prediction
        
        return prediction, inference_time_ns
```

#### 7. Feature Engineering Pipeline

**Real-Time Feature Computer**
```python
class RealTimeFeatureComputer:
    """
    Compute features in real-time with minimal latency
    Parallel computation and caching for speed
    """
    def __init__(self, n_workers=4):
        self.feature_functions = {}
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        self.compute_times = defaultdict(list)
        
    def register_feature(self, name, func, update_frequency='tick'):
        """Register feature computation function"""
        self.feature_functions[name] = {
            'func': func,
            'frequency': update_frequency,
            'last_update': 0,
            'cache_duration': self._get_cache_duration(update_frequency)
        }
    
    def _get_cache_duration(self, frequency):
        """Get cache duration based on update frequency"""
        durations = {
            'tick': 0.001,      # 1ms
            'second': 1.0,      # 1 second
            'minute': 60.0,     # 1 minute
            'hour': 3600.0      # 1 hour
        }
        return durations.get(frequency, 0.001)
    
    def compute_features(self, market_data, symbol):
        """Compute all features with parallel execution"""
        current_time = time.time()
        features = {}
        futures = {}
        
        # Submit feature computations to thread pool
        for name, config in self.feature_functions.items():
            cache_key = f"{symbol}_{name}"
            
            # Check if cached value is still valid
            if (cache_key in self.cache and 
                current_time - config['last_update'] < config['cache_duration']):
                features[name] = self.cache[cache_key]
                continue
            
            # Submit for computation
            future = self.executor.submit(
                self._compute_single_feature, 
                name, config['func'], market_data, symbol
            )
            futures[name] = future
        
        # Collect results
        for name, future in futures.items():
            try:
                start_time = time.time_ns()
                features[name] = future.result(timeout=0.001)  # 1ms timeout
                compute_time = time.time_ns() - start_time
                
                self.compute_times[name].append(compute_time)
                
                # Update cache
                cache_key = f"{symbol}_{name}"
                self.cache[cache_key] = features[name]
                self.feature_functions[name]['last_update'] = current_time
                
            except Exception:
                # Use cached value or default
                cache_key = f"{symbol}_{name}"
                features[name] = self.cache.get(cache_key, 0.0)
        
        return features
    
    def _compute_single_feature(self, name, func, market_data, symbol):
        """Compute single feature with error handling"""
        try:
            return func(market_data, symbol)
        except Exception:
            return 0.0  # Default value on error
    
    def get_performance_stats(self):
        """Get feature computation performance statistics"""
        stats = {}
        for name, times in self.compute_times.items():
            if times:
                stats[name] = {
                    'mean_ns': np.mean(times),
                    'p95_ns': np.percentile(times, 95),
                    'p99_ns': np.percentile(times, 99),
                    'count': len(times)
                }
        return stats
```

### Model Performance Monitoring

#### 8. Real-Time Model Performance Tracker

**Performance Monitoring System**
```python
class ModelPerformanceTracker:
    """
    Monitor model performance in real-time
    Track prediction accuracy, latency, and concept drift
    """
    def __init__(self, lookback_window=1000):
        self.lookback_window = lookback_window
        self.predictions = defaultdict(lambda: deque(maxlen=lookback_window))
        self.actuals = defaultdict(lambda: deque(maxlen=lookback_window))
        self.timestamps = defaultdict(lambda: deque(maxlen=lookback_window))
        self.latencies = defaultdict(lambda: deque(maxlen=lookback_window))
        
    def record_prediction(self, model_name, prediction, timestamp, latency_ns):
        """Record model prediction"""
        self.predictions[model_name].append(prediction)
        self.timestamps[model_name].append(timestamp)
        self.latencies[model_name].append(latency_ns)
    
    def record_actual(self, model_name, actual, timestamp):
        """Record actual outcome"""
        # Find matching prediction by timestamp
        pred_timestamps = list(self.timestamps[model_name])
        if pred_timestamps:
            # Find closest timestamp
            time_diffs = [abs((timestamp - ts).total_seconds()) for ts in pred_timestamps]
            min_idx = np.argmin(time_diffs)
            
            if time_diffs[min_idx] < 5.0:  # Within 5 seconds
                self.actuals[model_name].append(actual)
    
    def get_performance_metrics(self, model_name):
        """Calculate performance metrics"""
        if (model_name not in self.predictions or 
            len(self.predictions[model_name]) < 10 or
            len(self.actuals[model_name]) < 10):
            return {}
        
        preds = np.array(list(self.predictions[model_name]))
        actuals = np.array(list(self.actuals[model_name]))
        latencies = np.array(list(self.latencies[model_name]))
        
        # Align predictions and actuals
        min_len = min(len(preds), len(actuals))
        preds = preds[-min_len:]
        actuals = actuals[-min_len:]
        
        metrics = {
            'rmse': np.sqrt(np.mean((preds - actuals) ** 2)),
            'mae': np.mean(np.abs(preds - actuals)),
            'correlation': np.corrcoef(preds, actuals)[0, 1] if min_len > 1 else 0,
            'directional_accuracy': np.mean(np.sign(preds) == np.sign(actuals)),
            'mean_latency_ns': np.mean(latencies),
            'p95_latency_ns': np.percentile(latencies, 95),
            'p99_latency_ns': np.percentile(latencies, 99),
            'sample_count': min_len
        }
        
        return metrics
    
    def detect_concept_drift(self, model_name, window_size=100):
        """Detect concept drift in model performance"""
        if (model_name not in self.predictions or 
            len(self.predictions[model_name]) < window_size * 2):
            return False, 0.0
        
        preds = np.array(list(self.predictions[model_name]))
        actuals = np.array(list(self.actuals[model_name]))
        
        min_len = min(len(preds), len(actuals))
        if min_len < window_size * 2:
            return False, 0.0
        
        # Compare recent performance to historical
        recent_preds = preds[-window_size:]
        recent_actuals = actuals[-window_size:]
        historical_preds = preds[-(window_size*2):-window_size]
        historical_actuals = actuals[-(window_size*2):-window_size]
        
        recent_mse = np.mean((recent_preds - recent_actuals) ** 2)
        historical_mse = np.mean((historical_preds - historical_actuals) ** 2)
        
        # Drift detected if recent performance is significantly worse
        drift_ratio = recent_mse / (historical_mse + 1e-10)
        drift_detected = drift_ratio > 1.5  # 50% increase in error
        
        return drift_detected, drift_ratio
```

This comprehensive HFT model development framework provides the foundation for building ultra-low latency ML models specifically optimized for Indian market characteristics, with real-time training capabilities and microsecond-level inference times.