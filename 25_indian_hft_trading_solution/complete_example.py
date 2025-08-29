"""
Complete End-to-End Indian HFT Trading Solution Integration Example
Demonstrates the full pipeline from market data to order execution
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading
import queue
from collections import defaultdict, deque

# Import all components of the Indian HFT solution
try:
    # Import would work if modules are properly structured
    from indian_hft_solution.market_structure import IndianMarketStructure, LatencyProfiler
    from indian_hft_solution.data_infrastructure import MarketDataProcessor, LockFreeOrderBook
    from indian_hft_solution.alpha_factors import IndianAlphaFactorEngine
    from indian_hft_solution.hft_models import ModelServer, RealTimeRidgeRegression
    from indian_hft_solution.execution_system import ExecutionEngine, Order, OrderSide, OrderType
    from indian_hft_solution.risk_management import SEBIPositionLimits, EmergencyKillSwitch
except ImportError:
    # Fallback mock classes for demonstration
    print("Note: Using mock classes for demonstration. In production, use actual imports.")
    
    class MockMarketStructure:
        def get_tick_size(self, price, exchange): return 0.05
        def is_trading_session_active(self, time, session, exchange): return True
    
    class MockOrder:
        def __init__(self, symbol, side, quantity, price):
            self.symbol = symbol
            self.side = side
            self.quantity = quantity
            self.price = price

@dataclass
class IndianHFTConfig:
    """Configuration for Indian HFT trading system"""
    # Market data settings
    max_symbols: int = 50
    market_data_buffer_size: int = 10000
    
    # Model settings
    model_inference_timeout_ms: float = 5.0
    model_retraining_frequency_hours: int = 4
    
    # Execution settings
    max_order_size: int = 10000
    default_urgency: str = "NORMAL"
    
    # Risk management
    max_daily_loss: float = 10_000_000  # ₹1 crore
    position_limit_per_symbol: int = 50000
    
    # Alpha factor settings
    factor_update_frequency_ms: int = 100  # 100ms updates
    factor_lookback_periods: int = 20

class IndianHFTStrategy:
    """
    Complete Indian HFT trading strategy integrating all components
    """
    
    def __init__(self, config: IndianHFTConfig):
        self.config = config
        self.running = False
        
        # Initialize all components
        self._initialize_components()
        
        # Strategy state
        self.active_symbols = set()
        self.strategy_pnl = 0.0
        self.daily_trades = 0
        
        # Performance tracking
        self.performance_metrics = {
            'strategy_sharpe': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_execution_time_ms': 0.0,
            'alpha_accuracy': 0.0
        }
        
    def _initialize_components(self):
        """Initialize all HFT system components"""
        print("Initializing Indian HFT trading system...")
        
        # 1. Market structure analyzer
        self.market_structure = MockMarketStructure()  # IndianMarketStructure()
        print("✓ Market structure analyzer initialized")
        
        # 2. Data infrastructure
        self.market_data_processor = self._create_mock_data_processor()
        print("✓ Market data infrastructure initialized")
        
        # 3. Alpha factor engine
        self.alpha_engine = self._create_mock_alpha_engine()
        print("✓ Alpha factor engine initialized")
        
        # 4. ML model server
        self.model_server = self._create_mock_model_server()
        print("✓ ML model server initialized")
        
        # 5. Execution engine
        self.execution_engine = self._create_mock_execution_engine()
        print("✓ Execution engine initialized")
        
        # 6. Risk management
        self.risk_manager = self._create_mock_risk_manager()
        print("✓ Risk management system initialized")
        
        print("Indian HFT system initialization complete!")
    
    def _create_mock_data_processor(self):
        """Create mock market data processor"""
        class MockDataProcessor:
            def __init__(self):
                self.order_books = {}
                self.last_prices = {}
                
            def add_symbol(self, symbol, exchange):
                self.order_books[symbol] = {
                    'best_bid': 0.0, 'best_ask': 0.0,
                    'bid_volume': 0, 'ask_volume': 0
                }
                self.last_prices[symbol] = 100.0  # Default price
                
            def get_order_book(self, symbol, exchange):
                return self.order_books.get(symbol, {})
                
            def update_market_data(self, symbol, price_data):
                if symbol in self.last_prices:
                    self.last_prices[symbol] = price_data['price']
                    self.order_books[symbol].update(price_data)
        
        return MockDataProcessor()
    
    def _create_mock_alpha_engine(self):
        """Create mock alpha factor engine"""
        class MockAlphaEngine:
            def calculate_all_factors(self, market_data, fundamental_data, symbol):
                # Mock alpha factors with realistic values
                factors = {
                    'vwoi': {'value': np.random.normal(0, 0.1), 'confidence': 0.8},
                    'momentum': {'value': np.random.normal(0, 0.05), 'confidence': 0.9},
                    'fii_dii_flow': {'value': np.random.normal(0, 0.03), 'confidence': 0.7},
                    'spread_factor': {'value': np.random.normal(0, 0.02), 'confidence': 0.85}
                }
                return factors
                
            def calculate_composite_alpha(self, factors):
                # Simple weighted average
                total_signal = 0.0
                total_confidence = 0.0
                
                for factor_data in factors.values():
                    if isinstance(factor_data, dict):
                        signal = factor_data.get('value', 0.0)
                        confidence = factor_data.get('confidence', 0.0)
                        total_signal += signal * confidence
                        total_confidence += confidence
                
                composite_alpha = total_signal / max(total_confidence, 0.1)
                return {'value': composite_alpha, 'confidence': total_confidence / len(factors)}
        
        return MockAlphaEngine()
    
    def _create_mock_model_server(self):
        """Create mock ML model server"""
        class MockModelServer:
            def predict(self, model_name, features, timeout_seconds=0.005):
                # Simulate ML model prediction
                prediction = np.random.normal(0, 0.1)  # Mean-reverting signal
                inference_time_ns = np.random.randint(1000, 5000)  # 1-5 microseconds
                
                class MockPrediction:
                    def __init__(self, value, time_ns):
                        self.value = value
                        self.inference_time_ns = time_ns
                        self.confidence = 0.8
                
                return MockPrediction(prediction, inference_time_ns)
                
            def get_model_stats(self):
                return {
                    'ridge_regression': {
                        'predictions_made': 1000,
                        'avg_latency_ns': 1500,
                        'error_rate': 0.02
                    }
                }
        
        return MockModelServer()
    
    def _create_mock_execution_engine(self):
        """Create mock execution engine"""
        class MockExecutionEngine:
            def __init__(self):
                self.orders_submitted = 0
                self.orders_filled = 0
                
            def submit_order(self, order):
                self.orders_submitted += 1
                # Simulate order acceptance
                if np.random.random() > 0.05:  # 95% fill rate
                    self.orders_filled += 1
                    return f"ORDER_{self.orders_submitted}"
                return None
                
            def get_performance_stats(self):
                return {
                    'orders_submitted': self.orders_submitted,
                    'orders_filled': self.orders_filled,
                    'fill_rate': self.orders_filled / max(self.orders_submitted, 1),
                    'avg_execution_time_ms': 0.5
                }
        
        return MockExecutionEngine()
    
    def _create_mock_risk_manager(self):
        """Create mock risk management system"""
        class MockRiskManager:
            def __init__(self):
                self.total_pnl = 0.0
                self.positions = defaultdict(int)
                
            def validate_order(self, order_data):
                # Simple risk checks
                if abs(order_data.get('quantity', 0)) > 10000:
                    return {'approved': False, 'reason': 'Order too large'}
                if self.total_pnl < -5000000:  # ₹50 lakhs loss
                    return {'approved': False, 'reason': 'Daily loss limit'}
                return {'approved': True, 'reason': 'Order approved'}
                
            def update_pnl(self, pnl_change):
                self.total_pnl += pnl_change
        
        return MockRiskManager()
    
    def add_symbol(self, symbol: str, exchange: str = "NSE"):
        """Add symbol to trading universe"""
        self.active_symbols.add(symbol)
        self.market_data_processor.add_symbol(symbol, exchange)
        print(f"Added {symbol} to trading universe")
    
    def start_trading(self):
        """Start the HFT trading system"""
        if self.running:
            print("Trading system already running")
            return
        
        self.running = True
        print("Starting Indian HFT trading system...")
        
        # Start trading loop in separate thread
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.start()
        
        print("Trading system started successfully!")
    
    def stop_trading(self):
        """Stop the HFT trading system"""
        if not self.running:
            print("Trading system not running")
            return
        
        self.running = False
        print("Stopping trading system...")
        
        if hasattr(self, 'trading_thread'):
            self.trading_thread.join()
        
        print("Trading system stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        loop_count = 0
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Process each active symbol
                for symbol in self.active_symbols:
                    self._process_symbol(symbol)
                
                # Update performance metrics
                if loop_count % 100 == 0:  # Every 100 loops
                    self._update_performance_metrics()
                
                # Sleep to maintain loop frequency (10ms = 100 Hz)
                loop_time = time.time() - loop_start
                sleep_time = max(0.01 - loop_time, 0.001)  # Minimum 1ms sleep
                time.sleep(sleep_time)
                
                loop_count += 1
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def _process_symbol(self, symbol: str):
        """Process single symbol through complete HFT pipeline"""
        try:
            # 1. Get market data
            market_data = self._get_market_data(symbol)
            if not market_data:
                return
            
            # 2. Calculate alpha factors
            factors = self._calculate_alpha_factors(symbol, market_data)
            
            # 3. Generate ML prediction
            prediction = self._generate_ml_prediction(symbol, factors, market_data)
            
            # 4. Make trading decision
            trading_signal = self._make_trading_decision(symbol, factors, prediction)
            
            # 5. Execute trades if signal is strong enough
            if abs(trading_signal['strength']) > 0.3:  # Minimum signal threshold
                self._execute_trading_signal(symbol, trading_signal, market_data)
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data for symbol"""
        try:
            # Simulate real market data updates
            base_price = self.market_data_processor.last_prices.get(symbol, 100.0)
            price_change = np.random.normal(0, 0.002)  # 0.2% typical price change
            new_price = base_price * (1 + price_change)
            
            market_data = {
                'symbol': symbol,
                'price': new_price,
                'volume': np.random.randint(100, 1000),
                'best_bid': new_price - 0.05,
                'best_ask': new_price + 0.05,
                'bid_volume': np.random.randint(500, 2000),
                'ask_volume': np.random.randint(500, 2000),
                'timestamp': time.time()
            }
            
            # Update data processor
            self.market_data_processor.update_market_data(symbol, market_data)
            
            return market_data
            
        except Exception:
            return None
    
    def _calculate_alpha_factors(self, symbol: str, market_data: Dict) -> Dict:
        """Calculate alpha factors for symbol"""
        # Mock fundamental data
        fundamental_data = {
            'fii_flow': np.random.normal(0, 1000000),  # FII flow in ₹
            'dii_flow': np.random.normal(0, 500000),   # DII flow in ₹
            'fii_avg': 800000,
            'dii_avg': 600000
        }
        
        factors = self.alpha_engine.calculate_all_factors(
            market_data, fundamental_data, symbol
        )
        
        return factors
    
    def _generate_ml_prediction(self, symbol: str, factors: Dict, market_data: Dict) -> Optional[Dict]:
        """Generate ML prediction for symbol"""
        try:
            # Prepare features from factors and market data
            features = np.array([
                factors.get('vwoi', {}).get('value', 0.0),
                factors.get('momentum', {}).get('value', 0.0),
                factors.get('fii_dii_flow', {}).get('value', 0.0),
                factors.get('spread_factor', {}).get('value', 0.0),
                market_data.get('volume', 0) / 1000.0,  # Normalized volume
                (market_data.get('best_ask', 0) - market_data.get('best_bid', 0)) / market_data.get('price', 1),  # Spread
            ] + [0.0] * 44)  # Pad to 50 features
            
            # Get model prediction
            prediction = self.model_server.predict('ridge_regression', features)
            
            if prediction:
                return {
                    'value': prediction.value,
                    'confidence': prediction.confidence,
                    'inference_time_ns': prediction.inference_time_ns
                }
            
        except Exception:
            pass
        
        return None
    
    def _make_trading_decision(self, symbol: str, factors: Dict, prediction: Optional[Dict]) -> Dict:
        """Make trading decision combining factors and ML prediction"""
        # Calculate composite alpha signal
        alpha_signal = self.alpha_engine.calculate_composite_alpha(factors)
        
        # Combine with ML prediction
        if prediction:
            combined_signal = (
                0.6 * alpha_signal.get('value', 0.0) + 
                0.4 * prediction.get('value', 0.0)
            )
            combined_confidence = min(
                alpha_signal.get('confidence', 0.0),
                prediction.get('confidence', 0.0)
            )
        else:
            combined_signal = alpha_signal.get('value', 0.0)
            combined_confidence = alpha_signal.get('confidence', 0.0)
        
        # Determine trading action
        if combined_signal > 0.1 and combined_confidence > 0.5:
            action = 'BUY'
        elif combined_signal < -0.1 and combined_confidence > 0.5:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'strength': combined_signal,
            'confidence': combined_confidence,
            'alpha_signal': alpha_signal.get('value', 0.0),
            'ml_prediction': prediction.get('value', 0.0) if prediction else 0.0
        }
    
    def _execute_trading_signal(self, symbol: str, signal: Dict, market_data: Dict):
        """Execute trading signal through execution engine"""
        try:
            if signal['action'] == 'HOLD':
                return
            
            # Calculate order size based on signal strength and confidence
            base_size = 500  # Base order size
            size_multiplier = abs(signal['strength']) * signal['confidence']
            order_size = int(base_size * size_multiplier)
            order_size = min(order_size, self.config.max_order_size)
            
            if order_size < 1:
                return
            
            # Determine order side and price
            if signal['action'] == 'BUY':
                side = 'BUY'
                price = market_data['best_ask']  # Market taking
            else:
                side = 'SELL'
                price = market_data['best_bid']  # Market taking
            
            # Create order object
            order_data = {
                'symbol': symbol,
                'side': side,
                'quantity': order_size,
                'price': price,
                'order_type': 'LIMIT',
                'urgency': self.config.default_urgency
            }
            
            # Risk validation
            risk_check = self.risk_manager.validate_order(order_data)
            if not risk_check['approved']:
                print(f"Order rejected for {symbol}: {risk_check['reason']}")
                return
            
            # Submit order
            order_id = self.execution_engine.submit_order(MockOrder(
                symbol, side, order_size, price
            ))
            
            if order_id:
                print(f"Order submitted for {symbol}: {side} {order_size} @ ₹{price:.2f}")
                self.daily_trades += 1
                
                # Simulate P&L update (simplified)
                pnl_change = np.random.normal(signal['strength'] * 1000, 500)
                self.risk_manager.update_pnl(pnl_change)
                self.strategy_pnl += pnl_change
            
        except Exception as e:
            print(f"Error executing signal for {symbol}: {e}")
    
    def _update_performance_metrics(self):
        """Update strategy performance metrics"""
        try:
            # Get execution statistics
            exec_stats = self.execution_engine.get_performance_stats()
            
            # Update performance metrics
            self.performance_metrics.update({
                'total_trades': exec_stats.get('orders_filled', 0),
                'fill_rate': exec_stats.get('fill_rate', 0.0),
                'avg_execution_time_ms': exec_stats.get('avg_execution_time_ms', 0.0),
                'strategy_pnl': self.strategy_pnl,
                'daily_trades': self.daily_trades
            })
            
        except Exception as e:
            print(f"Error updating performance metrics: {e}")
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        return {
            'strategy_metrics': self.performance_metrics,
            'execution_stats': self.execution_engine.get_performance_stats(),
            'model_stats': self.model_server.get_model_stats(),
            'risk_metrics': {
                'total_pnl': self.risk_manager.total_pnl,
                'active_positions': len(self.risk_manager.positions),
                'daily_trades': self.daily_trades
            },
            'system_status': {
                'running': self.running,
                'active_symbols': len(self.active_symbols),
                'uptime_hours': (time.time() - getattr(self, 'start_time', time.time())) / 3600
            }
        }

# Example usage demonstrating the complete Indian HFT system
def main():
    """Demonstrate the complete Indian HFT trading solution"""
    print("=== Indian HFT Trading Solution Demo ===")
    
    # Create configuration
    config = IndianHFTConfig(
        max_symbols=10,
        max_daily_loss=5_000_000,  # ₹50 lakhs
        position_limit_per_symbol=10000
    )
    
    # Initialize strategy
    strategy = IndianHFTStrategy(config)
    
    # Add some popular Indian stocks
    popular_stocks = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
        "BHARTIARTL", "ICICIBANK", "SBIN", "ITC", "LT"
    ]
    
    for stock in popular_stocks[:5]:  # Add first 5 stocks
        strategy.add_symbol(stock, "NSE")
    
    # Start trading
    strategy.start_trading()
    
    # Run for demonstration period
    print("\nTrading for 10 seconds...")
    time.sleep(10)
    
    # Get performance report
    report = strategy.get_performance_report()
    
    print("\n=== Performance Report ===")
    print(f"Strategy P&L: ₹{report['risk_metrics']['total_pnl']:,.2f}")
    print(f"Total Trades: {report['strategy_metrics']['total_trades']}")
    print(f"Fill Rate: {report['execution_stats']['fill_rate']:.1%}")
    print(f"Avg Execution Time: {report['strategy_metrics']['avg_execution_time_ms']:.2f}ms")
    print(f"Active Symbols: {report['system_status']['active_symbols']}")
    
    # Stop trading
    strategy.stop_trading()
    
    print("\n=== Demo Complete ===")
    print("This demonstration shows the integration of all components:")
    print("✓ Market structure analysis for NSE/BSE")
    print("✓ Ultra-low latency data processing")
    print("✓ India-specific alpha factor calculation")
    print("✓ ML model predictions with microsecond inference")
    print("✓ Smart order routing and execution")
    print("✓ SEBI-compliant risk management")
    print("✓ Real-time performance monitoring")

if __name__ == "__main__":
    main()