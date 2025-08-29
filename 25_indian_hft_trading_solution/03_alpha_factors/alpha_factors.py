"""
India-Specific Alpha Factor Engineering for High-Frequency Trading
Comprehensive factor library adapted for NSE/BSE market microstructure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, time
import scipy.stats as stats
from collections import deque, defaultdict
import warnings

# Import the market structure components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_market_structure'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_data_infrastructure'))

try:
    from market_structure import IndianMarketStructure, Exchange
    from data_infrastructure import LockFreeOrderBook, MarketDataMessage
except ImportError:
    # Fallback if imports not available
    class Exchange:
        NSE = "NSE"
        BSE = "BSE"

@dataclass
class AlphaFactorResult:
    """Container for alpha factor calculation result"""
    value: float
    confidence: float
    timestamp: datetime
    factor_name: str
    symbol: str
    metadata: Optional[Dict] = None

class IndianMicrostructureFactors:
    """
    Microstructure-based alpha factors specific to Indian markets
    """
    
    def __init__(self, market_structure: Optional[object] = None):
        self.market_structure = market_structure or IndianMarketStructure()
        self.factor_cache = defaultdict(deque)
        self.cache_size = 1000
        
    def volume_weighted_order_imbalance(self, order_book: LockFreeOrderBook, 
                                      levels: int = 5) -> AlphaFactorResult:
        """
        Calculate Volume Weighted Order Imbalance adapted for Indian markets
        Accounts for NSE/BSE specific tick sizes and lot structures
        """
        try:
            # Calculate weighted volumes for top levels
            bid_volume = sum(
                level.quantity * level.price 
                for level in order_book.bid_levels[:levels] 
                if level.price > 0
            )
            
            ask_volume = sum(
                level.quantity * level.price 
                for level in order_book.ask_levels[:levels] 
                if level.price > 0
            )
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                imbalance = 0.0
                confidence = 0.0
            else:
                imbalance = (bid_volume - ask_volume) / total_volume
                # Confidence based on total volume and number of active levels
                active_levels = sum(1 for level in order_book.bid_levels[:levels] if level.price > 0)
                active_levels += sum(1 for level in order_book.ask_levels[:levels] if level.price > 0)
                confidence = min(active_levels / (2 * levels), 1.0)
            
            return AlphaFactorResult(
                value=imbalance,
                confidence=confidence,
                timestamp=datetime.now(),
                factor_name="vwoi",
                symbol=order_book.symbol,
                metadata={'bid_volume': bid_volume, 'ask_volume': ask_volume, 'levels': levels}
            )
            
        except Exception as e:
            return AlphaFactorResult(
                value=0.0, confidence=0.0, timestamp=datetime.now(),
                factor_name="vwoi", symbol=order_book.symbol
            )
    
    def tick_normalized_spread(self, order_book: LockFreeOrderBook, 
                              exchange: str = "NSE") -> AlphaFactorResult:
        """
        Calculate spread normalized by applicable tick size
        Critical for comparing liquidity across different price bands
        """
        try:
            best_bid, best_ask = order_book.get_best_bid_ask()
            
            if not best_bid or not best_ask:
                return AlphaFactorResult(
                    value=np.nan, confidence=0.0, timestamp=datetime.now(),
                    factor_name="tick_normalized_spread", symbol=order_book.symbol
                )
            
            # Get appropriate tick size for the mid price
            mid_price = (best_bid + best_ask) / 2.0
            tick_size = self.market_structure.get_tick_size(
                mid_price, Exchange.NSE if exchange == "NSE" else Exchange.BSE
            )
            
            raw_spread = best_ask - best_bid
            normalized_spread = raw_spread / tick_size
            
            # Confidence based on how close spread is to minimum (1 tick)
            confidence = 1.0 / max(normalized_spread, 1.0)
            
            return AlphaFactorResult(
                value=normalized_spread,
                confidence=confidence,
                timestamp=datetime.now(),
                factor_name="tick_normalized_spread",
                symbol=order_book.symbol,
                metadata={'raw_spread': raw_spread, 'tick_size': tick_size, 'mid_price': mid_price}
            )
            
        except Exception as e:
            return AlphaFactorResult(
                value=np.nan, confidence=0.0, timestamp=datetime.now(),
                factor_name="tick_normalized_spread", symbol=order_book.symbol
            )
    
    def order_book_depth_pressure(self, order_book: LockFreeOrderBook, 
                                 levels: int = 10) -> AlphaFactorResult:
        """
        Calculate pressure from order book depth
        Adapted for NSE/BSE typical order book depths
        """
        try:
            bid_depth = sum(level.quantity for level in order_book.bid_levels[:levels] if level.price > 0)
            ask_depth = sum(level.quantity for level in order_book.ask_levels[:levels] if level.price > 0)
            
            total_depth = bid_depth + ask_depth
            if total_depth == 0:
                pressure = 0.0
                confidence = 0.0
            else:
                pressure = (bid_depth - ask_depth) / total_depth
                # Confidence based on total depth relative to typical values
                confidence = min(total_depth / 10000, 1.0)  # Assuming 10k is good depth
            
            return AlphaFactorResult(
                value=pressure,
                confidence=confidence,
                timestamp=datetime.now(),
                factor_name="depth_pressure",
                symbol=order_book.symbol,
                metadata={'bid_depth': bid_depth, 'ask_depth': ask_depth, 'levels': levels}
            )
            
        except Exception:
            return AlphaFactorResult(
                value=0.0, confidence=0.0, timestamp=datetime.now(),
                factor_name="depth_pressure", symbol=order_book.symbol
            )

class FIIDIIFlowFactors:
    """
    Alpha factors based on Foreign and Domestic Institutional Investor flows
    """
    
    def __init__(self):
        self.fii_flow_cache = deque(maxlen=100)
        self.dii_flow_cache = deque(maxlen=100)
        
    def fii_dii_adjusted_imbalance(self, base_imbalance: float, 
                                  fii_net_flow: float, dii_net_flow: float,
                                  fii_avg_flow: float, dii_avg_flow: float,
                                  symbol: str) -> AlphaFactorResult:
        """
        Adjust order imbalance based on FII/DII activity patterns
        Higher weight during FII/DII active hours and high flow days
        """
        try:
            # Calculate flow intensity ratios
            fii_intensity = fii_net_flow / max(abs(fii_avg_flow), 1e6) if fii_avg_flow != 0 else 0
            dii_intensity = dii_net_flow / max(abs(dii_avg_flow), 1e6) if dii_avg_flow != 0 else 0
            
            # FII/DII activity multipliers (capped for stability)
            fii_multiplier = 1.0 + np.clip(0.3 * fii_intensity, -0.5, 0.5)
            dii_multiplier = 1.0 + np.clip(0.2 * dii_intensity, -0.3, 0.3)
            
            # Time-of-day adjustment for FII/DII activity
            current_time = datetime.now().time()
            if (time(9, 15) <= current_time <= time(10, 30)) or (time(14, 30) <= current_time <= time(15, 30)):
                time_multiplier = 1.2  # Higher weight during active periods
            else:
                time_multiplier = 1.0
                
            adjusted_imbalance = base_imbalance * fii_multiplier * dii_multiplier * time_multiplier
            adjusted_imbalance = np.clip(adjusted_imbalance, -1.0, 1.0)
            
            # Confidence based on flow data quality and magnitude
            confidence = min(
                abs(fii_intensity) + abs(dii_intensity),
                1.0
            ) * 0.8  # Never fully confident due to delayed flow data
            
            return AlphaFactorResult(
                value=adjusted_imbalance,
                confidence=confidence,
                timestamp=datetime.now(),
                factor_name="fii_dii_adjusted_imbalance",
                symbol=symbol,
                metadata={
                    'fii_intensity': fii_intensity,
                    'dii_intensity': dii_intensity,
                    'time_multiplier': time_multiplier
                }
            )
            
        except Exception:
            return AlphaFactorResult(
                value=base_imbalance, confidence=0.0, timestamp=datetime.now(),
                factor_name="fii_dii_adjusted_imbalance", symbol=symbol
            )

class MomentumFactors:
    """
    Momentum-based alpha factors adapted for Indian market patterns
    """
    
    def __init__(self):
        self.price_cache = defaultdict(lambda: deque(maxlen=500))
        self.volume_cache = defaultdict(lambda: deque(maxlen=500))
        self.timestamp_cache = defaultdict(lambda: deque(maxlen=500))
        
    def intraday_momentum_india(self, symbol: str, prices: List[float], 
                               volumes: List[int], timestamps: List[datetime],
                               decay_factor: float = 0.95) -> AlphaFactorResult:
        """
        Calculate intraday momentum adjusted for Indian market patterns
        Higher weights during high-activity periods
        """
        try:
            if len(prices) < 2:
                return AlphaFactorResult(
                    value=0.0, confidence=0.0, timestamp=datetime.now(),
                    factor_name="intraday_momentum", symbol=symbol
                )
            
            def get_time_weight(timestamp: datetime) -> float:
                """Get time-of-day weight for Indian markets"""
                hour = timestamp.hour
                minute = timestamp.minute
                
                # High activity periods get higher weights
                if (9 <= hour <= 10) or (14 <= hour <= 15):  # Opening and closing hours
                    return 1.5
                elif 11 <= hour <= 13:  # Mid-day low activity
                    return 0.7
                else:
                    return 1.0
            
            momentum = 0.0
            total_weight = 0.0
            
            # Calculate average volume for normalization
            avg_volume = np.mean(volumes) if volumes else 1
            
            for i in range(1, len(prices)):
                # Price change
                price_change = (prices[i] - prices[i-1]) / prices[i-1]
                
                # Volume weight
                volume_weight = volumes[i] / max(avg_volume, 1)
                
                # Time weight
                time_weight = get_time_weight(timestamps[i])
                
                # Decay weight (recent data gets higher weight)
                decay_weight = decay_factor ** (len(prices) - 1 - i)
                
                # Combined weight
                total_weight_i = volume_weight * time_weight * decay_weight
                
                momentum += price_change * total_weight_i
                total_weight += total_weight_i
            
            final_momentum = momentum / total_weight if total_weight > 0 else 0
            
            # Confidence based on data quality and consistency
            confidence = min(
                total_weight / len(prices),  # Weight quality
                min(len(prices) / 20, 1.0)   # Data sufficiency
            )
            
            return AlphaFactorResult(
                value=final_momentum,
                confidence=confidence,
                timestamp=datetime.now(),
                factor_name="intraday_momentum",
                symbol=symbol,
                metadata={'total_weight': total_weight, 'data_points': len(prices)}
            )
            
        except Exception:
            return AlphaFactorResult(
                value=0.0, confidence=0.0, timestamp=datetime.now(),
                factor_name="intraday_momentum", symbol=symbol
            )

class CircuitBreakerFactors:
    """
    Factors related to circuit breaker proximity and market stress
    """
    
    def __init__(self, market_structure: Optional[object] = None):
        self.market_structure = market_structure or IndianMarketStructure()
        
    def circuit_breaker_proximity(self, current_price: float, reference_price: float,
                                 symbol: str, circuit_percentage: float = 2.0) -> AlphaFactorResult:
        """
        Calculate proximity to circuit breaker limits
        Uses NSE/BSE specific circuit breaker percentages
        """
        try:
            upper_limit = reference_price * (1 + circuit_percentage / 100)
            lower_limit = reference_price * (1 - circuit_percentage / 100)
            
            if current_price >= reference_price:
                proximity = (current_price - reference_price) / (upper_limit - reference_price)
            else:
                proximity = (reference_price - current_price) / (reference_price - lower_limit)
            
            proximity = min(proximity, 1.0)
            
            # Confidence increases as we approach limits
            confidence = proximity
            
            return AlphaFactorResult(
                value=proximity,
                confidence=confidence,
                timestamp=datetime.now(),
                factor_name="circuit_breaker_proximity",
                symbol=symbol,
                metadata={
                    'upper_limit': upper_limit,
                    'lower_limit': lower_limit,
                    'circuit_percentage': circuit_percentage
                }
            )
            
        except Exception:
            return AlphaFactorResult(
                value=0.0, confidence=0.0, timestamp=datetime.now(),
                factor_name="circuit_breaker_proximity", symbol=symbol
            )

class MarketMakerFactors:
    """
    Factors for detecting and analyzing market maker activity
    """
    
    def __init__(self):
        self.order_flow_cache = defaultdict(lambda: deque(maxlen=1000))
        self.trade_cache = defaultdict(lambda: deque(maxlen=1000))
        
    def market_maker_presence_indicator(self, symbol: str, order_count: int, 
                                      trade_count: int, bid_posts: int, 
                                      ask_posts: int, time_window: int = 60) -> AlphaFactorResult:
        """
        Detect market maker presence and activity
        Based on order-to-trade ratios and bid-ask posting patterns
        """
        try:
            # Calculate order-to-trade ratio
            otr = order_count / max(trade_count, 1)
            
            # Analyze bid-ask posting balance
            total_posts = bid_posts + ask_posts
            posting_balance = abs(bid_posts - ask_posts) / max(total_posts, 1) if total_posts > 0 else 1.0
            
            # Market maker score (lower posting imbalance and higher OTR indicates MM activity)
            mm_score = otr * (1 - posting_balance)
            
            # Normalize to 0-1 scale (typical OTR for MM is 20-50)
            normalized_score = min(mm_score / 50.0, 1.0)
            
            # Confidence based on data sufficiency
            confidence = min(
                (order_count + trade_count) / 100,  # Sufficient activity
                1.0
            )
            
            return AlphaFactorResult(
                value=normalized_score,
                confidence=confidence,
                timestamp=datetime.now(),
                factor_name="market_maker_presence",
                symbol=symbol,
                metadata={
                    'otr': otr,
                    'posting_balance': posting_balance,
                    'bid_posts': bid_posts,
                    'ask_posts': ask_posts
                }
            )
            
        except Exception:
            return AlphaFactorResult(
                value=0.0, confidence=0.0, timestamp=datetime.now(),
                factor_name="market_maker_presence", symbol=symbol
            )

class RiskFactors:
    """
    Risk-based alpha factors for position sizing and risk management
    """
    
    def __init__(self):
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        
    def intraday_var_predictor(self, symbol: str, price_series: List[float], 
                              volume_series: List[int], 
                              confidence_level: float = 0.95) -> AlphaFactorResult:
        """
        Predict intraday Value at Risk using volume-weighted approach
        """
        try:
            if len(price_series) < 10:
                return AlphaFactorResult(
                    value=np.nan, confidence=0.0, timestamp=datetime.now(),
                    factor_name="intraday_var", symbol=symbol
                )
            
            # Calculate returns
            returns = np.diff(price_series) / np.array(price_series[:-1])
            
            # Volume-weighted returns
            if len(volume_series) > len(returns):
                volume_weights = np.array(volume_series[1:len(returns)+1])
            else:
                volume_weights = np.array(volume_series[:len(returns)])
                
            volume_weights = volume_weights / np.sum(volume_weights)
            weighted_returns = returns * volume_weights
            
            # Calculate volatility
            volatility = np.std(weighted_returns)
            
            # VaR calculation
            z_score = stats.norm.ppf(confidence_level)
            var_estimate = volatility * z_score
            
            # Confidence based on data sufficiency and stability
            confidence = min(len(returns) / 50, 1.0)
            
            return AlphaFactorResult(
                value=var_estimate,
                confidence=confidence,
                timestamp=datetime.now(),
                factor_name="intraday_var",
                symbol=symbol,
                metadata={
                    'volatility': volatility,
                    'confidence_level': confidence_level,
                    'data_points': len(returns)
                }
            )
            
        except Exception:
            return AlphaFactorResult(
                value=np.nan, confidence=0.0, timestamp=datetime.now(),
                factor_name="intraday_var", symbol=symbol
            )

class IndianAlphaFactorEngine:
    """
    Comprehensive alpha factor calculation engine for Indian markets
    Coordinates all factor calculations and provides unified interface
    """
    
    def __init__(self):
        self.microstructure_factors = IndianMicrostructureFactors()
        self.fii_dii_factors = FIIDIIFlowFactors()
        self.momentum_factors = MomentumFactors()
        self.circuit_breaker_factors = CircuitBreakerFactors()
        self.market_maker_factors = MarketMakerFactors()
        self.risk_factors = RiskFactors()
        
        # Factor update frequencies
        self.update_frequency = {
            'tick': ['vwoi', 'tick_normalized_spread', 'depth_pressure'],
            'minute': ['intraday_momentum', 'market_maker_presence'],
            'hour': ['fii_dii_adjusted_imbalance', 'circuit_breaker_proximity'],
            'daily': ['intraday_var']
        }
        
        # Factor importance weights (to be updated based on performance)
        self.factor_weights = {
            'vwoi': 0.15,
            'tick_normalized_spread': 0.10,
            'depth_pressure': 0.12,
            'intraday_momentum': 0.20,
            'fii_dii_adjusted_imbalance': 0.18,
            'market_maker_presence': 0.08,
            'circuit_breaker_proximity': 0.07,
            'intraday_var': 0.10
        }
    
    def calculate_all_factors(self, market_data: Dict, fundamental_data: Dict, 
                            symbol: str) -> Dict[str, AlphaFactorResult]:
        """
        Calculate all applicable alpha factors for given market state
        
        Args:
            market_data: Dictionary containing order book, prices, volumes, etc.
            fundamental_data: Dictionary containing FII/DII flows, sector data, etc.
            symbol: Trading symbol
            
        Returns:
            Dictionary of factor results
        """
        factors = {}
        
        try:
            # Microstructure factors (tick frequency)
            if 'order_book' in market_data:
                factors['vwoi'] = self.microstructure_factors.volume_weighted_order_imbalance(
                    market_data['order_book']
                )
                factors['tick_normalized_spread'] = self.microstructure_factors.tick_normalized_spread(
                    market_data['order_book']
                )
                factors['depth_pressure'] = self.microstructure_factors.order_book_depth_pressure(
                    market_data['order_book']
                )
            
            # Momentum factors
            if all(k in market_data for k in ['prices', 'volumes', 'timestamps']):
                factors['intraday_momentum'] = self.momentum_factors.intraday_momentum_india(
                    symbol,
                    market_data['prices'],
                    market_data['volumes'],
                    market_data['timestamps']
                )
            
            # FII/DII flow factors
            if all(k in fundamental_data for k in ['fii_flow', 'dii_flow', 'fii_avg', 'dii_avg']):
                base_imbalance = factors.get('vwoi', AlphaFactorResult(0, 0, datetime.now(), '', '')).value
                factors['fii_dii_adjusted_imbalance'] = self.fii_dii_factors.fii_dii_adjusted_imbalance(
                    base_imbalance,
                    fundamental_data['fii_flow'],
                    fundamental_data['dii_flow'],
                    fundamental_data['fii_avg'],
                    fundamental_data['dii_avg'],
                    symbol
                )
            
            # Circuit breaker factors
            if 'current_price' in market_data and 'reference_price' in market_data:
                factors['circuit_breaker_proximity'] = self.circuit_breaker_factors.circuit_breaker_proximity(
                    market_data['current_price'],
                    market_data['reference_price'],
                    symbol
                )
            
            # Market maker factors
            if all(k in market_data for k in ['order_count', 'trade_count', 'bid_posts', 'ask_posts']):
                factors['market_maker_presence'] = self.market_maker_factors.market_maker_presence_indicator(
                    symbol,
                    market_data['order_count'],
                    market_data['trade_count'],
                    market_data['bid_posts'],
                    market_data['ask_posts']
                )
            
            # Risk factors
            if 'prices' in market_data and 'volumes' in market_data:
                factors['intraday_var'] = self.risk_factors.intraday_var_predictor(
                    symbol,
                    market_data['prices'],
                    market_data['volumes']
                )
            
        except Exception as e:
            # Log error but don't fail completely
            factors['error'] = AlphaFactorResult(
                value=0.0, confidence=0.0, timestamp=datetime.now(),
                factor_name="error", symbol=symbol, metadata={'error': str(e)}
            )
        
        return factors
    
    def calculate_composite_alpha(self, factors: Dict[str, AlphaFactorResult]) -> AlphaFactorResult:
        """
        Calculate composite alpha signal from individual factors
        
        Args:
            factors: Dictionary of individual factor results
            
        Returns:
            Composite alpha factor result
        """
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            confidence_sum = 0.0
            
            for factor_name, result in factors.items():
                if factor_name in self.factor_weights and not np.isnan(result.value):
                    weight = self.factor_weights[factor_name] * result.confidence
                    weighted_sum += result.value * weight
                    total_weight += weight
                    confidence_sum += result.confidence
            
            if total_weight > 0:
                composite_alpha = weighted_sum / total_weight
                composite_confidence = confidence_sum / len([f for f in factors.values() if not np.isnan(f.value)])
            else:
                composite_alpha = 0.0
                composite_confidence = 0.0
            
            # Ensure alpha is bounded
            composite_alpha = np.clip(composite_alpha, -1.0, 1.0)
            
            return AlphaFactorResult(
                value=composite_alpha,
                confidence=composite_confidence,
                timestamp=datetime.now(),
                factor_name="composite_alpha",
                symbol=factors[list(factors.keys())[0]].symbol if factors else "UNKNOWN",
                metadata={
                    'individual_factors': len(factors),
                    'total_weight': total_weight,
                    'contributing_factors': [name for name in factors.keys() if name in self.factor_weights]
                }
            )
            
        except Exception:
            return AlphaFactorResult(
                value=0.0, confidence=0.0, timestamp=datetime.now(),
                factor_name="composite_alpha", symbol="UNKNOWN"
            )
    
    def update_factor_weights(self, performance_data: Dict[str, float]):
        """
        Update factor weights based on historical performance
        
        Args:
            performance_data: Dictionary mapping factor names to performance scores
        """
        # Simple weight update based on performance
        total_performance = sum(abs(p) for p in performance_data.values())
        
        if total_performance > 0:
            for factor_name in self.factor_weights:
                if factor_name in performance_data:
                    # Update weight based on relative performance
                    relative_performance = abs(performance_data[factor_name]) / total_performance
                    self.factor_weights[factor_name] = 0.9 * self.factor_weights[factor_name] + 0.1 * relative_performance
        
        # Normalize weights to sum to 1
        total_weight = sum(self.factor_weights.values())
        if total_weight > 0:
            for factor_name in self.factor_weights:
                self.factor_weights[factor_name] /= total_weight

# Example usage and testing
if __name__ == "__main__":
    # Initialize the alpha factor engine
    engine = IndianAlphaFactorEngine()
    
    # Simulate market data for testing
    from collections import namedtuple
    
    # Create mock order book
    OrderBookLevel = namedtuple('OrderBookLevel', ['price', 'quantity'])
    MockOrderBook = namedtuple('MockOrderBook', ['symbol', 'bid_levels', 'ask_levels'])
    
    def mock_get_best_bid_ask(self):
        best_bid = self.bid_levels[0].price if self.bid_levels else None
        best_ask = self.ask_levels[0].price if self.ask_levels else None
        return best_bid, best_ask
    
    MockOrderBook.get_best_bid_ask = mock_get_best_bid_ask
    
    order_book = MockOrderBook(
        symbol="RELIANCE",
        bid_levels=[
            OrderBookLevel(2500.0, 100),
            OrderBookLevel(2499.5, 200),
            OrderBookLevel(2499.0, 150)
        ],
        ask_levels=[
            OrderBookLevel(2500.5, 80),
            OrderBookLevel(2501.0, 120),
            OrderBookLevel(2501.5, 90)
        ]
    )
    
    # Mock market data
    market_data = {
        'order_book': order_book,
        'prices': [2499.0, 2500.0, 2501.0, 2500.5, 2502.0],
        'volumes': [1000, 1200, 800, 1500, 900],
        'timestamps': [datetime.now() for _ in range(5)],
        'current_price': 2500.5,
        'reference_price': 2500.0,
        'order_count': 150,
        'trade_count': 25,
        'bid_posts': 75,
        'ask_posts': 70
    }
    
    # Mock fundamental data
    fundamental_data = {
        'fii_flow': 1000000000,  # ₹100 crores net buying
        'dii_flow': -500000000,  # ₹50 crores net selling
        'fii_avg': 800000000,    # ₹80 crores average
        'dii_avg': 600000000     # ₹60 crores average
    }
    
    # Calculate all factors
    print("Testing Indian Alpha Factor Engine...")
    factors = engine.calculate_all_factors(market_data, fundamental_data, "RELIANCE")
    
    # Display results
    print("\nIndividual Factor Results:")
    for name, result in factors.items():
        print(f"{name:25s}: {result.value:8.4f} (confidence: {result.confidence:.2f})")
    
    # Calculate composite alpha
    composite = engine.calculate_composite_alpha(factors)
    print(f"\nComposite Alpha: {composite.value:.4f} (confidence: {composite.confidence:.2f})")
    
    # Test factor weight updates
    mock_performance = {
        'vwoi': 0.15,
        'intraday_momentum': 0.25,
        'fii_dii_adjusted_imbalance': 0.20,
        'tick_normalized_spread': 0.08
    }
    
    print("\nOriginal factor weights:")
    for name, weight in engine.factor_weights.items():
        print(f"{name:25s}: {weight:.3f}")
    
    engine.update_factor_weights(mock_performance)
    
    print("\nUpdated factor weights:")
    for name, weight in engine.factor_weights.items():
        print(f"{name:25s}: {weight:.3f}")