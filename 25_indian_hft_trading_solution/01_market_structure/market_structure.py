"""
Indian Market Structure Analysis Module
Provides detailed analysis and utilities for NSE/BSE market microstructure
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class Exchange(Enum):
    NSE = "NSE"
    BSE = "BSE"

class OrderType(Enum):
    LIMIT = "LMT"
    MARKET = "MKT" 
    STOP_LOSS = "SL"
    IMMEDIATE_OR_CANCEL = "IOC"
    FILL_OR_KILL = "FOK"

@dataclass
class TickSize:
    price_min: float
    price_max: float
    tick_size: float
    typical_spread_pct: Tuple[float, float]

@dataclass
class CircuitBreaker:
    percentage: float
    halt_duration_minutes: int
    description: str

@dataclass
class TradingSession:
    name: str
    start_time: time
    end_time: time
    description: str

class IndianMarketStructure:
    """
    Comprehensive market structure analysis for Indian exchanges
    """
    
    def __init__(self):
        self.nse_tick_sizes = self._initialize_nse_tick_sizes()
        self.bse_tick_sizes = self._initialize_bse_tick_sizes() 
        self.circuit_breakers = self._initialize_circuit_breakers()
        self.trading_sessions = self._initialize_trading_sessions()
        self.lot_sizes = self._initialize_lot_sizes()
        
    def _initialize_nse_tick_sizes(self) -> List[TickSize]:
        """NSE tick size structure"""
        return [
            TickSize(0.00, 2.00, 0.0025, (0.5, 2.0)),
            TickSize(2.00, 5.00, 0.0050, (0.2, 0.8)),
            TickSize(5.00, 10.00, 0.0100, (0.1, 0.5)),
            TickSize(10.00, 20.00, 0.0200, (0.05, 0.3)),
            TickSize(20.00, 50.00, 0.0500, (0.02, 0.2)),
            TickSize(50.00, 100.00, 0.1000, (0.01, 0.15)),
            TickSize(100.00, 500.00, 0.2000, (0.005, 0.1)),
            TickSize(500.00, float('inf'), 1.0000, (0.002, 0.05))
        ]
    
    def _initialize_bse_tick_sizes(self) -> List[TickSize]:
        """BSE tick size structure (similar to NSE with minor variations)"""
        return self._initialize_nse_tick_sizes()  # Simplified for this example
    
    def _initialize_circuit_breakers(self) -> Dict[str, List[CircuitBreaker]]:
        """Circuit breaker configuration"""
        return {
            'market_wide': [
                CircuitBreaker(10.0, 15, "15-minute halt on 10% decline"),
                CircuitBreaker(15.0, 60, "1-hour halt on 15% decline"),
                CircuitBreaker(20.0, 1440, "Trading halted for the day")
            ],
            'individual_stock': [
                CircuitBreaker(2.0, 0, "2% band for most liquid stocks"),
                CircuitBreaker(5.0, 0, "5% band for moderately liquid stocks"),
                CircuitBreaker(10.0, 0, "10% band for less liquid stocks"),
                CircuitBreaker(20.0, 0, "20% band for illiquid stocks")
            ]
        }
    
    def _initialize_trading_sessions(self) -> Dict[str, List[TradingSession]]:
        """Trading session configuration"""
        return {
            'NSE': [
                TradingSession("Pre-opening", time(9, 0), time(9, 15), "Order entry and matching"),
                TradingSession("Normal Market", time(9, 15), time(15, 30), "Continuous trading"),
                TradingSession("Post-closing", time(15, 40), time(16, 0), "Odd lot trading")
            ],
            'BSE': [
                TradingSession("Pre-opening", time(9, 0), time(9, 15), "Order entry and matching"),
                TradingSession("Normal Market", time(9, 15), time(15, 30), "Continuous trading"),
                TradingSession("Post-closing", time(15, 40), time(16, 0), "Odd lot trading")
            ]
        }
    
    def _initialize_lot_sizes(self) -> Dict[str, int]:
        """Standard lot sizes for derivatives"""
        return {
            'NIFTY': 50,
            'BANKNIFTY': 25,
            'NIFTYMIDCAP': 75,
            'NIFTYIT': 50,
            'NIFTYPHARMA': 50,
            'NIFTYAUTO': 50,
            'NIFTYMETAL': 50,
            'NIFTYFMCG': 50
        }
    
    def get_tick_size(self, price: float, exchange: Exchange = Exchange.NSE) -> float:
        """
        Get appropriate tick size for given price
        
        Args:
            price: Current stock price
            exchange: NSE or BSE
            
        Returns:
            Applicable tick size
        """
        tick_sizes = self.nse_tick_sizes if exchange == Exchange.NSE else self.bse_tick_sizes
        
        for tick_info in tick_sizes:
            if tick_info.price_min <= price < tick_info.price_max:
                return tick_info.tick_size
        
        return tick_sizes[-1].tick_size  # Return largest tick size for very high prices
    
    def get_typical_spread(self, price: float, exchange: Exchange = Exchange.NSE) -> Tuple[float, float]:
        """
        Get typical spread range for given price
        
        Args:
            price: Current stock price
            exchange: NSE or BSE
            
        Returns:
            Tuple of (min_spread_pct, max_spread_pct)
        """
        tick_sizes = self.nse_tick_sizes if exchange == Exchange.NSE else self.bse_tick_sizes
        
        for tick_info in tick_sizes:
            if tick_info.price_min <= price < tick_info.price_max:
                return tick_info.typical_spread_pct
        
        return tick_sizes[-1].typical_spread_pct
    
    def is_trading_session_active(self, current_time: datetime, 
                                 session_type: str = "Normal Market",
                                 exchange: Exchange = Exchange.NSE) -> bool:
        """
        Check if specified trading session is currently active
        
        Args:
            current_time: Current datetime
            session_type: Type of session to check
            exchange: NSE or BSE
            
        Returns:
            True if session is active
        """
        sessions = self.trading_sessions[exchange.value]
        current_time_only = current_time.time()
        
        for session in sessions:
            if (session.name == session_type and 
                session.start_time <= current_time_only <= session.end_time):
                return True
        
        return False
    
    def get_optimal_trading_periods(self) -> Dict[str, List[Tuple[time, time, str]]]:
        """
        Get optimal trading periods for HFT strategies
        
        Returns:
            Dictionary with high/low activity periods
        """
        return {
            'high_activity': [
                (time(9, 15), time(9, 45), "Opening volatility - 35% daily volume"),
                (time(14, 30), time(15, 30), "Closing session - 25% daily volume"),
                (time(11, 0), time(11, 30), "Mid-morning rebalancing"),
                (time(14, 0), time(14, 30), "Afternoon institutional activity")
            ],
            'low_activity': [
                (time(10, 30), time(11, 0), "Post-opening lull"),
                (time(12, 0), time(13, 0), "Lunch hour reduced activity"),
                (time(13, 30), time(14, 0), "Afternoon quiet period")
            ]
        }
    
    def calculate_impact_cost(self, price: float, quantity: int, 
                            avg_daily_volume: int, exchange: Exchange = Exchange.NSE) -> float:
        """
        Estimate market impact cost for order
        
        Args:
            price: Current stock price
            quantity: Order quantity
            avg_daily_volume: Average daily volume
            exchange: NSE or BSE
            
        Returns:
            Estimated impact cost in basis points
        """
        # Simplified impact model - real implementation would be more sophisticated
        volume_participation = quantity / avg_daily_volume
        
        if volume_participation <= 0.001:  # < 0.1% of daily volume
            return 1.0  # 1 basis point
        elif volume_participation <= 0.005:  # < 0.5% of daily volume
            return 5.0  # 5 basis points
        elif volume_participation <= 0.01:   # < 1% of daily volume
            return 15.0  # 15 basis points
        else:
            return 50.0 + (volume_participation - 0.01) * 1000  # Higher for large orders
    
    def get_regulatory_limits(self) -> Dict[str, Dict]:
        """
        Get SEBI regulatory limits for algorithmic trading
        
        Returns:
            Dictionary of regulatory constraints
        """
        return {
            'position_limits': {
                'single_stock_pct': 1.0,  # 1% of market cap
                'single_stock_amount': 50000000000,  # ₹500 crores
                'portfolio_pct': 5.0,  # 5% of AUM
                'daily_volume_pct': 10.0  # 10% of average daily volume
            },
            'velocity_limits': {
                'max_orders_per_second': 100,  # Per symbol
                'max_otr_across_symbols': 500,  # Order-to-trade ratio
                'max_otr_per_symbol': 50  # For liquid stocks
            },
            'risk_controls': {
                'price_band_pct': 20.0,  # ±20% from LTP
                'kill_switch_mandatory': True,
                'real_time_monitoring': True,
                'audit_trail_years': 7
            }
        }

class LatencyProfiler:
    """
    Models latency characteristics of Indian exchanges
    """
    
    def __init__(self):
        self.nse_latency_profile = {
            'colocation': {
                'order_to_execution': (50, 150),  # microseconds (min, max)
                'market_data': (10, 50),
                'cross_connect': (1, 5)
            },
            'external': {
                'order_to_execution': (200, 500),
                'market_data': (100, 300),
                'network_latency': (50, 200)
            }
        }
        
        self.bse_latency_profile = {
            'colocation': {
                'order_to_execution': (80, 200),
                'market_data': (20, 80),
                'cross_connect': (2, 8)
            },
            'external': {
                'order_to_execution': (300, 700),
                'market_data': (150, 400),
                'network_latency': (80, 300)
            }
        }
    
    def estimate_latency(self, exchange: Exchange, connection_type: str, 
                        latency_type: str) -> Tuple[float, float]:
        """
        Estimate latency for specific exchange and connection type
        
        Args:
            exchange: NSE or BSE
            connection_type: 'colocation' or 'external'
            latency_type: Type of latency to estimate
            
        Returns:
            Tuple of (min_latency, max_latency) in microseconds
        """
        if exchange == Exchange.NSE:
            profile = self.nse_latency_profile
        else:
            profile = self.bse_latency_profile
            
        return profile.get(connection_type, {}).get(latency_type, (1000, 2000))
    
    def calculate_round_trip_latency(self, exchange: Exchange, 
                                   connection_type: str) -> float:
        """
        Calculate estimated round-trip latency
        
        Args:
            exchange: NSE or BSE
            connection_type: 'colocation' or 'external'
            
        Returns:
            Estimated round-trip latency in microseconds
        """
        order_latency = self.estimate_latency(exchange, connection_type, 'order_to_execution')
        market_data_latency = self.estimate_latency(exchange, connection_type, 'market_data')
        
        # Simplified calculation - real implementation would be more complex
        return order_latency[1] + market_data_latency[1]

class ComplianceChecker:
    """
    SEBI compliance validation for HFT strategies
    """
    
    def __init__(self):
        self.market_structure = IndianMarketStructure()
        self.regulatory_limits = self.market_structure.get_regulatory_limits()
    
    def validate_order(self, order: Dict) -> Tuple[bool, List[str]]:
        """
        Validate order against SEBI compliance requirements
        
        Args:
            order: Order dictionary with required fields
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check price band
        if 'price' in order and 'reference_price' in order:
            price_deviation = abs(order['price'] - order['reference_price']) / order['reference_price']
            max_deviation = self.regulatory_limits['risk_controls']['price_band_pct'] / 100
            
            if price_deviation > max_deviation:
                violations.append(f"Price deviation {price_deviation:.2%} exceeds limit {max_deviation:.2%}")
        
        # Check quantity limits
        if 'quantity' in order and 'avg_daily_volume' in order:
            volume_pct = order['quantity'] / order['avg_daily_volume']
            max_volume_pct = self.regulatory_limits['position_limits']['daily_volume_pct'] / 100
            
            if volume_pct > max_volume_pct:
                violations.append(f"Order size {volume_pct:.2%} of daily volume exceeds limit {max_volume_pct:.2%}")
        
        return len(violations) == 0, violations
    
    def check_velocity_limits(self, orders_per_second: int, symbol: str) -> bool:
        """
        Check if order velocity is within SEBI limits
        
        Args:
            orders_per_second: Current order rate
            symbol: Trading symbol
            
        Returns:
            True if within limits
        """
        max_orders = self.regulatory_limits['velocity_limits']['max_orders_per_second']
        return orders_per_second <= max_orders

# Example usage and testing
if __name__ == "__main__":
    # Initialize market structure analyzer
    market = IndianMarketStructure()
    latency = LatencyProfiler()
    compliance = ComplianceChecker()
    
    # Test tick size calculation
    print("NSE Tick Size Examples:")
    test_prices = [1.50, 5.75, 15.30, 75.20, 250.40, 750.60]
    for price in test_prices:
        tick = market.get_tick_size(price, Exchange.NSE)
        spread = market.get_typical_spread(price, Exchange.NSE)
        print(f"Price: ₹{price:6.2f} | Tick: ₹{tick:6.4f} | Spread: {spread[0]:.2%}-{spread[1]:.2%}")
    
    print("\nLatency Estimates:")
    for exchange in [Exchange.NSE, Exchange.BSE]:
        for conn_type in ['colocation', 'external']:
            rtt = latency.calculate_round_trip_latency(exchange, conn_type)
            print(f"{exchange.value} {conn_type}: {rtt:.0f} μs round-trip")
    
    print("\nCompliance Check Example:")
    test_order = {
        'price': 105.0,
        'reference_price': 100.0,
        'quantity': 1000,
        'avg_daily_volume': 100000
    }
    
    is_valid, violations = compliance.validate_order(test_order)
    print(f"Order valid: {is_valid}")
    if violations:
        print("Violations:", violations)