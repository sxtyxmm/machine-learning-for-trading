"""
Ultra-Low Latency Execution System for Indian HFT Trading
Smart order routing, execution, and market making for NSE/BSE
"""

import time
import threading
import queue
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class Venue(Enum):
    NSE = "NSE"
    BSE = "BSE"

@dataclass
class Order:
    """Standardized order representation"""
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    order_type: OrderType
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    venue: Optional[Venue] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    submit_time: Optional[float] = None
    fill_time: Optional[float] = None
    cancel_time: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Execution result container"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    filled_quantity: int
    avg_price: float
    venue: Venue
    execution_time_ms: float
    slippage_bps: Optional[float]
    status: OrderStatus
    fees: float = 0.0
    error_message: Optional[str] = None

class VenueConnector:
    """Base class for exchange connectivity"""
    
    def __init__(self, venue: Venue):
        self.venue = venue
        self.connected = False
        self.latency_stats = deque(maxlen=1000)
        
    def connect(self) -> bool:
        """Establish connection to exchange"""
        # Simulate connection
        self.connected = True
        return True
    
    def disconnect(self):
        """Disconnect from exchange"""
        self.connected = False
    
    def submit_order(self, order: Order) -> ExecutionResult:
        """Submit order to exchange"""
        if not self.connected:
            return ExecutionResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=0,
                avg_price=0.0,
                venue=self.venue,
                execution_time_ms=0.0,
                slippage_bps=None,
                status=OrderStatus.REJECTED,
                error_message="Not connected to exchange"
            )
        
        start_time = time.time_ns()
        
        # Simulate order processing
        result = self._process_order(order)
        
        execution_time_ms = (time.time_ns() - start_time) / 1_000_000
        result.execution_time_ms = execution_time_ms
        
        self.latency_stats.append(execution_time_ms)
        
        return result
    
    def _process_order(self, order: Order) -> ExecutionResult:
        """Process order execution (venue-specific implementation)"""
        # Simulate different latencies for NSE vs BSE
        if self.venue == Venue.NSE:
            latency_base_us = random.uniform(50, 150)  # 50-150 microseconds
        else:  # BSE
            latency_base_us = random.uniform(80, 200)  # 80-200 microseconds
        
        # Add market impact delay for larger orders
        size_factor = min(order.quantity / 1000, 5.0)
        market_impact_us = size_factor * random.uniform(10, 50)
        
        total_latency_us = latency_base_us + market_impact_us
        time.sleep(total_latency_us / 1_000_000)  # Convert to seconds
        
        # Simulate fill probability (higher for smaller orders)
        fill_probability = max(0.85, 1.0 - (order.quantity / 10000) * 0.3)
        
        if random.random() < fill_probability:
            # Full fill
            filled_quantity = order.quantity
            status = OrderStatus.FILLED
        else:
            # Partial fill or rejection
            if random.random() < 0.6:  # 60% chance of partial fill
                filled_quantity = random.randint(order.quantity // 4, order.quantity * 3 // 4)
                status = OrderStatus.PARTIALLY_FILLED
            else:
                filled_quantity = 0
                status = OrderStatus.REJECTED
        
        # Calculate slippage (simulate market impact)
        if filled_quantity > 0:
            slippage_bps = random.uniform(0.5, 3.0) * (order.quantity / 1000)
            slippage_bps = min(slippage_bps, 20.0)  # Cap at 20 bps
        else:
            slippage_bps = None
        
        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=filled_quantity,
            avg_price=order.price,
            venue=self.venue,
            execution_time_ms=0.0,  # Will be set by caller
            slippage_bps=slippage_bps,
            status=status
        )

class NSEConnector(VenueConnector):
    """NSE-specific connector with optimizations"""
    
    def __init__(self):
        super().__init__(Venue.NSE)
        self.order_rate_limit = 1000  # Orders per second
        self.last_order_time = 0
        
    def submit_order(self, order: Order) -> ExecutionResult:
        """NSE-specific order submission with rate limiting"""
        current_time = time.time()
        
        # Rate limiting
        time_since_last = current_time - self.last_order_time
        min_interval = 1.0 / self.order_rate_limit
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_order_time = time.time()
        
        return super().submit_order(order)

class BSEConnector(VenueConnector):
    """BSE-specific connector"""
    
    def __init__(self):
        super().__init__(Venue.BSE)
        self.order_rate_limit = 500  # Lower rate limit for BSE

class SmartOrderRouter:
    """
    Intelligent order routing between NSE and BSE
    """
    
    def __init__(self):
        self.connectors = {
            Venue.NSE: NSEConnector(),
            Venue.BSE: BSEConnector()
        }
        
        # Venue selection parameters
        self.venue_preferences = defaultdict(lambda: {'NSE': 0.7, 'BSE': 0.3})
        self.execution_history = defaultdict(list)
        self.venue_performance = defaultdict(lambda: defaultdict(list))
        
        # Market data cache
        self.market_data_cache = {}
        self.last_market_update = {}
        
    def connect_all(self):
        """Connect to all venues"""
        for connector in self.connectors.values():
            connector.connect()
    
    def disconnect_all(self):
        """Disconnect from all venues"""
        for connector in self.connectors.values():
            connector.disconnect()
    
    def route_order(self, order: Order) -> ExecutionResult:
        """
        Route order to optimal venue
        """
        # Select venue
        selected_venue = self._select_venue(order)
        order.venue = selected_venue
        
        # Get connector
        connector = self.connectors.get(selected_venue)
        if not connector:
            return ExecutionResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=0,
                avg_price=0.0,
                venue=selected_venue,
                execution_time_ms=0.0,
                slippage_bps=None,
                status=OrderStatus.REJECTED,
                error_message=f"No connector for venue {selected_venue}"
            )
        
        # Execute order
        result = connector.submit_order(order)
        
        # Update performance tracking
        self._update_performance_tracking(result)
        
        return result
    
    def _select_venue(self, order: Order) -> Venue:
        """
        Select optimal venue based on multiple factors
        """
        nse_score = self._calculate_venue_score(Venue.NSE, order)
        bse_score = self._calculate_venue_score(Venue.BSE, order)
        
        # Add randomization to avoid predictability
        nse_score += random.uniform(-0.05, 0.05)
        bse_score += random.uniform(-0.05, 0.05)
        
        return Venue.NSE if nse_score > bse_score else Venue.BSE
    
    def _calculate_venue_score(self, venue: Venue, order: Order) -> float:
        """
        Calculate venue score based on historical performance and current conditions
        """
        score = 0.0
        
        # Base preference (NSE typically preferred for liquidity)
        if venue == Venue.NSE:
            score += 0.6
        else:
            score += 0.4
        
        # Historical fill rate factor
        symbol_history = self.venue_performance[order.symbol][venue.value]
        if symbol_history:
            recent_fills = symbol_history[-20:]  # Last 20 orders
            fill_rate = sum(1 for h in recent_fills if h['status'] == OrderStatus.FILLED) / len(recent_fills)
            score += 0.3 * fill_rate
        
        # Latency factor
        connector = self.connectors.get(venue)
        if connector and connector.latency_stats:
            avg_latency = np.mean(list(connector.latency_stats))
            # Lower latency = higher score (inverse relationship)
            latency_score = max(0, 1.0 - (avg_latency / 1000.0))  # Normalize by 1 second
            score += 0.1 * latency_score
        
        return score
    
    def _update_performance_tracking(self, result: ExecutionResult):
        """Update venue performance tracking"""
        venue_key = result.venue.value
        symbol_key = result.symbol
        
        performance_record = {
            'timestamp': time.time(),
            'status': result.status,
            'execution_time_ms': result.execution_time_ms,
            'slippage_bps': result.slippage_bps,
            'filled_quantity': result.filled_quantity,
            'total_quantity': result.quantity
        }
        
        self.venue_performance[symbol_key][venue_key].append(performance_record)
        
        # Keep only recent history (last 100 orders per symbol per venue)
        if len(self.venue_performance[symbol_key][venue_key]) > 100:
            self.venue_performance[symbol_key][venue_key] = \
                self.venue_performance[symbol_key][venue_key][-100:]

class OrderSplittingEngine:
    """
    SEBI-compliant order splitting with multiple strategies
    """
    
    def __init__(self):
        self.max_order_size_pct = 0.10  # Max 10% of daily volume
        self.min_order_size = 1
        self.default_time_horizon = 600  # 10 minutes
        
    def split_order(self, parent_order: Order, strategy: str = 'TWAP', 
                   time_horizon: int = None, max_participation: float = 0.20) -> List[Order]:
        """
        Split large order into smaller child orders
        
        Args:
            parent_order: Original large order
            strategy: Splitting strategy ('TWAP', 'VWAP', 'POV', 'URGENT')
            time_horizon: Time horizon in seconds
            max_participation: Maximum participation rate in volume
            
        Returns:
            List of child orders
        """
        if time_horizon is None:
            time_horizon = self.default_time_horizon
            
        # Determine if splitting is needed
        if not self._needs_splitting(parent_order):
            return [parent_order]
        
        if strategy == 'TWAP':
            return self._twap_split(parent_order, time_horizon)
        elif strategy == 'VWAP':
            return self._vwap_split(parent_order, time_horizon)
        elif strategy == 'POV':
            return self._pov_split(parent_order, max_participation)
        elif strategy == 'URGENT':
            return self._urgent_split(parent_order)
        else:
            return self._twap_split(parent_order, time_horizon)
    
    def _needs_splitting(self, order: Order) -> bool:
        """Determine if order needs to be split"""
        # Simple heuristic: split if order is larger than 1000 shares
        return order.quantity > 1000
    
    def _twap_split(self, order: Order, time_horizon: int) -> List[Order]:
        """Time-Weighted Average Price splitting"""
        intervals = min(time_horizon // 30, 20)  # Max 20 intervals, 30 seconds each
        if intervals <= 1:
            return [order]
        
        child_orders = []
        base_size = order.quantity // intervals
        remainder = order.quantity % intervals
        
        current_time = time.time()
        
        for i in range(intervals):
            size = base_size + (1 if i < remainder else 0)
            if size > 0:
                child_order = Order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=size,
                    price=order.price,
                    order_type=order.order_type,
                    parent_id=order.order_id,
                    metadata={
                        'strategy': 'TWAP',
                        'execution_time': current_time + (i * time_horizon / intervals),
                        'interval': i + 1,
                        'total_intervals': intervals
                    }
                )
                child_orders.append(child_order)
        
        return child_orders
    
    def _vwap_split(self, order: Order, time_horizon: int) -> List[Order]:
        """Volume-Weighted Average Price splitting"""
        # Simulate volume profile (in production, would use historical data)
        intervals = min(time_horizon // 60, 10)  # 1-minute intervals
        volume_profile = self._get_volume_profile(order.symbol, intervals)
        
        if not volume_profile or sum(volume_profile) == 0:
            return self._twap_split(order, time_horizon)
        
        child_orders = []
        total_volume = sum(volume_profile)
        current_time = time.time()
        
        for i, period_volume in enumerate(volume_profile):
            if period_volume > 0:
                size_fraction = period_volume / total_volume
                size = max(1, int(order.quantity * size_fraction))
                
                child_order = Order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=size,
                    price=order.price,
                    order_type=order.order_type,
                    parent_id=order.order_id,
                    metadata={
                        'strategy': 'VWAP',
                        'execution_time': current_time + (i * 60),  # 60-second intervals
                        'interval': i + 1,
                        'volume_weight': size_fraction
                    }
                )
                child_orders.append(child_order)
        
        return child_orders
    
    def _get_volume_profile(self, symbol: str, intervals: int) -> List[float]:
        """Get historical volume profile for symbol"""
        # Simulate typical intraday volume pattern
        # Higher volume at open/close, lower during mid-day
        profile = []
        for i in range(intervals):
            if i < 2 or i >= intervals - 2:  # First/last 2 intervals
                volume = random.uniform(0.15, 0.25)
            else:  # Middle intervals
                volume = random.uniform(0.05, 0.15)
            profile.append(volume)
        
        return profile

class AntiGamingEngine:
    """
    Protection against predatory trading algorithms
    """
    
    def __init__(self):
        self.order_patterns = defaultdict(lambda: deque(maxlen=1000))
        self.gaming_scores = defaultdict(float)
        self.alert_threshold = 0.6
        
        # Gaming pattern detectors
        self.detectors = {
            'order_stuffing': self._detect_order_stuffing,
            'quote_stuffing': self._detect_quote_stuffing,
            'layering': self._detect_layering,
            'momentum_ignition': self._detect_momentum_ignition
        }
        
    def analyze_order_flow(self, symbol: str, recent_orders: List[Dict], 
                          market_data: Dict) -> Tuple[float, List[str]]:
        """
        Analyze recent order flow for gaming patterns
        
        Returns:
            Tuple of (gaming_score, detected_patterns)
        """
        gaming_score = 0.0
        detected_patterns = []
        
        # Update order patterns
        for order in recent_orders:
            self.order_patterns[symbol].append({
                'timestamp': order.get('timestamp', time.time()),
                'side': order.get('side', ''),
                'quantity': order.get('quantity', 0),
                'price': order.get('price', 0),
                'status': order.get('status', ''),
                'order_type': order.get('order_type', '')
            })
        
        # Run detection algorithms
        for pattern_name, detector in self.detectors.items():
            try:
                score, detected = detector(symbol, market_data)
                gaming_score += score
                if detected:
                    detected_patterns.append(pattern_name)
            except Exception:
                continue  # Skip failed detectors
        
        # Update gaming score for symbol
        self.gaming_scores[symbol] = gaming_score
        
        return gaming_score, detected_patterns
    
    def _detect_order_stuffing(self, symbol: str, market_data: Dict) -> Tuple[float, bool]:
        """Detect order stuffing patterns"""
        orders = list(self.order_patterns[symbol])
        if len(orders) < 10:
            return 0.0, False
        
        # Analyze last 60 seconds
        cutoff_time = time.time() - 60
        recent_orders = [o for o in orders if o['timestamp'] > cutoff_time]
        
        if len(recent_orders) < 10:
            return 0.0, False
        
        # Calculate cancellation rate
        cancelled_orders = sum(1 for o in recent_orders if o['status'] == 'CANCELLED')
        cancel_rate = cancelled_orders / len(recent_orders)
        
        # High order rate with high cancellation suggests stuffing
        order_rate = len(recent_orders) / 60  # Orders per second
        
        if cancel_rate > 0.7 and order_rate > 5:
            return 0.8, True
        elif cancel_rate > 0.5 and order_rate > 10:
            return 0.6, True
        
        return cancel_rate * 0.3, False
    
    def _detect_layering(self, symbol: str, market_data: Dict) -> Tuple[float, bool]:
        """Detect layering/spoofing patterns"""
        orders = list(self.order_patterns[symbol])
        if len(orders) < 20:
            return 0.0, False
        
        # Analyze last 120 seconds
        cutoff_time = time.time() - 120
        recent_orders = [o for o in orders if o['timestamp'] > cutoff_time]
        
        # Separate by side
        buy_orders = [o for o in recent_orders if o['side'] == 'BUY']
        sell_orders = [o for o in recent_orders if o['side'] == 'SELL']
        
        # Calculate volume imbalance
        buy_volume = sum(o['quantity'] for o in buy_orders)
        sell_volume = sum(o['quantity'] for o in sell_orders)
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0, False
        
        imbalance = abs(buy_volume - sell_volume) / total_volume
        
        # Check execution rate
        filled_orders = [o for o in recent_orders if o['status'] == 'FILLED']
        execution_rate = len(filled_orders) / len(recent_orders) if recent_orders else 0
        
        # High imbalance with low execution suggests layering
        if imbalance > 0.8 and execution_rate < 0.2:
            return 0.7, True
        elif imbalance > 0.6 and execution_rate < 0.3:
            return 0.4, True
        
        return imbalance * 0.2, False
    
    def _detect_quote_stuffing(self, symbol: str, market_data: Dict) -> Tuple[float, bool]:
        """Detect quote stuffing patterns"""
        # Simplified detection based on order frequency
        orders = list(self.order_patterns[symbol])
        if len(orders) < 50:
            return 0.0, False
        
        # Check for very high order frequency
        cutoff_time = time.time() - 30  # Last 30 seconds
        recent_orders = [o for o in orders if o['timestamp'] > cutoff_time]
        
        order_frequency = len(recent_orders) / 30  # Orders per second
        
        if order_frequency > 20:  # More than 20 orders per second
            return 0.9, True
        elif order_frequency > 10:
            return 0.5, True
        
        return min(order_frequency / 20, 0.4), False
    
    def _detect_momentum_ignition(self, symbol: str, market_data: Dict) -> Tuple[float, bool]:
        """Detect momentum ignition patterns"""
        orders = list(self.order_patterns[symbol])
        if len(orders) < 20:
            return 0.0, False
        
        # Look for sudden large orders followed by many small orders in same direction
        cutoff_time = time.time() - 180  # Last 3 minutes
        recent_orders = [o for o in orders if o['timestamp'] > cutoff_time]
        
        if len(recent_orders) < 10:
            return 0.0, False
        
        # Sort by timestamp
        recent_orders.sort(key=lambda x: x['timestamp'])
        
        # Look for initial large order
        large_orders = [o for o in recent_orders[:5] if o['quantity'] > 1000]
        if not large_orders:
            return 0.0, False
        
        # Check if followed by many smaller orders in same direction
        first_large = large_orders[0]
        subsequent_orders = recent_orders[5:]
        same_direction = [o for o in subsequent_orders 
                         if o['side'] == first_large['side'] and o['quantity'] < 500]
        
        if len(same_direction) > 10:
            return 0.6, True
        
        return 0.0, False
    
    def should_delay_order(self, symbol: str, order: Order) -> Tuple[bool, float]:
        """
        Determine if order should be delayed due to gaming activity
        
        Returns:
            Tuple of (should_delay, delay_milliseconds)
        """
        gaming_score = self.gaming_scores.get(symbol, 0.0)
        
        if gaming_score > self.alert_threshold:
            # Add random delay between 10-100ms
            delay_ms = random.uniform(10, 100)
            return True, delay_ms
        
        return False, 0.0

class RiskManager:
    """
    Pre-trade and real-time risk management
    """
    
    def __init__(self):
        # Position limits
        self.max_position_per_symbol = 10000
        self.max_portfolio_value = 100_000_000  # ₹10 crores
        self.max_order_value = 5_000_000  # ₹50 lakhs per order
        
        # Current positions
        self.positions = defaultdict(int)
        self.position_values = defaultdict(float)
        
        # Risk metrics
        self.var_limits = defaultdict(lambda: 500_000)  # ₹5 lakhs VaR per symbol
        self.daily_loss_limit = 2_000_000  # ₹20 lakhs daily loss limit
        self.current_pnl = 0.0
        
    def validate_order(self, order: Order) -> Dict[str, Union[bool, str]]:
        """
        Validate order against risk limits
        
        Returns:
            Dictionary with 'approved' (bool) and 'reason' (str) keys
        """
        # Order value check
        order_value = order.quantity * order.price
        if order_value > self.max_order_value:
            return {
                'approved': False,
                'reason': f'Order value {order_value:,.0f} exceeds limit {self.max_order_value:,.0f}'
            }
        
        # Position limit check
        current_position = self.positions[order.symbol]
        
        if order.side == OrderSide.BUY:
            new_position = current_position + order.quantity
        else:
            new_position = current_position - order.quantity
        
        if abs(new_position) > self.max_position_per_symbol:
            return {
                'approved': False,
                'reason': f'Position limit exceeded for {order.symbol}'
            }
        
        # Portfolio value check
        current_portfolio_value = sum(abs(val) for val in self.position_values.values())
        if current_portfolio_value + order_value > self.max_portfolio_value:
            return {
                'approved': False,
                'reason': 'Portfolio value limit exceeded'
            }
        
        # Daily loss limit check
        if self.current_pnl < -self.daily_loss_limit:
            return {
                'approved': False,
                'reason': 'Daily loss limit exceeded'
            }
        
        # All checks passed
        return {'approved': True, 'reason': 'Order approved'}
    
    def update_position(self, execution_result: ExecutionResult):
        """Update positions after order execution"""
        if execution_result.status == OrderStatus.FILLED:
            symbol = execution_result.symbol
            
            if execution_result.side == OrderSide.BUY:
                self.positions[symbol] += execution_result.filled_quantity
            else:
                self.positions[symbol] -= execution_result.filled_quantity
            
            # Update position value
            position_value = self.positions[symbol] * execution_result.avg_price
            self.position_values[symbol] = position_value

class ExecutionEngine:
    """
    Main execution engine coordinating all components
    """
    
    def __init__(self):
        self.router = SmartOrderRouter()
        self.splitter = OrderSplittingEngine()
        self.anti_gaming = AntiGamingEngine()
        self.risk_manager = RiskManager()
        
        # Order management
        self.active_orders = {}
        self.order_history = []
        self.execution_queue = queue.PriorityQueue()
        
        # Performance tracking
        self.execution_stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'total_execution_time_ms': 0,
            'fill_rate': 0.0,
            'avg_slippage_bps': 0.0
        }
        
        # Threading
        self.execution_thread = None
        self.running = False
        
    def start(self):
        """Start the execution engine"""
        self.router.connect_all()
        self.running = True
        self.execution_thread = threading.Thread(target=self._execution_loop)
        self.execution_thread.start()
    
    def stop(self):
        """Stop the execution engine"""
        self.running = False
        if self.execution_thread:
            self.execution_thread.join()
        self.router.disconnect_all()
    
    def submit_order(self, order: Order) -> str:
        """
        Submit order for execution
        
        Returns:
            Order ID
        """
        # Generate unique order ID if not provided
        if not order.order_id:
            order.order_id = str(uuid.uuid4())
        
        order.submit_time = time.time()
        
        # Add to execution queue with priority
        # Higher urgency = lower priority number (higher priority)
        urgency = order.metadata.get('urgency', 'NORMAL')
        priority_map = {'URGENT': 1, 'HIGH': 2, 'NORMAL': 3, 'LOW': 4}
        priority = priority_map.get(urgency, 3)
        
        self.execution_queue.put((priority, time.time(), order))
        self.execution_stats['orders_submitted'] += 1
        
        return order.order_id
    
    def _execution_loop(self):
        """Main execution loop running in separate thread"""
        while self.running:
            try:
                # Get next order from queue (blocking with timeout)
                priority, submit_time, order = self.execution_queue.get(timeout=0.1)
                
                # Process the order
                self._process_order(order)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in execution loop: {e}")
                continue
    
    def _process_order(self, order: Order):
        """Process individual order through full pipeline"""
        start_time = time.time_ns()
        
        try:
            # 1. Risk validation
            risk_check = self.risk_manager.validate_order(order)
            if not risk_check['approved']:
                self._handle_rejection(order, risk_check['reason'])
                return
            
            # 2. Anti-gaming analysis
            recent_orders = self._get_recent_orders(order.symbol)
            gaming_score, patterns = self.anti_gaming.analyze_order_flow(
                order.symbol, recent_orders, {}
            )
            
            # 3. Apply gaming protection delay
            should_delay, delay_ms = self.anti_gaming.should_delay_order(order.symbol, order)
            if should_delay:
                time.sleep(delay_ms / 1000.0)
            
            # 4. Order splitting if needed
            child_orders = self.splitter.split_order(order)
            
            # 5. Execute orders
            for child_order in child_orders:
                result = self.router.route_order(child_order)
                self._handle_execution_result(result)
            
            # 6. Update statistics
            total_time_ms = (time.time_ns() - start_time) / 1_000_000
            self.execution_stats['total_execution_time_ms'] += total_time_ms
            
        except Exception as e:
            self._handle_error(order, str(e))
    
    def _handle_execution_result(self, result: ExecutionResult):
        """Handle execution result"""
        # Update positions
        self.risk_manager.update_position(result)
        
        # Update statistics
        if result.status == OrderStatus.FILLED:
            self.execution_stats['orders_filled'] += 1
        
        # Store in history
        self.order_history.append(result)
        
        # Keep history size manageable
        if len(self.order_history) > 10000:
            self.order_history = self.order_history[-5000:]
    
    def _handle_rejection(self, order: Order, reason: str):
        """Handle order rejection"""
        result = ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=0,
            avg_price=0.0,
            venue=Venue.NSE,  # Default
            execution_time_ms=0.0,
            slippage_bps=None,
            status=OrderStatus.REJECTED,
            error_message=reason
        )
        self.order_history.append(result)
    
    def _handle_error(self, order: Order, error_message: str):
        """Handle execution error"""
        result = ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=0,
            avg_price=0.0,
            venue=Venue.NSE,  # Default
            execution_time_ms=0.0,
            slippage_bps=None,
            status=OrderStatus.REJECTED,
            error_message=error_message
        )
        self.order_history.append(result)
    
    def _get_recent_orders(self, symbol: str) -> List[Dict]:
        """Get recent orders for symbol"""
        cutoff_time = time.time() - 300  # Last 5 minutes
        
        recent = []
        for result in reversed(self.order_history):
            if result.symbol == symbol and time.time() - (result.execution_time_ms / 1000) < 300:
                recent.append({
                    'timestamp': time.time() - (result.execution_time_ms / 1000),
                    'side': result.side.value,
                    'quantity': result.quantity,
                    'price': result.avg_price,
                    'status': result.status.value,
                    'order_type': 'LIMIT'  # Simplified
                })
        
        return recent[-100:]  # Last 100 orders
    
    def get_performance_stats(self) -> Dict:
        """Get execution performance statistics"""
        if self.execution_stats['orders_submitted'] > 0:
            fill_rate = self.execution_stats['orders_filled'] / self.execution_stats['orders_submitted']
        else:
            fill_rate = 0.0
        
        # Calculate average slippage
        recent_executions = [r for r in self.order_history[-1000:] 
                           if r.slippage_bps is not None]
        avg_slippage = np.mean([r.slippage_bps for r in recent_executions]) if recent_executions else 0.0
        
        return {
            'orders_submitted': self.execution_stats['orders_submitted'],
            'orders_filled': self.execution_stats['orders_filled'],
            'fill_rate': fill_rate,
            'avg_execution_time_ms': (self.execution_stats['total_execution_time_ms'] / 
                                    max(self.execution_stats['orders_submitted'], 1)),
            'avg_slippage_bps': avg_slippage,
            'active_positions': len([k for k, v in self.risk_manager.positions.items() if v != 0]),
            'current_pnl': self.risk_manager.current_pnl
        }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Execution System...")
    
    # Create execution engine
    engine = ExecutionEngine()
    engine.start()
    
    # Test orders
    test_orders = [
        Order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=1000,
            price=2500.0,
            order_type=OrderType.LIMIT,
            metadata={'urgency': 'NORMAL'}
        ),
        Order(
            symbol="TCS",
            side=OrderSide.SELL,
            quantity=500,
            price=3200.0,
            order_type=OrderType.LIMIT,
            metadata={'urgency': 'HIGH'}
        ),
        Order(
            symbol="HDFCBANK",
            side=OrderSide.BUY,
            quantity=2000,
            price=1500.0,
            order_type=OrderType.LIMIT,
            metadata={'urgency': 'URGENT'}
        )
    ]
    
    # Submit test orders
    order_ids = []
    for order in test_orders:
        order_id = engine.submit_order(order)
        order_ids.append(order_id)
        print(f"Submitted order: {order_id} for {order.symbol}")
    
    # Wait for execution
    import time
    time.sleep(2)
    
    # Get performance stats
    stats = engine.get_performance_stats()
    print("\nExecution Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Stop engine
    engine.stop()
    print("\nExecution system test completed!")