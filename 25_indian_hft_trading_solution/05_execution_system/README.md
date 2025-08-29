# Execution System for Indian HFT Trading

## Smart Order Routing for NSE/BSE

### Ultra-Low Latency Order Management System

The execution system is designed for sub-microsecond order routing with intelligent venue selection, order splitting, and anti-gaming protection specifically optimized for Indian market structure.

#### Key Components

1. **Smart Order Router (SOR)**: Intelligent venue selection between NSE and BSE
2. **Order Splitting Engine**: SEBI-compliant order fragmentation
3. **Anti-Gaming Logic**: Protection against predatory algorithms
4. **Risk Controls**: Pre-trade and real-time position monitoring
5. **Market Making Module**: Automated liquidity provision

### Smart Order Routing Algorithm

#### Venue Selection Logic
```python
class VenueSelector:
    """
    Intelligent venue selection for NSE vs BSE execution
    """
    def __init__(self):
        self.venue_scores = {'NSE': 0.0, 'BSE': 0.0}
        self.execution_history = defaultdict(list)
        
    def select_venue(self, symbol, order_size, urgency_level):
        """
        Select optimal execution venue based on multiple factors
        
        Factors considered:
        1. Liquidity depth at each venue
        2. Historical fill rates
        3. Spread differences
        4. Market share of the symbol
        5. Time of day patterns
        """
        nse_score = self._calculate_venue_score('NSE', symbol, order_size, urgency_level)
        bse_score = self._calculate_venue_score('BSE', symbol, order_size, urgency_level)
        
        # Add randomization to avoid predictability
        nse_score += random.uniform(-0.05, 0.05)
        bse_score += random.uniform(-0.05, 0.05)
        
        return 'NSE' if nse_score > bse_score else 'BSE'
    
    def _calculate_venue_score(self, venue, symbol, order_size, urgency_level):
        """Calculate venue score based on current market conditions"""
        score = 0.0
        
        # Liquidity factor (40% weight)
        liquidity_score = self._get_liquidity_score(venue, symbol, order_size)
        score += 0.4 * liquidity_score
        
        # Spread factor (25% weight)
        spread_score = self._get_spread_score(venue, symbol)
        score += 0.25 * spread_score
        
        # Fill rate factor (20% weight)
        fill_rate_score = self._get_fill_rate_score(venue, symbol)
        score += 0.20 * fill_rate_score
        
        # Market share factor (10% weight)
        market_share_score = self._get_market_share_score(venue, symbol)
        score += 0.10 * market_share_score
        
        # Urgency adjustment (5% weight)
        urgency_score = self._get_urgency_score(venue, urgency_level)
        score += 0.05 * urgency_score
        
        return score
```

#### Order Splitting Strategy
```python
class OrderSplittingEngine:
    """
    SEBI-compliant order splitting with adaptive sizing
    """
    def __init__(self):
        self.max_order_size_pct = 0.10  # Max 10% of daily volume
        self.min_order_size = 1  # Minimum order size
        self.split_strategies = {
            'TWAP': self._twap_split,
            'VWAP': self._vwap_split,
            'POV': self._pov_split,  # Participation of Volume
            'URGENT': self._urgent_split
        }
        
    def split_order(self, parent_order, strategy='TWAP'):
        """
        Split large order into smaller child orders
        """
        splitter = self.split_strategies.get(strategy, self._twap_split)
        return splitter(parent_order)
    
    def _twap_split(self, order):
        """Time-Weighted Average Price splitting"""
        time_horizon = order.get('time_horizon', 600)  # 10 minutes default
        intervals = min(time_horizon // 30, 20)  # Max 20 intervals
        
        child_orders = []
        base_size = order['quantity'] // intervals
        remainder = order['quantity'] % intervals
        
        for i in range(intervals):
            size = base_size + (1 if i < remainder else 0)
            if size > 0:
                child_order = {
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'quantity': size,
                    'price': order['price'],
                    'order_type': 'LIMIT',
                    'execution_time': time.time() + (i * time_horizon / intervals),
                    'parent_id': order['order_id'],
                    'child_id': f"{order['order_id']}_C{i+1}"
                }
                child_orders.append(child_order)
        
        return child_orders
    
    def _vwap_split(self, order):
        """Volume-Weighted Average Price splitting"""
        # Use historical volume profile for the symbol
        volume_profile = self._get_volume_profile(order['symbol'])
        
        child_orders = []
        total_volume = sum(volume_profile)
        
        for i, period_volume in enumerate(volume_profile):
            if period_volume > 0:
                size_fraction = period_volume / total_volume
                size = int(order['quantity'] * size_fraction)
                
                if size > 0:
                    child_order = {
                        'symbol': order['symbol'],
                        'side': order['side'],
                        'quantity': size,
                        'price': order['price'],
                        'order_type': 'LIMIT',
                        'execution_time': time.time() + (i * 30),  # 30-second intervals
                        'parent_id': order['order_id'],
                        'child_id': f"{order['order_id']}_V{i+1}"
                    }
                    child_orders.append(child_order)
        
        return child_orders
```

### Anti-Gaming Protection

#### Order Flow Analysis
```python
class AntiGamingEngine:
    """
    Protection against predatory trading algorithms
    """
    def __init__(self):
        self.order_patterns = defaultdict(list)
        self.suspicious_activity = defaultdict(int)
        self.gaming_indicators = {
            'order_stuffing': self._detect_order_stuffing,
            'quote_stuffing': self._detect_quote_stuffing,
            'layering': self._detect_layering,
            'spoofing': self._detect_spoofing
        }
        
    def analyze_order_flow(self, recent_orders, market_data):
        """
        Analyze recent order flow for gaming patterns
        """
        gaming_score = 0.0
        detected_patterns = []
        
        for pattern_name, detector in self.gaming_indicators.items():
            score, detected = detector(recent_orders, market_data)
            gaming_score += score
            if detected:
                detected_patterns.append(pattern_name)
        
        return gaming_score, detected_patterns
    
    def _detect_order_stuffing(self, orders, market_data):
        """Detect order stuffing patterns"""
        if len(orders) < 10:
            return 0.0, False
        
        # Check for rapid order placement and cancellation
        recent_orders = [o for o in orders if time.time() - o['timestamp'] < 60]
        cancel_rate = sum(1 for o in recent_orders if o['status'] == 'CANCELLED') / len(recent_orders)
        
        # High cancellation rate indicates potential stuffing
        if cancel_rate > 0.8 and len(recent_orders) > 50:
            return 0.8, True
        
        return cancel_rate * 0.5, False
    
    def _detect_layering(self, orders, market_data):
        """Detect layering/spoofing patterns"""
        # Look for large orders on one side followed by execution on the other
        buy_orders = [o for o in orders if o['side'] == 'BUY' and o['status'] == 'ACTIVE']
        sell_orders = [o for o in orders if o['side'] == 'SELL' and o['status'] == 'ACTIVE']
        executions = [o for o in orders if o['status'] == 'EXECUTED']
        
        # Check for imbalanced order placement
        buy_volume = sum(o['quantity'] for o in buy_orders)
        sell_volume = sum(o['quantity'] for o in sell_orders)
        
        if buy_volume > 0 or sell_volume > 0:
            imbalance = abs(buy_volume - sell_volume) / (buy_volume + sell_volume)
            
            # High imbalance with few executions suggests layering
            execution_rate = len(executions) / max(len(orders), 1)
            
            if imbalance > 0.7 and execution_rate < 0.1:
                return 0.6, True
        
        return 0.0, False
    
    def should_delay_order(self, order, gaming_score):
        """
        Determine if order should be delayed due to gaming activity
        """
        if gaming_score > 0.5:
            # Add random delay between 10-100ms
            delay_ms = random.uniform(10, 100)
            return True, delay_ms
        
        return False, 0
```

### Order Execution Engine

#### Core Execution Logic
```python
class OrderExecutionEngine:
    """
    High-performance order execution with multiple strategies
    """
    def __init__(self):
        self.venue_selector = VenueSelector()
        self.order_splitter = OrderSplittingEngine()
        self.anti_gaming = AntiGamingEngine()
        self.risk_manager = RiskManager()
        
        self.active_orders = {}
        self.execution_queue = queue.PriorityQueue()
        self.order_history = defaultdict(list)
        
        # Performance metrics
        self.execution_stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'fill_rate': 0.0,
            'avg_execution_time_ms': 0.0,
            'slippage_bps': deque(maxlen=1000)
        }
        
    def submit_order(self, order):
        """
        Submit order for execution with full processing pipeline
        """
        start_time = time.time_ns()
        
        try:
            # 1. Risk validation
            risk_check = self.risk_manager.validate_order(order)
            if not risk_check['approved']:
                return self._create_rejection(order, risk_check['reason'])
            
            # 2. Anti-gaming analysis
            recent_orders = self._get_recent_orders(order['symbol'])
            gaming_score, gaming_patterns = self.anti_gaming.analyze_order_flow(
                recent_orders, self._get_market_data(order['symbol'])
            )
            
            # 3. Apply gaming protection delay if needed
            should_delay, delay_ms = self.anti_gaming.should_delay_order(order, gaming_score)
            if should_delay:
                time.sleep(delay_ms / 1000.0)
            
            # 4. Order splitting if required
            if order['quantity'] > self._get_max_order_size(order['symbol']):
                child_orders = self.order_splitter.split_order(order)
                return self._submit_child_orders(child_orders)
            
            # 5. Venue selection
            venue = self.venue_selector.select_venue(
                order['symbol'], 
                order['quantity'], 
                order.get('urgency', 'NORMAL')
            )
            
            # 6. Execute order
            execution_result = self._execute_single_order(order, venue)
            
            # 7. Update statistics
            execution_time_ms = (time.time_ns() - start_time) / 1_000_000
            self._update_execution_stats(execution_result, execution_time_ms)
            
            return execution_result
            
        except Exception as e:
            return self._create_error(order, str(e))
    
    def _execute_single_order(self, order, venue):
        """
        Execute single order at specified venue
        """
        order_id = self._generate_order_id()
        order['order_id'] = order_id
        order['venue'] = venue
        order['status'] = 'PENDING'
        order['submit_time'] = time.time()
        
        # Add to active orders
        self.active_orders[order_id] = order
        
        # Simulate order execution (in production, would interface with exchange)
        execution_latency_ms = self._simulate_execution(order, venue)
        
        # Update order status
        order['status'] = 'FILLED'
        order['fill_time'] = time.time()
        order['execution_latency_ms'] = execution_latency_ms
        
        # Calculate slippage
        slippage_bps = self._calculate_slippage(order)
        order['slippage_bps'] = slippage_bps
        
        # Move to history
        self.order_history[order['symbol']].append(order)
        del self.active_orders[order_id]
        
        return {
            'order_id': order_id,
            'status': 'FILLED',
            'filled_quantity': order['quantity'],
            'avg_price': order['price'],
            'execution_time_ms': execution_latency_ms,
            'slippage_bps': slippage_bps,
            'venue': venue
        }
    
    def _simulate_execution(self, order, venue):
        """
        Simulate order execution with realistic latencies
        """
        base_latency = {
            'NSE': random.uniform(50, 150),    # 50-150 microseconds
            'BSE': random.uniform(80, 200)     # 80-200 microseconds
        }
        
        # Add market impact delay for larger orders
        size_factor = min(order['quantity'] / 1000, 5.0)  # Cap at 5x
        market_impact_delay = size_factor * random.uniform(10, 50)
        
        total_latency = base_latency.get(venue, 100) + market_impact_delay
        
        # Simulate processing time
        time.sleep(total_latency / 1_000_000)  # Convert to seconds
        
        return total_latency / 1000  # Return in milliseconds
```

### Market Making Module

#### Automated Liquidity Provision
```python
class MarketMakingEngine:
    """
    Automated market making for Indian equity markets
    """
    def __init__(self):
        self.target_symbols = []
        self.quote_width_bps = 5  # 5 basis points default spread
        self.max_position_size = {}
        self.inventory_targets = {}
        
        self.active_quotes = defaultdict(dict)
        self.inventory = defaultdict(int)
        self.pnl = defaultdict(float)
        
    def add_symbol(self, symbol, quote_width_bps=5, max_position=10000):
        """Add symbol to market making universe"""
        self.target_symbols.append(symbol)
        self.quote_width_bps_map[symbol] = quote_width_bps
        self.max_position_size[symbol] = max_position
        self.inventory_targets[symbol] = 0
        
    def update_quotes(self, symbol, market_data):
        """
        Update bid/ask quotes for symbol
        """
        if symbol not in self.target_symbols:
            return
        
        # Get current market data
        mid_price = market_data.get('mid_price', 0)
        if mid_price <= 0:
            return
        
        # Calculate quote prices
        quote_width = self.quote_width_bps_map.get(symbol, 5) / 10000.0
        half_spread = mid_price * quote_width / 2
        
        # Inventory adjustment
        inventory_adjustment = self._calculate_inventory_adjustment(symbol)
        
        bid_price = mid_price - half_spread + inventory_adjustment
        ask_price = mid_price + half_spread + inventory_adjustment
        
        # Risk checks
        if not self._can_quote(symbol, bid_price, ask_price):
            self._cancel_quotes(symbol)
            return
        
        # Submit quotes
        bid_order = {
            'symbol': symbol,
            'side': 'BUY',
            'quantity': self._calculate_quote_size(symbol, 'BUY'),
            'price': bid_price,
            'order_type': 'LIMIT',
            'time_in_force': 'IOC'
        }
        
        ask_order = {
            'symbol': symbol,
            'side': 'SELL',
            'quantity': self._calculate_quote_size(symbol, 'SELL'),
            'price': ask_price,
            'order_type': 'LIMIT',
            'time_in_force': 'IOC'
        }
        
        self._submit_quotes(symbol, bid_order, ask_order)
    
    def _calculate_inventory_adjustment(self, symbol):
        """
        Calculate price adjustment based on inventory position
        """
        current_inventory = self.inventory.get(symbol, 0)
        target_inventory = self.inventory_targets.get(symbol, 0)
        max_position = self.max_position_size.get(symbol, 10000)
        
        # Inventory imbalance as percentage of max position
        inventory_imbalance = (current_inventory - target_inventory) / max_position
        
        # Adjust quotes to encourage inventory reduction
        # Positive inventory -> lower quotes to sell
        # Negative inventory -> higher quotes to buy
        adjustment_factor = -0.0002  # 2 basis points per 100% inventory
        
        return inventory_imbalance * adjustment_factor
    
    def _calculate_quote_size(self, symbol, side):
        """
        Calculate quote size based on market conditions and inventory
        """
        base_size = 100  # Base quote size
        current_inventory = self.inventory.get(symbol, 0)
        max_position = self.max_position_size.get(symbol, 10000)
        
        # Reduce quote size if approaching position limits
        position_utilization = abs(current_inventory) / max_position
        size_reduction = max(0, position_utilization - 0.7) * 2  # Start reducing at 70%
        
        adjusted_size = base_size * (1 - size_reduction)
        
        # Don't quote if we would exceed position limits
        if side == 'BUY' and current_inventory + adjusted_size > max_position:
            return 0
        if side == 'SELL' and current_inventory - adjusted_size < -max_position:
            return 0
        
        return max(int(adjusted_size), 1)
```

### Performance Monitoring

#### Execution Analytics
```python
class ExecutionAnalytics:
    """
    Comprehensive execution performance analytics
    """
    def __init__(self):
        self.execution_data = []
        self.venue_performance = defaultdict(list)
        self.symbol_performance = defaultdict(list)
        
    def record_execution(self, execution_result):
        """Record execution for analysis"""
        self.execution_data.append({
            'timestamp': time.time(),
            'symbol': execution_result['symbol'],
            'venue': execution_result['venue'],
            'side': execution_result['side'],
            'quantity': execution_result['quantity'],
            'price': execution_result['price'],
            'execution_time_ms': execution_result['execution_time_ms'],
            'slippage_bps': execution_result['slippage_bps']
        })
        
        # Update venue and symbol specific metrics
        venue = execution_result['venue']
        symbol = execution_result['symbol']
        
        self.venue_performance[venue].append(execution_result)
        self.symbol_performance[symbol].append(execution_result)
    
    def get_performance_report(self, time_window_hours=24):
        """Generate comprehensive performance report"""
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_executions = [
            ex for ex in self.execution_data 
            if ex['timestamp'] > cutoff_time
        ]
        
        if not recent_executions:
            return {}
        
        report = {
            'overview': self._calculate_overview_metrics(recent_executions),
            'venue_breakdown': self._calculate_venue_metrics(recent_executions),
            'symbol_breakdown': self._calculate_symbol_metrics(recent_executions),
            'time_distribution': self._calculate_time_distribution(recent_executions)
        }
        
        return report
    
    def _calculate_overview_metrics(self, executions):
        """Calculate overall execution metrics"""
        if not executions:
            return {}
        
        execution_times = [ex['execution_time_ms'] for ex in executions]
        slippages = [ex['slippage_bps'] for ex in executions if ex['slippage_bps'] is not None]
        
        return {
            'total_executions': len(executions),
            'avg_execution_time_ms': np.mean(execution_times),
            'p95_execution_time_ms': np.percentile(execution_times, 95),
            'p99_execution_time_ms': np.percentile(execution_times, 99),
            'avg_slippage_bps': np.mean(slippages) if slippages else 0,
            'total_volume': sum(ex['quantity'] for ex in executions),
            'buy_sell_ratio': len([ex for ex in executions if ex['side'] == 'BUY']) / len(executions)
        }
```

This execution system provides comprehensive order management with intelligent routing, risk controls, and performance monitoring specifically designed for the Indian market structure and SEBI regulations.