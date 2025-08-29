# Risk Management System for Indian HFT Trading

## SEBI-Compliant Risk Controls

### Comprehensive Risk Framework

The risk management system implements multi-layered controls designed specifically for SEBI regulations and Indian market dynamics, operating at nanosecond frequencies to prevent risk violations before they occur.

#### Core Risk Components

1. **Pre-trade Risk Controls**: Real-time validation before order submission
2. **Position Monitoring**: Continuous tracking with microsecond updates
3. **Circuit Breaker Response**: Automated reaction to market halts
4. **Regulatory Compliance**: SEBI-specific limit enforcement
5. **Emergency Protocols**: Kill switches and failover mechanisms

### Pre-trade Risk Controls

#### Position and Exposure Limits
```python
class SEBIPositionLimits:
    """
    SEBI-compliant position and exposure limits
    """
    def __init__(self):
        # SEBI Algorithm Trading Guidelines
        self.limits = {
            'single_stock_market_cap_pct': 1.0,  # 1% of market cap
            'single_stock_absolute_amount': 50_000_000_000,  # ₹500 crores
            'portfolio_exposure_pct': 5.0,  # 5% of AUM
            'daily_volume_participation_pct': 10.0,  # 10% of avg daily volume
            'order_to_trade_ratio_global': 500,  # Maximum OTR across all symbols
            'order_to_trade_ratio_symbol': 50,  # Maximum OTR per symbol
            'velocity_orders_per_second': 100,  # Per symbol
            'price_band_deviation_pct': 20.0,  # ±20% from LTP
        }
        
        # Current tracking
        self.current_positions = defaultdict(int)
        self.current_exposures = defaultdict(float)
        self.order_counts = defaultdict(int)
        self.trade_counts = defaultdict(int)
        self.velocity_trackers = defaultdict(lambda: deque(maxlen=100))
        
    def validate_position_limit(self, symbol, new_quantity, market_cap, current_price):
        """Validate position against SEBI limits"""
        current_position = self.current_positions[symbol]
        new_position = current_position + new_quantity
        
        # Market cap percentage check
        position_value = abs(new_position) * current_price
        market_cap_pct = (position_value / market_cap) * 100
        
        if market_cap_pct > self.limits['single_stock_market_cap_pct']:
            return False, f"Position would exceed {self.limits['single_stock_market_cap_pct']}% of market cap"
        
        # Absolute amount check
        if position_value > self.limits['single_stock_absolute_amount']:
            return False, f"Position value would exceed ₹{self.limits['single_stock_absolute_amount']:,.0f}"
        
        return True, "Position limit check passed"
    
    def validate_velocity_limit(self, symbol):
        """Validate order velocity against SEBI limits"""
        current_time = time.time()
        
        # Clean old entries (older than 1 second)
        self.velocity_trackers[symbol] = deque(
            [t for t in self.velocity_trackers[symbol] if current_time - t < 1.0],
            maxlen=100
        )
        
        # Check current velocity
        orders_last_second = len(self.velocity_trackers[symbol])
        
        if orders_last_second >= self.limits['velocity_orders_per_second']:
            return False, f"Order velocity {orders_last_second}/sec exceeds limit {self.limits['velocity_orders_per_second']}/sec"
        
        # Add current order timestamp
        self.velocity_trackers[symbol].append(current_time)
        
        return True, "Velocity limit check passed"
    
    def validate_order_to_trade_ratio(self, symbol):
        """Validate order-to-trade ratio"""
        orders = self.order_counts[symbol]
        trades = self.trade_counts[symbol]
        
        if trades == 0:
            current_otr = orders
        else:
            current_otr = orders / trades
        
        # Symbol-specific OTR check
        if current_otr > self.limits['order_to_trade_ratio_symbol']:
            return False, f"OTR {current_otr:.1f} exceeds symbol limit {self.limits['order_to_trade_ratio_symbol']}"
        
        # Global OTR check (simplified - would aggregate across all symbols)
        total_orders = sum(self.order_counts.values())
        total_trades = sum(self.trade_counts.values())
        global_otr = total_orders / max(total_trades, 1)
        
        if global_otr > self.limits['order_to_trade_ratio_global']:
            return False, f"Global OTR {global_otr:.1f} exceeds limit {self.limits['order_to_trade_ratio_global']}"
        
        return True, "OTR check passed"
```

#### Price Band Validation
```python
class PriceBandValidator:
    """
    Validate orders against SEBI price band requirements
    """
    def __init__(self):
        # Different price bands for different stock categories
        self.price_bands = {
            'liquid_stocks': 2.0,      # 2% band
            'semi_liquid_stocks': 5.0,  # 5% band
            'illiquid_stocks': 10.0,    # 10% band
            'trade_to_trade': 20.0      # 20% band
        }
        
        # Cache for stock categorization
        self.stock_categories = {}
        self.reference_prices = {}
        
    def validate_price_band(self, symbol, order_price, reference_price, stock_category='liquid_stocks'):
        """
        Validate order price against applicable price band
        """
        band_percentage = self.price_bands.get(stock_category, 2.0)
        
        upper_limit = reference_price * (1 + band_percentage / 100)
        lower_limit = reference_price * (1 - band_percentage / 100)
        
        if order_price > upper_limit:
            return False, f"Price {order_price:.2f} exceeds upper limit {upper_limit:.2f}"
        
        if order_price < lower_limit:
            return False, f"Price {order_price:.2f} below lower limit {lower_limit:.2f}"
        
        return True, "Price band check passed"
    
    def get_dynamic_price_band(self, symbol, volatility, liquidity_score):
        """
        Calculate dynamic price band based on market conditions
        """
        base_band = 2.0  # Base 2% band
        
        # Volatility adjustment
        volatility_adjustment = min(volatility * 10, 5.0)  # Cap at 5%
        
        # Liquidity adjustment
        liquidity_adjustment = max(0, (1.0 - liquidity_score) * 3.0)  # Max 3% for illiquid stocks
        
        dynamic_band = base_band + volatility_adjustment + liquidity_adjustment
        
        # Cap at maximum allowed band
        return min(dynamic_band, 20.0)
```

### Real-time Position Monitoring

#### Nanosecond Position Tracking
```python
class PositionMonitor:
    """
    Real-time position monitoring with nanosecond precision
    """
    def __init__(self):
        self.positions = {}  # symbol -> position data
        self.position_lock = threading.RLock()
        self.monitoring_active = False
        self.alert_thresholds = {
            'position_warning_pct': 80.0,  # Warn at 80% of limit
            'pnl_warning_threshold': -1_000_000,  # ₹10 lakhs loss
            'var_breach_threshold': 1.5  # 1.5x VaR limit
        }
        
        # Performance tracking
        self.update_latencies = deque(maxlen=1000)
        
    def update_position(self, symbol, quantity_change, price, timestamp_ns):
        """
        Update position with nanosecond precision
        """
        update_start = time.time_ns()
        
        with self.position_lock:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'net_position': 0,
                    'avg_price': 0.0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0,
                    'total_trades': 0,
                    'last_update_ns': timestamp_ns,
                    'trade_history': deque(maxlen=1000)
                }
            
            position_data = self.positions[symbol]
            
            # Update position using FIFO accounting
            old_position = position_data['net_position']
            new_position = old_position + quantity_change
            
            # Update average price
            if quantity_change > 0:  # Adding to position
                if old_position >= 0:  # Same direction
                    total_value = (old_position * position_data['avg_price']) + (quantity_change * price)
                    position_data['avg_price'] = total_value / new_position if new_position != 0 else 0
                else:  # Reducing short position
                    if new_position >= 0:  # Closed short and went long
                        realized_pnl = (position_data['avg_price'] - price) * abs(old_position)
                        position_data['realized_pnl'] += realized_pnl
                        position_data['avg_price'] = price
                    # else: still short, no change to avg_price
            else:  # Reducing position
                if old_position > 0 and new_position >= 0:  # Reducing long
                    realized_pnl = (price - position_data['avg_price']) * abs(quantity_change)
                    position_data['realized_pnl'] += realized_pnl
                elif old_position <= 0:  # Adding to short
                    if old_position == 0:
                        position_data['avg_price'] = price
                    else:
                        total_value = (abs(old_position) * position_data['avg_price']) + (abs(quantity_change) * price)
                        position_data['avg_price'] = total_value / abs(new_position)
            
            position_data['net_position'] = new_position
            position_data['total_trades'] += 1
            position_data['last_update_ns'] = timestamp_ns
            
            # Record trade
            trade_record = {
                'timestamp_ns': timestamp_ns,
                'quantity': quantity_change,
                'price': price,
                'new_position': new_position
            }
            position_data['trade_history'].append(trade_record)
            
        # Track update latency
        update_latency = time.time_ns() - update_start
        self.update_latencies.append(update_latency)
        
        # Check for alerts
        self._check_position_alerts(symbol, position_data)
        
        return position_data
    
    def calculate_unrealized_pnl(self, symbol, current_price):
        """Calculate unrealized P&L for position"""
        with self.position_lock:
            if symbol not in self.positions:
                return 0.0
            
            position_data = self.positions[symbol]
            net_position = position_data['net_position']
            avg_price = position_data['avg_price']
            
            if net_position == 0:
                return 0.0
            
            unrealized_pnl = (current_price - avg_price) * net_position
            position_data['unrealized_pnl'] = unrealized_pnl
            
            return unrealized_pnl
    
    def get_portfolio_metrics(self, current_prices):
        """Calculate portfolio-level risk metrics"""
        with self.position_lock:
            total_pnl = 0.0
            total_exposure = 0.0
            position_count = 0
            
            for symbol, position_data in self.positions.items():
                if position_data['net_position'] != 0:
                    position_count += 1
                    
                    # Calculate current values
                    current_price = current_prices.get(symbol, position_data['avg_price'])
                    unrealized_pnl = self.calculate_unrealized_pnl(symbol, current_price)
                    
                    total_pnl += position_data['realized_pnl'] + unrealized_pnl
                    total_exposure += abs(position_data['net_position'] * current_price)
            
            return {
                'total_pnl': total_pnl,
                'total_exposure': total_exposure,
                'position_count': position_count,
                'avg_update_latency_ns': np.mean(self.update_latencies) if self.update_latencies else 0
            }
```

### Circuit Breaker Response System

#### Automated Circuit Breaker Handling
```python
class CircuitBreakerManager:
    """
    Automated response to NSE/BSE circuit breaker events
    """
    def __init__(self):
        # Circuit breaker levels for Indian markets
        self.market_wide_levels = [
            {'threshold_pct': 10.0, 'halt_duration_min': 15, 'description': '10% decline - 15 min halt'},
            {'threshold_pct': 15.0, 'halt_duration_min': 60, 'description': '15% decline - 1 hour halt'},
            {'threshold_pct': 20.0, 'halt_duration_min': 1440, 'description': '20% decline - trading halted'}
        ]
        
        self.individual_stock_bands = {
            2.0: 'liquid_stocks',
            5.0: 'semi_liquid_stocks',
            10.0: 'illiquid_stocks',
            20.0: 'trade_to_trade_segment'
        }
        
        # State tracking
        self.circuit_breaker_status = {}
        self.halt_start_times = {}
        self.auto_responses = {
            'cancel_all_orders': True,
            'flatten_positions': False,  # Configurable
            'stop_new_orders': True,
            'send_alerts': True
        }
        
    def detect_circuit_breaker(self, symbol, current_price, reference_price, market_index_change=None):
        """
        Detect circuit breaker triggers
        """
        price_change_pct = ((current_price - reference_price) / reference_price) * 100
        
        # Individual stock circuit breaker
        stock_band = self._get_stock_band(symbol)
        if abs(price_change_pct) >= stock_band:
            return self._trigger_stock_circuit_breaker(symbol, price_change_pct, stock_band)
        
        # Market-wide circuit breaker (if market index data available)
        if market_index_change is not None:
            for level in self.market_wide_levels:
                if market_index_change <= -level['threshold_pct']:
                    return self._trigger_market_circuit_breaker(level)
        
        return None
    
    def _trigger_stock_circuit_breaker(self, symbol, price_change_pct, band_pct):
        """Handle individual stock circuit breaker"""
        event = {
            'type': 'STOCK_CIRCUIT_BREAKER',
            'symbol': symbol,
            'price_change_pct': price_change_pct,
            'band_pct': band_pct,
            'timestamp': time.time(),
            'actions_taken': []
        }
        
        # Execute automatic responses
        if self.auto_responses['cancel_all_orders']:
            self._cancel_symbol_orders(symbol)
            event['actions_taken'].append('CANCELLED_ALL_ORDERS')
        
        if self.auto_responses['stop_new_orders']:
            self._block_symbol_trading(symbol)
            event['actions_taken'].append('BLOCKED_NEW_ORDERS')
        
        if self.auto_responses['send_alerts']:
            self._send_circuit_breaker_alert(event)
            event['actions_taken'].append('SENT_ALERTS')
        
        self.circuit_breaker_status[symbol] = event
        
        return event
    
    def _trigger_market_circuit_breaker(self, level):
        """Handle market-wide circuit breaker"""
        event = {
            'type': 'MARKET_CIRCUIT_BREAKER',
            'level': level,
            'timestamp': time.time(),
            'halt_duration_min': level['halt_duration_min'],
            'actions_taken': []
        }
        
        # Execute market-wide responses
        if self.auto_responses['cancel_all_orders']:
            self._cancel_all_orders()
            event['actions_taken'].append('CANCELLED_ALL_ORDERS')
        
        if self.auto_responses['flatten_positions']:
            self._emergency_position_flatten()
            event['actions_taken'].append('FLATTENED_POSITIONS')
        
        if self.auto_responses['stop_new_orders']:
            self._block_all_trading()
            event['actions_taken'].append('BLOCKED_ALL_TRADING')
        
        # Set halt timer
        self.halt_start_times['MARKET'] = time.time()
        
        return event
```

### Emergency Protocols

#### Kill Switch Implementation
```python
class EmergencyKillSwitch:
    """
    Emergency kill switch for immediate trading halt
    """
    def __init__(self):
        self.kill_switch_active = False
        self.kill_switch_reasons = []
        self.emergency_contacts = []
        
        # Kill switch triggers
        self.triggers = {
            'max_loss_threshold': -10_000_000,  # ₹1 crore loss
            'position_limit_breach': True,
            'system_error_rate_threshold': 0.1,  # 10% error rate
            'latency_threshold_ms': 1000,  # 1 second latency
            'manual_trigger': False
        }
        
        # Recovery procedures
        self.recovery_steps = [
            'assess_system_status',
            'validate_positions',
            'check_market_conditions',
            'verify_connectivity',
            'run_diagnostic_tests',
            'manual_approval_required'
        ]
        
    def activate_kill_switch(self, reason, manual=False):
        """
        Activate emergency kill switch
        """
        if self.kill_switch_active:
            return {"status": "already_active", "reason": "Kill switch already activated"}
        
        activation_time = time.time()
        
        self.kill_switch_active = True
        self.kill_switch_reasons.append({
            'reason': reason,
            'timestamp': activation_time,
            'manual': manual
        })
        
        # Execute emergency procedures
        actions_taken = []
        
        try:
            # 1. Stop all new order submissions
            self._block_all_order_submission()
            actions_taken.append('BLOCKED_ORDER_SUBMISSION')
            
            # 2. Cancel all active orders
            cancelled_orders = self._emergency_cancel_all_orders()
            actions_taken.append(f'CANCELLED_{cancelled_orders}_ORDERS')
            
            # 3. Disconnect from exchanges (if required)
            if reason in ['system_error', 'connectivity_issue']:
                self._emergency_disconnect()
                actions_taken.append('DISCONNECTED_EXCHANGES')
            
            # 4. Send emergency notifications
            self._send_emergency_notifications(reason, actions_taken)
            actions_taken.append('SENT_NOTIFICATIONS')
            
            # 5. Log emergency event
            self._log_emergency_event(reason, actions_taken, activation_time)
            
        except Exception as e:
            # Even if some emergency actions fail, ensure kill switch is active
            self._log_emergency_error(str(e))
        
        return {
            "status": "activated",
            "reason": reason,
            "timestamp": activation_time,
            "actions_taken": actions_taken
        }
    
    def check_kill_switch_triggers(self, system_metrics):
        """
        Continuously monitor for kill switch trigger conditions
        """
        if self.kill_switch_active:
            return
        
        # Check loss threshold
        if system_metrics.get('total_pnl', 0) <= self.triggers['max_loss_threshold']:
            self.activate_kill_switch('MAX_LOSS_EXCEEDED')
            return
        
        # Check error rate
        error_rate = system_metrics.get('error_rate', 0)
        if error_rate >= self.triggers['system_error_rate_threshold']:
            self.activate_kill_switch('HIGH_ERROR_RATE')
            return
        
        # Check latency
        avg_latency = system_metrics.get('avg_latency_ms', 0)
        if avg_latency >= self.triggers['latency_threshold_ms']:
            self.activate_kill_switch('HIGH_LATENCY')
            return
        
        # Check position limits
        if system_metrics.get('position_limit_breached', False):
            self.activate_kill_switch('POSITION_LIMIT_BREACH')
            return
    
    def deactivate_kill_switch(self, operator_id, recovery_checklist_completed=False):
        """
        Deactivate kill switch after proper recovery procedures
        """
        if not self.kill_switch_active:
            return {"status": "not_active", "message": "Kill switch is not active"}
        
        if not recovery_checklist_completed:
            return {
                "status": "recovery_required",
                "recovery_steps": self.recovery_steps,
                "message": "Complete recovery checklist before deactivation"
            }
        
        deactivation_time = time.time()
        
        self.kill_switch_active = False
        
        # Log deactivation
        self._log_kill_switch_deactivation(operator_id, deactivation_time)
        
        # Re-enable systems gradually
        self._gradual_system_reactivation()
        
        return {
            "status": "deactivated",
            "operator": operator_id,
            "timestamp": deactivation_time,
            "message": "Kill switch deactivated - systems reactivating"
        }
```

### Regulatory Compliance Monitoring

#### SEBI Audit Trail
```python
class SEBIAuditTrail:
    """
    Comprehensive audit trail system for SEBI compliance
    """
    def __init__(self):
        self.audit_logs = []
        self.log_lock = threading.Lock()
        self.log_file_path = "/var/log/hft_audit/"
        self.retention_days = 2555  # 7 years (SEBI requirement)
        
        # Audit event types
        self.event_types = {
            'ORDER_SUBMITTED': 'Order submission',
            'ORDER_CANCELLED': 'Order cancellation',
            'ORDER_MODIFIED': 'Order modification',
            'TRADE_EXECUTED': 'Trade execution',
            'POSITION_UPDATE': 'Position update',
            'RISK_VIOLATION': 'Risk limit violation',
            'CIRCUIT_BREAKER': 'Circuit breaker event',
            'KILL_SWITCH': 'Emergency kill switch',
            'SYSTEM_ERROR': 'System error',
            'USER_ACTION': 'User action'
        }
        
    def log_audit_event(self, event_type, details, user_id=None, system_id=None):
        """
        Log audit event with SEBI-required details
        """
        timestamp_ns = time.time_ns()
        
        audit_record = {
            'timestamp_ns': timestamp_ns,
            'timestamp_readable': datetime.fromtimestamp(timestamp_ns / 1e9).isoformat(),
            'event_type': event_type,
            'event_description': self.event_types.get(event_type, 'Unknown event'),
            'details': details,
            'user_id': user_id,
            'system_id': system_id,
            'session_id': self._get_session_id(),
            'checksum': self._calculate_checksum(details)
        }
        
        with self.log_lock:
            self.audit_logs.append(audit_record)
            
            # Write to persistent storage
            self._write_audit_log(audit_record)
            
            # Maintain memory buffer size
            if len(self.audit_logs) > 10000:
                self.audit_logs = self.audit_logs[-5000:]
    
    def log_order_event(self, order, event_type, additional_details=None):
        """Log order-related events"""
        details = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': order.price,
            'order_type': order.order_type.value,
            'venue': order.venue.value if order.venue else None,
            'status': order.status.value,
        }
        
        if additional_details:
            details.update(additional_details)
        
        self.log_audit_event(event_type, details)
    
    def generate_compliance_report(self, start_date, end_date, symbol=None):
        """
        Generate SEBI compliance report
        """
        start_timestamp = start_date.timestamp() * 1e9
        end_timestamp = end_date.timestamp() * 1e9
        
        # Filter logs by date range
        filtered_logs = [
            log for log in self.audit_logs
            if start_timestamp <= log['timestamp_ns'] <= end_timestamp
        ]
        
        # Filter by symbol if specified
        if symbol:
            filtered_logs = [
                log for log in filtered_logs
                if log['details'].get('symbol') == symbol
            ]
        
        # Generate summary statistics
        report = {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'symbol_filter': symbol
            },
            'summary': {
                'total_events': len(filtered_logs),
                'event_breakdown': defaultdict(int),
                'order_statistics': self._calculate_order_statistics(filtered_logs),
                'risk_events': self._calculate_risk_events(filtered_logs),
                'compliance_metrics': self._calculate_compliance_metrics(filtered_logs)
            },
            'detailed_logs': filtered_logs
        }
        
        # Count events by type
        for log in filtered_logs:
            report['summary']['event_breakdown'][log['event_type']] += 1
        
        return report
```

This comprehensive risk management system provides real-time monitoring, SEBI compliance, and emergency protocols specifically designed for Indian HFT trading operations.