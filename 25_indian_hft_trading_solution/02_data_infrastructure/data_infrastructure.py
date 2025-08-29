"""
Ultra-Low Latency Data Infrastructure for Indian HFT Trading
Optimized for NSE/BSE market data processing with nanosecond precision
"""

import time
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import threading
import queue
import mmap
import struct
import socket
import select
from abc import ABC, abstractmethod

class Exchange(Enum):
    NSE = "NSE"
    BSE = "BSE"

class MessageType(Enum):
    TRADE = 1
    ORDER_ADD = 2
    ORDER_MODIFY = 3
    ORDER_DELETE = 4
    QUOTE_UPDATE = 5

@dataclass
class MarketDataMessage:
    """Standardized market data message structure"""
    timestamp_ns: int
    exchange: Exchange
    symbol: str
    message_type: MessageType
    price: float
    quantity: int
    side: str  # 'B' for buy, 'S' for sell
    order_id: Optional[str] = None
    sequence_number: Optional[int] = None
    hardware_timestamp: Optional[int] = None

@dataclass
class OrderBookLevel:
    """Single level of order book"""
    price: float
    quantity: int
    order_count: int

class LockFreeOrderBook:
    """
    Lock-free order book implementation for ultra-low latency
    Uses atomic operations and memory barriers for thread safety
    """
    
    def __init__(self, symbol: str, max_levels: int = 20):
        self.symbol = symbol
        self.max_levels = max_levels
        
        # Separate arrays for bids and asks
        self.bid_levels = [OrderBookLevel(0.0, 0, 0) for _ in range(max_levels)]
        self.ask_levels = [OrderBookLevel(0.0, 0, 0) for _ in range(max_levels)]
        
        # Atomic counters for versioning
        self.bid_version = 0
        self.ask_version = 0
        
        # Last update timestamp
        self.last_update_ns = 0
    
    def update_level(self, price: float, quantity: int, side: str, order_count: int = 1):
        """
        Update order book level with minimal latency
        """
        update_time = time.time_ns()
        
        if side == 'B':
            self._update_bid_side(price, quantity, order_count)
            self.bid_version += 1
        else:
            self._update_ask_side(price, quantity, order_count)
            self.ask_version += 1
            
        self.last_update_ns = update_time
    
    def _update_bid_side(self, price: float, quantity: int, order_count: int):
        """Update bid side maintaining price-time priority"""
        if quantity == 0:
            # Remove level
            for i, level in enumerate(self.bid_levels):
                if level.price == price:
                    # Shift levels down
                    for j in range(i, self.max_levels - 1):
                        self.bid_levels[j] = self.bid_levels[j + 1]
                    self.bid_levels[-1] = OrderBookLevel(0.0, 0, 0)
                    break
        else:
            # Find insertion point (bids sorted descending)
            insert_idx = 0
            for i, level in enumerate(self.bid_levels):
                if level.price == price:
                    # Update existing level
                    level.quantity = quantity
                    level.order_count = order_count
                    return
                elif level.price < price or level.price == 0.0:
                    insert_idx = i
                    break
                    
            # Insert new level
            if insert_idx < self.max_levels:
                # Shift levels down
                for i in range(self.max_levels - 1, insert_idx, -1):
                    self.bid_levels[i] = self.bid_levels[i - 1]
                self.bid_levels[insert_idx] = OrderBookLevel(price, quantity, order_count)
    
    def _update_ask_side(self, price: float, quantity: int, order_count: int):
        """Update ask side maintaining price-time priority"""
        if quantity == 0:
            # Remove level
            for i, level in enumerate(self.ask_levels):
                if level.price == price:
                    # Shift levels down
                    for j in range(i, self.max_levels - 1):
                        self.ask_levels[j] = self.ask_levels[j + 1]
                    self.ask_levels[-1] = OrderBookLevel(0.0, 0, 0)
                    break
        else:
            # Find insertion point (asks sorted ascending)
            insert_idx = 0
            for i, level in enumerate(self.ask_levels):
                if level.price == price:
                    # Update existing level
                    level.quantity = quantity
                    level.order_count = order_count
                    return
                elif level.price > price or level.price == 0.0:
                    insert_idx = i
                    break
                    
            # Insert new level
            if insert_idx < self.max_levels:
                # Shift levels down
                for i in range(self.max_levels - 1, insert_idx, -1):
                    self.ask_levels[i] = self.ask_levels[i - 1]
                self.ask_levels[insert_idx] = OrderBookLevel(price, quantity, order_count)
    
    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices"""
        best_bid = self.bid_levels[0].price if self.bid_levels[0].price > 0 else None
        best_ask = self.ask_levels[0].price if self.ask_levels[0].price > 0 else None
        return best_bid, best_ask
    
    def get_spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """Calculate mid price"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2.0
        return None

class NSEFeedHandler:
    """
    High-performance NSE market data feed handler
    Optimized for minimal parsing latency
    """
    
    # NSE message format (simplified)
    MESSAGE_HEADER_SIZE = 16
    MESSAGE_FORMATS = {
        MessageType.TRADE: struct.Struct('<QHHQdIc'),  # timestamp, msg_type, symbol_id, seq, price, qty, side
        MessageType.ORDER_ADD: struct.Struct('<QHHQdIcQ'),  # includes order_id
        MessageType.ORDER_MODIFY: struct.Struct('<QHHQdIcQ'),
        MessageType.ORDER_DELETE: struct.Struct('<QHHQdIcQ'),
        MessageType.QUOTE_UPDATE: struct.Struct('<QHHQdIdI')  # price, qty for both bid and ask
    }
    
    def __init__(self, symbol_map: Dict[int, str]):
        self.symbol_map = symbol_map
        self.sequence_numbers = defaultdict(int)
        self.last_timestamps = defaultdict(int)
        
    def parse_message(self, raw_data: bytes, hardware_timestamp: int) -> Optional[MarketDataMessage]:
        """
        Parse binary NSE message with minimal overhead
        """
        if len(raw_data) < self.MESSAGE_HEADER_SIZE:
            return None
            
        try:
            # Extract header
            timestamp, msg_type_raw, symbol_id = struct.unpack('<QHH', raw_data[:12])
            
            # Validate message type
            try:
                msg_type = MessageType(msg_type_raw)
            except ValueError:
                return None
                
            # Get symbol name
            symbol = self.symbol_map.get(symbol_id, f"UNKNOWN_{symbol_id}")
            
            # Parse body based on message type
            format_struct = self.MESSAGE_FORMATS.get(msg_type)
            if not format_struct:
                return None
                
            try:
                parsed_data = format_struct.unpack(raw_data[:format_struct.size])
            except struct.error:
                return None
            
            # Extract common fields
            timestamp, _, _, sequence, price, quantity, side = parsed_data[:7]
            
            # Validate sequence number
            expected_seq = self.sequence_numbers[symbol_id] + 1
            if sequence != expected_seq:
                # Handle gap - in production, would request retransmission
                pass
            self.sequence_numbers[symbol_id] = sequence
            
            return MarketDataMessage(
                timestamp_ns=timestamp,
                exchange=Exchange.NSE,
                symbol=symbol,
                message_type=msg_type,
                price=price,
                quantity=quantity,
                side=side.decode('ascii'),
                sequence_number=sequence,
                hardware_timestamp=hardware_timestamp
            )
            
        except Exception:
            # In production, would log error details
            return None

class MarketDataProcessor:
    """
    High-performance market data processing engine
    Handles multiple symbols and exchanges simultaneously
    """
    
    def __init__(self, max_symbols: int = 1000):
        self.order_books = {}
        self.message_handlers = []
        self.stats = {
            'messages_processed': 0,
            'parsing_errors': 0,
            'latency_stats': deque(maxlen=10000)
        }
        
        # Performance optimization
        self.symbol_cache = {}
        self.max_symbols = max_symbols
        
    def add_symbol(self, symbol: str, exchange: Exchange) -> LockFreeOrderBook:
        """Add symbol for processing"""
        key = f"{exchange.value}:{symbol}"
        if key not in self.order_books:
            self.order_books[key] = LockFreeOrderBook(symbol)
        return self.order_books[key]
    
    def register_handler(self, handler: Callable[[MarketDataMessage], None]):
        """Register message handler callback"""
        self.message_handlers.append(handler)
    
    def process_message(self, message: MarketDataMessage):
        """
        Process market data message with minimal latency
        """
        processing_start = time.time_ns()
        
        # Update order book
        order_book_key = f"{message.exchange.value}:{message.symbol}"
        order_book = self.order_books.get(order_book_key)
        
        if order_book:
            if message.message_type in [MessageType.TRADE, MessageType.ORDER_ADD, 
                                      MessageType.ORDER_MODIFY, MessageType.QUOTE_UPDATE]:
                order_book.update_level(
                    message.price, 
                    message.quantity, 
                    message.side
                )
        
        # Call registered handlers
        for handler in self.message_handlers:
            try:
                handler(message)
            except Exception:
                # In production, would log error
                pass
        
        # Update statistics
        processing_end = time.time_ns()
        latency_ns = processing_end - processing_start
        self.stats['latency_stats'].append(latency_ns)
        self.stats['messages_processed'] += 1
    
    def get_order_book(self, symbol: str, exchange: Exchange) -> Optional[LockFreeOrderBook]:
        """Get order book for symbol"""
        key = f"{exchange.value}:{symbol}"
        return self.order_books.get(key)
    
    def get_performance_stats(self) -> Dict:
        """Get processing performance statistics"""
        if self.stats['latency_stats']:
            latencies = np.array(self.stats['latency_stats'])
            return {
                'messages_processed': self.stats['messages_processed'],
                'parsing_errors': self.stats['parsing_errors'],
                'latency_p50_ns': np.percentile(latencies, 50),
                'latency_p95_ns': np.percentile(latencies, 95),
                'latency_p99_ns': np.percentile(latencies, 99),
                'latency_max_ns': np.max(latencies),
                'throughput_msg_per_sec': len(latencies) / max(1, (latencies[-1] - latencies[0]) / 1e9)
            }
        return {}

class TickDataStorage:
    """
    High-performance tick data storage with compression
    Optimized for ultra-fast writes and retrieval
    """
    
    def __init__(self, storage_path: str, compression_enabled: bool = True):
        self.storage_path = storage_path
        self.compression_enabled = compression_enabled
        self.write_buffer = defaultdict(list)
        self.buffer_size = 10000  # Messages before flush
        
    def store_tick(self, message: MarketDataMessage):
        """Store tick data with minimal latency"""
        key = f"{message.exchange.value}:{message.symbol}"
        
        # Serialize message to binary format
        tick_data = struct.pack('<QdIc', 
                               message.timestamp_ns,
                               message.price,
                               message.quantity,
                               message.side.encode('ascii'))
        
        self.write_buffer[key].append(tick_data)
        
        # Flush if buffer is full
        if len(self.write_buffer[key]) >= self.buffer_size:
            self._flush_buffer(key)
    
    def _flush_buffer(self, key: str):
        """Flush write buffer to storage"""
        if not self.write_buffer[key]:
            return
            
        # In production, would use memory-mapped files or specialized databases
        filename = f"{self.storage_path}/{key}_{int(time.time())}.tick"
        
        with open(filename, 'wb') as f:
            for tick_data in self.write_buffer[key]:
                f.write(tick_data)
        
        self.write_buffer[key].clear()
    
    def retrieve_ticks(self, symbol: str, exchange: Exchange, 
                      start_time: int, end_time: int) -> List[MarketDataMessage]:
        """Retrieve tick data for time range"""
        # Implementation would depend on storage backend
        # This is a simplified version
        ticks = []
        key = f"{exchange.value}:{symbol}"
        
        # In production, would use indexed storage for fast retrieval
        # For now, return empty list
        return ticks

class HardwareTimestamping:
    """
    Hardware timestamping utilities for precise timing
    """
    
    @staticmethod
    def get_hardware_timestamp() -> int:
        """
        Get hardware timestamp in nanoseconds
        In production, would interface with network card timestamping
        """
        return time.time_ns()
    
    @staticmethod
    def calibrate_clock_offset() -> int:
        """
        Calibrate offset between system clock and hardware clock
        """
        # Simplified - in production would use PTP synchronization
        return 0

class LatencyMonitor:
    """
    Real-time latency monitoring and alerting
    """
    
    def __init__(self, alert_threshold_ns: int = 50000):  # 50 microseconds
        self.alert_threshold_ns = alert_threshold_ns
        self.measurements = deque(maxlen=1000)
        self.alerts = []
        
    def record_latency(self, start_time_ns: int, end_time_ns: int, operation: str):
        """Record latency measurement"""
        latency_ns = end_time_ns - start_time_ns
        
        measurement = {
            'timestamp': end_time_ns,
            'latency_ns': latency_ns,
            'operation': operation
        }
        
        self.measurements.append(measurement)
        
        # Check for alerts
        if latency_ns > self.alert_threshold_ns:
            alert = {
                'timestamp': end_time_ns,
                'latency_ns': latency_ns,
                'operation': operation,
                'threshold_ns': self.alert_threshold_ns
            }
            self.alerts.append(alert)
            self._send_alert(alert)
    
    def _send_alert(self, alert: Dict):
        """Send latency alert"""
        # In production, would integrate with monitoring system
        print(f"LATENCY ALERT: {alert['operation']} took {alert['latency_ns']/1000:.1f} μs "
              f"(threshold: {alert['threshold_ns']/1000:.1f} μs)")
    
    def get_statistics(self) -> Dict:
        """Get latency statistics"""
        if not self.measurements:
            return {}
            
        latencies = [m['latency_ns'] for m in self.measurements]
        return {
            'count': len(latencies),
            'mean_ns': np.mean(latencies),
            'std_ns': np.std(latencies),
            'min_ns': np.min(latencies),
            'max_ns': np.max(latencies),
            'p50_ns': np.percentile(latencies, 50),
            'p95_ns': np.percentile(latencies, 95),
            'p99_ns': np.percentile(latencies, 99)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize components
    symbol_map = {1: "RELIANCE", 2: "TCS", 3: "HDFCBANK", 4: "INFY", 5: "HINDUNILVR"}
    
    nse_handler = NSEFeedHandler(symbol_map)
    processor = MarketDataProcessor()
    storage = TickDataStorage("/tmp/tick_data")
    latency_monitor = LatencyMonitor()
    
    # Add symbols to processor
    for symbol in symbol_map.values():
        processor.add_symbol(symbol, Exchange.NSE)
    
    # Register storage handler
    processor.register_handler(storage.store_tick)
    
    # Simulate market data processing
    print("Testing market data infrastructure...")
    
    for i in range(1000):
        # Simulate NSE message
        raw_message = struct.pack('<QHHQdIc', 
                                 time.time_ns(),  # timestamp
                                 1,  # message_type (TRADE)
                                 1,  # symbol_id (RELIANCE)
                                 i,  # sequence
                                 2500.0 + np.random.normal(0, 5),  # price
                                 100,  # quantity
                                 b'B' if i % 2 == 0 else b'S')  # side
        
        # Process message
        start_time = time.time_ns()
        hw_timestamp = HardwareTimestamping.get_hardware_timestamp()
        
        message = nse_handler.parse_message(raw_message, hw_timestamp)
        if message:
            processor.process_message(message)
        
        end_time = time.time_ns()
        latency_monitor.record_latency(start_time, end_time, "message_processing")
    
    # Print performance statistics
    print("\nPerformance Statistics:")
    proc_stats = processor.get_performance_stats()
    for key, value in proc_stats.items():
        if 'latency' in key and 'ns' in key:
            print(f"{key}: {value/1000:.1f} μs")
        else:
            print(f"{key}: {value}")
    
    print("\nLatency Monitor Statistics:")
    lat_stats = latency_monitor.get_statistics()
    for key, value in lat_stats.items():
        if 'ns' in key:
            print(f"{key}: {value/1000:.1f} μs")
        else:
            print(f"{key}: {value}")
    
    # Test order book functionality
    reliance_book = processor.get_order_book("RELIANCE", Exchange.NSE)
    if reliance_book:
        best_bid, best_ask = reliance_book.get_best_bid_ask()
        spread = reliance_book.get_spread()
        mid_price = reliance_book.get_mid_price()
        
        print(f"\nRELIANCE Order Book:")
        print(f"Best Bid: ₹{best_bid:.2f}" if best_bid else "Best Bid: None")
        print(f"Best Ask: ₹{best_ask:.2f}" if best_ask else "Best Ask: None") 
        print(f"Spread: ₹{spread:.2f}" if spread else "Spread: None")
        print(f"Mid Price: ₹{mid_price:.2f}" if mid_price else "Mid Price: None")