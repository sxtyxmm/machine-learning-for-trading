# Data Infrastructure for Indian HFT Trading

## Ultra-Low Latency Architecture for NSE/BSE

### Direct Market Data Feeds

#### NSE Market Data Specifications

##### NEAT-NOW Feed Protocol
- **Protocol**: Binary format with message-based structure
- **Frequency**: Real-time tick-by-tick updates
- **Latency**: < 100 microseconds from exchange matching engine
- **Bandwidth**: 50-100 Mbps peak during market hours
- **Redundancy**: Primary + backup feeds from separate data centers

##### Data Elements
```
Message Structure:
- Header (8 bytes): Message type, sequence number, timestamp
- Body (Variable): Symbol, price, quantity, order details
- Footer (4 bytes): Checksum for data integrity
```

##### Subscription Levels
1. **Level 1**: Best bid/ask with last traded price
2. **Level 2**: 5-level order book depth
3. **Level 3**: 20-level order book depth (for members)
4. **Full Depth**: Complete order book (colocation only)

#### BSE Market Data (BOLT Feed)
- **Protocol**: Similar binary format to NSE
- **Latency**: 150-250 microseconds
- **Bandwidth**: 30-60 Mbps peak
- **Integration**: Unified with NSE feed via common API

### Timestamping and Synchronization

#### Hardware Timestamping Requirements
- **PTP (Precision Time Protocol)**: IEEE 1588v2 compliant
- **GPS Synchronization**: Primary time source
- **Atomic Clock Backup**: For GPS outages
- **Accuracy**: ±100 nanoseconds across all systems

#### Implementation Architecture
```
GPS Antenna → PTP Grandmaster → Network Switch → Trading Servers
                                             → Market Data Servers
                                             → Risk Management
```

### Hardware Specifications for Indian Colocation

#### Recommended Server Configuration

##### Computing Infrastructure
```yaml
Primary Trading Server:
  CPU: Intel Xeon Platinum 8280 (28 cores @ 2.7GHz)
  RAM: 384GB DDR4-3200 ECC
  Network: Mellanox ConnectX-6 (100GbE)
  Storage: Intel Optane P5800X NVMe (1.6TB)
  Motherboard: Supermicro X12DPi-NT6 (dual socket)
  
Market Data Server:
  CPU: Intel Xeon Gold 6348 (28 cores @ 2.6GHz)  
  RAM: 256GB DDR4-3200 ECC
  Network: Dual 25GbE + 10GbE management
  Storage: Samsung PM1735 NVMe (800GB)
  
Risk Management Server:
  CPU: Intel Xeon Silver 4316 (20 cores @ 2.3GHz)
  RAM: 128GB DDR4-2933 ECC
  Network: Dual 10GbE
  Storage: Intel SSD DC P4610 (1.6TB)
```

##### Network Infrastructure
```yaml
Top-of-Rack Switch: 
  Model: Arista 7280R3-32P 100GbE
  Latency: < 500 nanoseconds
  Features: PTP support, RDMA capable
  
Cross-connects:
  Type: Single-mode fiber optic
  Length: Standardized by exchange
  Latency: < 5 microseconds per connection
  
Load Balancers:
  Model: A10 Thunder 6630
  Throughput: 40 Gbps
  Latency: < 10 microseconds
```

### Software Stack for Ultra-Low Latency

#### Operating System Optimization
```bash
# Real-time Linux configuration
CONFIG_PREEMPT_RT=y
CONFIG_NO_HZ_FULL=y
CONFIG_RCU_NOCB_CPU=y

# Kernel parameters
isolcpus=2-27           # Isolate CPU cores for trading
nohz_full=2-27          # Disable timer ticks
rcu_nocbs=2-27          # RCU callbacks on specific cores
intel_pstate=disable    # Disable CPU frequency scaling
```

#### Network Stack Optimization
```yaml
DPDK Configuration:
  Version: 21.11 LTS
  Drivers: igb_uio or vfio-pci
  Memory: 8GB hugepages (1GB pages)
  Cores: 4 dedicated cores for packet processing
  
Kernel Bypass:
  Technology: Intel DPDK + SPDK
  Benefits: 90% latency reduction vs kernel stack
  Polling: Continuous polling vs interrupt-driven
  
Zero-Copy Networking:
  Implementation: Custom UDP with memory mapping
  Buffer Management: Lock-free ring buffers
  Message Passing: Shared memory IPC
```

#### Custom Protocol Implementation
```cpp
// High-performance market data parser
class NSEFeedHandler {
private:
    struct __attribute__((packed)) NSEMessage {
        uint64_t timestamp;      // Hardware timestamp
        uint32_t sequence;       // Message sequence
        uint16_t msg_type;       // Message type
        uint16_t symbol_id;      // Symbol identifier
        uint64_t price;          // Price in ticks
        uint32_t quantity;       // Quantity
        uint8_t  side;          // Buy/Sell
    };
    
public:
    inline bool parseMessage(const char* buffer, NSEMessage& msg) {
        // Optimized parsing with minimal branching
        memcpy(&msg, buffer, sizeof(NSEMessage));
        return validateChecksum(buffer);
    }
    
    inline void processOrderBook(const NSEMessage& msg) {
        // Lock-free order book updates
        orderBook.updateLevel(msg.symbol_id, msg.price, 
                             msg.quantity, msg.side);
    }
};
```

### Data Storage and Retrieval

#### Tick Data Storage Architecture
```yaml
Primary Storage:
  Technology: Intel Optane Persistent Memory
  Capacity: 128GB per server
  Latency: < 150 nanoseconds random access
  Use Case: Active order book and recent tick data
  
Secondary Storage:
  Technology: NVMe SSD arrays
  Capacity: 10TB usable per server
  Latency: < 100 microseconds
  Use Case: Intraday historical data and analytics
  
Archive Storage:
  Technology: High-capacity NVMe
  Capacity: 100TB+ per cluster
  Compression: LZ4 for 60% space savings
  Use Case: Long-term tick data retention
```

#### Data Compression and Indexing
```python
# Custom tick data compression
class TickDataCompressor:
    def __init__(self):
        self.delta_encoding = True
        self.bit_packing = True
        
    def compress_ticks(self, tick_data):
        # Delta encoding for prices and timestamps
        price_deltas = np.diff(tick_data['price'])
        time_deltas = np.diff(tick_data['timestamp'])
        
        # Bit packing for small integer fields
        packed_data = self.pack_bits([
            price_deltas, time_deltas, 
            tick_data['quantity'], tick_data['side']
        ])
        
        # LZ4 final compression
        return lz4.compress(packed_data)
```

### Market Data Processing Pipeline

#### Real-time Data Flow
```
Exchange Feed → Hardware NIC → DPDK → Parser → Order Book → Strategy
                                              → Risk Check → Order Router
```

#### Processing Components
```python
class MarketDataPipeline:
    def __init__(self):
        self.feed_handler = NSEFeedHandler()
        self.order_book = LockFreeOrderBook()
        self.strategy_engine = StrategyEngine()
        self.risk_manager = RiskManager()
        
    def process_tick(self, raw_data):
        # Hardware timestamping
        hw_timestamp = get_hardware_timestamp()
        
        # Parse message
        message = self.feed_handler.parse(raw_data)
        message.hw_timestamp = hw_timestamp
        
        # Update order book
        self.order_book.update(message)
        
        # Generate signals
        signals = self.strategy_engine.process(message)
        
        # Risk validation
        for signal in signals:
            if self.risk_manager.validate(signal):
                self.send_order(signal)
```

### Monitoring and Alerting

#### Latency Monitoring
```python
class LatencyMonitor:
    def __init__(self):
        self.metrics = {
            'feed_latency': deque(maxlen=1000),
            'processing_latency': deque(maxlen=1000),
            'order_latency': deque(maxlen=1000)
        }
        
    def record_latency(self, start_time, end_time, metric_type):
        latency_ns = end_time - start_time
        self.metrics[metric_type].append(latency_ns)
        
        # Alert if latency exceeds threshold
        if latency_ns > self.thresholds[metric_type]:
            self.send_alert(f"High {metric_type}: {latency_ns}ns")
    
    def get_percentiles(self, metric_type):
        data = np.array(self.metrics[metric_type])
        return {
            'p50': np.percentile(data, 50),
            'p95': np.percentile(data, 95),
            'p99': np.percentile(data, 99),
            'max': np.max(data)
        }
```

#### System Health Monitoring
```yaml
Monitoring Stack:
  Metrics: Prometheus + custom collectors
  Alerting: AlertManager + PagerDuty
  Dashboards: Grafana with real-time charts
  Log Aggregation: ELK stack with high-speed ingestion
  
Key Metrics:
  - Market data latency (p50, p95, p99)
  - Order execution latency
  - Network packet loss
  - CPU utilization per core
  - Memory usage and allocation
  - Disk I/O latency and IOPS
  - Exchange connectivity status
  - Order book quality metrics
```

### Disaster Recovery and Failover

#### Redundancy Architecture
```
Primary Site (Mumbai) ← Cross-connect → Secondary Site (Mumbai DR)
        ↓                                        ↓
Exchange Feeds                            Exchange Feeds (Backup)
        ↓                                        ↓
Trading Systems                           Standby Systems
```

#### Automated Failover
```python
class FailoverManager:
    def __init__(self):
        self.primary_healthy = True
        self.failover_threshold_ms = 10
        self.heartbeat_interval_ms = 1
        
    def monitor_primary(self):
        while True:
            if not self.check_primary_health():
                self.initiate_failover()
            time.sleep(self.heartbeat_interval_ms / 1000)
    
    def initiate_failover(self):
        # Stop all trading on primary
        self.stop_primary_trading()
        
        # Activate secondary systems
        self.activate_secondary()
        
        # Resume trading on secondary
        self.start_secondary_trading()
        
        # Update DNS/load balancer
        self.update_routing()
```

### Compliance and Audit Trail

#### SEBI Audit Requirements
```python
class AuditLogger:
    def __init__(self):
        self.log_format = {
            'timestamp': 'nanosecond_precision',
            'order_id': 'unique_identifier',
            'symbol': 'trading_symbol',
            'side': 'buy_sell',
            'quantity': 'order_quantity',
            'price': 'order_price',
            'order_type': 'limit_market_etc',
            'execution_venue': 'NSE_BSE',
            'user_id': 'trader_identifier',
            'strategy_id': 'algorithm_identifier',
            'risk_checks': 'validation_results'
        }
        
    def log_order(self, order_details):
        log_entry = {
            'timestamp': time.time_ns(),
            'event_type': 'ORDER_SENT',
            'details': order_details,
            'checksum': self.calculate_checksum(order_details)
        }
        
        # Write to multiple destinations for redundancy
        self.write_to_primary_log(log_entry)
        self.write_to_backup_log(log_entry)
        self.send_to_exchange_reporting(log_entry)
```

### Performance Benchmarks

#### Target Performance Metrics
```yaml
Latency Targets:
  Market Data Processing: < 5 microseconds
  Order Generation: < 2 microseconds  
  Risk Validation: < 1 microsecond
  Order Transmission: < 10 microseconds
  Total Order-to-Market: < 20 microseconds

Throughput Targets:
  Market Data Messages: 500,000/second
  Order Processing: 100,000/second
  Risk Checks: 1,000,000/second
  
Availability Targets:
  System Uptime: 99.99% during market hours
  Market Data Feed: 99.999% availability
  Order Execution: 99.95% success rate
```

#### Benchmark Testing Framework
```python
class PerformanceTester:
    def __init__(self):
        self.test_scenarios = [
            'market_open_surge',
            'high_volatility_period',
            'market_close_activity',
            'circuit_breaker_event'
        ]
        
    def run_latency_test(self, duration_seconds=300):
        results = {}
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Generate synthetic market data
            test_message = self.generate_test_message()
            
            # Measure end-to-end latency
            t1 = time.time_ns()
            self.process_message(test_message)
            t2 = time.time_ns()
            
            latency_ns = t2 - t1
            results[len(results)] = latency_ns
            
        return self.analyze_results(results)
```

### Cost Estimates for Data Infrastructure

#### Initial Setup Costs (₹ Lakhs)
```yaml
Hardware:
  Servers (4 units): 60
  Network Equipment: 25
  Storage Systems: 30
  Monitoring Tools: 10
  
Colocation:
  NSE Setup Fee: 10
  BSE Setup Fee: 8
  Rack Space (6 months): 18
  Cross-connects: 5
  Power and Cooling: 12
  
Software Licenses:
  Operating Systems: 3
  Monitoring Software: 8
  Development Tools: 5
  
Total Initial Investment: 194 Lakhs
```

#### Monthly Operational Costs (₹ Lakhs)
```yaml
Data Feeds:
  NSE Market Data: 12
  BSE Market Data: 6
  
Infrastructure:
  Colocation Rental: 8
  Power and Cooling: 4
  Internet Connectivity: 3
  
Maintenance:
  Hardware Support: 2
  Software Updates: 1
  Monitoring Services: 1
  
Total Monthly Cost: 37 Lakhs
```