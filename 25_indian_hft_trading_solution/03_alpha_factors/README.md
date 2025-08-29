# Alpha Factor Engineering for Indian HFT Markets

## India-Specific Microstructure Predictors

### Order Flow Imbalance Signals for NSE/BSE

#### 1. Volume Weighted Order Imbalance (VWOI)
Accounts for the different tick sizes and lot structures in Indian markets:

```python
def calculate_vwoi(order_book, lookback_window=10):
    """
    Calculate Volume Weighted Order Imbalance adapted for Indian markets
    Accounts for NSE/BSE specific tick sizes and lot structures
    """
    bid_volume = sum(level.quantity * level.price for level in order_book.bid_levels[:5])
    ask_volume = sum(level.quantity * level.price for level in order_book.ask_levels[:5])
    
    total_volume = bid_volume + ask_volume
    if total_volume == 0:
        return 0
    
    imbalance = (bid_volume - ask_volume) / total_volume
    return imbalance
```

#### 2. FII/DII Flow-Adjusted Imbalance
Incorporates Foreign Institutional Investor and Domestic Institutional Investor flow patterns:

```python
def calculate_fii_dii_adjusted_imbalance(order_flow, fii_data, dii_data):
    """
    Adjust order imbalance based on FII/DII activity patterns
    Higher weight during FII/DII active hours (9:15-10:30 AM, 2:30-3:30 PM)
    """
    base_imbalance = calculate_basic_imbalance(order_flow)
    
    # FII/DII activity multipliers based on historical patterns
    fii_multiplier = 1.0 + 0.3 * (fii_data['net_flow'] / fii_data['avg_daily_flow'])
    dii_multiplier = 1.0 + 0.2 * (dii_data['net_flow'] / dii_data['avg_daily_flow'])
    
    adjusted_imbalance = base_imbalance * fii_multiplier * dii_multiplier
    return np.clip(adjusted_imbalance, -1.0, 1.0)
```

### Market Microstructure Features

#### 3. Tick-Size Normalized Spread
Normalizes bid-ask spread by the applicable tick size for each price band:

```python
def normalized_spread(best_bid, best_ask, tick_size):
    """
    Calculate spread normalized by tick size
    Critical for comparing liquidity across different price bands in Indian markets
    """
    if best_bid and best_ask:
        raw_spread = best_ask - best_bid
        return raw_spread / tick_size
    return np.nan
```

#### 4. Order Book Depth Pressure
Measures pressure at different levels considering Indian market depth characteristics:

```python
def depth_pressure(order_book, levels=5):
    """
    Calculate pressure from order book depth
    Adapted for NSE/BSE typical order book depths
    """
    bid_depth = sum(level.quantity for level in order_book.bid_levels[:levels])
    ask_depth = sum(level.quantity for level in order_book.ask_levels[:levels])
    
    total_depth = bid_depth + ask_depth
    if total_depth == 0:
        return 0
    
    pressure = (bid_depth - ask_depth) / total_depth
    return pressure
```

### Advanced Alpha Factors

#### 5. Circuit Breaker Proximity Signal
Predicts approach to circuit breaker limits specific to Indian markets:

```python
def circuit_breaker_proximity(current_price, reference_price, circuit_percentage=2.0):
    """
    Calculate proximity to circuit breaker limits
    Uses NSE/BSE specific circuit breaker percentages (2%, 5%, 10%, 20%)
    """
    upper_limit = reference_price * (1 + circuit_percentage / 100)
    lower_limit = reference_price * (1 - circuit_percentage / 100)
    
    if current_price >= reference_price:
        proximity = (current_price - reference_price) / (upper_limit - reference_price)
    else:
        proximity = (reference_price - current_price) / (reference_price - lower_limit)
    
    return min(proximity, 1.0)
```

#### 6. Pre-Opening Session Signal
Captures predictive information from NSE/BSE pre-opening sessions:

```python
def pre_opening_signal(pre_opening_data, historical_data):
    """
    Extract signal from pre-opening session (9:00-9:15 AM)
    Analyzes order buildup and price discovery patterns
    """
    # Calculate indicative equilibrium price vs historical VWAP
    ieq_price = pre_opening_data['indicative_equilibrium_price']
    historical_vwap = historical_data['vwap']
    
    price_deviation = (ieq_price - historical_vwap) / historical_vwap
    
    # Order quantity and value ratios
    total_qty = pre_opening_data['total_buy_qty'] + pre_opening_data['total_sell_qty']
    qty_imbalance = (pre_opening_data['total_buy_qty'] - pre_opening_data['total_sell_qty']) / total_qty
    
    # Combine signals
    signal = 0.6 * price_deviation + 0.4 * qty_imbalance
    return np.tanh(signal)  # Bounded between -1 and 1
```

#### 7. Intraday Momentum with Time Decay
Indian market specific momentum calculation with time-of-day adjustments:

```python
def intraday_momentum_india(prices, volumes, timestamps, decay_factor=0.95):
    """
    Calculate intraday momentum adjusted for Indian market patterns
    Higher weights during high-activity periods
    """
    # Time-of-day weights for Indian markets
    def get_time_weight(timestamp):
        hour = timestamp.hour
        minute = timestamp.minute
        
        # High activity periods get higher weights
        if (9 <= hour <= 10) or (14 <= hour <= 15):  # Opening and closing hours
            return 1.5
        elif 11 <= hour <= 13:  # Mid-day low activity
            return 0.7
        else:
            return 1.0
    
    momentum = 0
    total_weight = 0
    
    for i in range(1, len(prices)):
        price_change = (prices[i] - prices[i-1]) / prices[i-1]
        volume_weight = volumes[i] / np.mean(volumes)
        time_weight = get_time_weight(timestamps[i])
        weight = volume_weight * time_weight * (decay_factor ** i)
        
        momentum += price_change * weight
        total_weight += weight
    
    return momentum / total_weight if total_weight > 0 else 0
```

### Market Maker Activity Detection

#### 8. Market Maker Presence Indicator
Identifies market maker activity patterns in Indian stocks:

```python
def detect_market_maker_activity(order_flow, trade_data, time_window=60):
    """
    Detect market maker presence and activity
    Based on order-to-trade ratios and bid-ask posting patterns
    """
    # Calculate order-to-trade ratio
    orders_in_window = len([o for o in order_flow if o['timestamp'] > time.time() - time_window])
    trades_in_window = len([t for t in trade_data if t['timestamp'] > time.time() - time_window])
    
    otr = orders_in_window / max(trades_in_window, 1)
    
    # Analyze bid-ask posting patterns
    bid_posts = len([o for o in order_flow if o['side'] == 'B' and o['type'] == 'NEW'])
    ask_posts = len([o for o in order_flow if o['side'] == 'S' and o['type'] == 'NEW'])
    
    posting_balance = abs(bid_posts - ask_posts) / max(bid_posts + ask_posts, 1)
    
    # Market maker score (lower posting imbalance and higher OTR indicates MM activity)
    mm_score = otr * (1 - posting_balance)
    
    return min(mm_score / 50.0, 1.0)  # Normalize to 0-1 scale
```

#### 9. Adverse Selection Cost
Measures cost of trading against informed traders:

```python
def adverse_selection_cost(trade_prices, trade_sides, quote_midpoint, time_window=300):
    """
    Calculate adverse selection cost for market makers
    Higher values indicate more informed trading
    """
    costs = []
    
    for i, (price, side) in enumerate(zip(trade_prices, trade_sides)):
        if i == 0:
            continue
            
        # Cost is the difference between trade price and subsequent midpoint movement
        future_window_end = min(i + time_window, len(quote_midpoint))
        if future_window_end > i + 1:
            future_midpoint = np.mean(quote_midpoint[i+1:future_window_end])
            
            if side == 'B':  # Buy trade (market taker bought, MM sold)
                cost = future_midpoint - price  # MM loss if price goes up
            else:  # Sell trade (market taker sold, MM bought)
                cost = price - future_midpoint  # MM loss if price goes down
                
            costs.append(cost / price)  # Normalize by price
    
    return np.mean(costs) if costs else 0
```

### Cross-Asset and Sector Signals

#### 10. Sector Momentum Spillover
Captures momentum spillover from sector indices to individual stocks:

```python
def sector_momentum_spillover(stock_price, sector_index, nifty_index, lookback=20):
    """
    Calculate spillover effect from sector and market momentum
    Uses NSE sector indices like Bank Nifty, IT Index, etc.
    """
    # Calculate returns
    stock_return = (stock_price[-1] - stock_price[-lookback]) / stock_price[-lookback]
    sector_return = (sector_index[-1] - sector_index[-lookback]) / sector_index[-lookback]
    market_return = (nifty_index[-1] - nifty_index[-lookback]) / nifty_index[-lookback]
    
    # Calculate betas (simplified)
    stock_sector_beta = np.corrcoef(np.diff(stock_price[-lookback:]), 
                                   np.diff(sector_index[-lookback:]))[0, 1]
    stock_market_beta = np.corrcoef(np.diff(stock_price[-lookback:]), 
                                   np.diff(nifty_index[-lookback:]))[0, 1]
    
    # Expected return based on sector and market moves
    expected_sector_component = stock_sector_beta * sector_return
    expected_market_component = stock_market_beta * (market_return - sector_return)
    expected_return = expected_sector_component + expected_market_component
    
    # Spillover signal is the difference between expected and actual
    spillover_signal = expected_return - stock_return
    
    return spillover_signal
```

#### 11. Currency Impact Factor
Incorporates INR movement impact on stock prices:

```python
def currency_impact_factor(stock_sector, usd_inr_rate, historical_rates, lookback=10):
    """
    Calculate currency impact on stock prices
    Different sectors have different USD-INR sensitivities
    """
    # Sector-specific USD-INR sensitivities (based on historical analysis)
    sector_sensitivities = {
        'IT': -0.8,         # IT benefits from weaker INR
        'PHARMA': -0.6,     # Pharma exports benefit
        'METALS': 0.4,      # Metals hurt by weaker INR (imports)
        'AUTO': 0.3,        # Auto sector mixed impact
        'BANKING': 0.1,     # Banking minimal direct impact
        'FMCG': 0.2,        # FMCG slight negative (input costs)
        'ENERGY': 0.5       # Energy hurt by weaker INR
    }
    
    # Calculate INR movement
    inr_return = (usd_inr_rate - historical_rates[-lookback]) / historical_rates[-lookback]
    
    # Apply sector sensitivity
    sensitivity = sector_sensitivities.get(stock_sector, 0.0)
    currency_factor = sensitivity * inr_return
    
    return currency_factor
```

### News and Sentiment Factors

#### 12. Earnings Surprise Predictor
Predicts earnings surprises using order flow patterns:

```python
def earnings_surprise_predictor(order_flow, volume_profile, days_to_earnings):
    """
    Predict earnings surprise probability using microstructure data
    Based on informed trading patterns before earnings
    """
    if days_to_earnings > 10:
        return 0  # Too far from earnings
    
    # Analyze unusual order flow patterns
    avg_volume = np.mean(volume_profile[-20:])  # 20-day average
    recent_volume = np.mean(volume_profile[-3:])  # 3-day recent
    volume_surge = recent_volume / avg_volume
    
    # Analyze order size distribution
    large_orders = len([o for o in order_flow if o['quantity'] > np.percentile([o['quantity'] for o in order_flow], 90)])
    total_orders = len(order_flow)
    large_order_ratio = large_orders / total_orders
    
    # Time decay factor (closer to earnings = higher weight)
    time_factor = 1.0 - (days_to_earnings / 10.0)
    
    # Combine signals
    surprise_score = (0.6 * np.log(volume_surge) + 0.4 * large_order_ratio) * time_factor
    
    return np.tanh(surprise_score)  # Bounded output
```

#### 13. Regulatory Impact Factor
Captures impact of regulatory announcements and policy changes:

```python
def regulatory_impact_factor(sector, announcement_type, announcement_time, current_time):
    """
    Calculate impact of regulatory announcements
    SEBI, RBI, and government policy impacts on different sectors
    """
    # Time decay of regulatory impact (in hours)
    hours_since = (current_time - announcement_time).total_seconds() / 3600
    time_decay = np.exp(-hours_since / 24)  # 24-hour half-life
    
    # Sector-specific regulatory sensitivities
    regulatory_impacts = {
        'BANKING': {
            'RBI_RATE_CUT': 0.8,
            'RBI_RATE_HIKE': -0.8,
            'CRR_CHANGE': 0.5,
            'SLR_CHANGE': 0.3,
            'NPA_NORMS': -0.6
        },
        'TELECOM': {
            'SPECTRUM_AUCTION': -0.4,
            'TARIFF_REGULATION': -0.5,
            'FDI_POLICY': 0.6
        },
        'PHARMA': {
            'DRUG_PRICING': -0.7,
            'FDA_APPROVAL': 0.8,
            'IMPORT_DUTY': -0.4
        },
        'IT': {
            'H1B_VISA': -0.6,
            'DATA_LOCALIZATION': -0.3,
            'TAX_POLICY': -0.5
        }
    }
    
    impact_magnitude = regulatory_impacts.get(sector, {}).get(announcement_type, 0.0)
    
    return impact_magnitude * time_decay
```

### Risk and Volatility Factors

#### 14. Intraday VaR Predictor
Predicts Value at Risk for intraday positions:

```python
def intraday_var_predictor(price_series, volume_series, confidence_level=0.95):
    """
    Predict intraday VaR using GARCH-like approach
    Adapted for Indian market intraday patterns
    """
    # Calculate returns
    returns = np.diff(price_series) / price_series[:-1]
    
    # Volume-weighted returns (higher volume = more reliable)
    volume_weights = volume_series[1:] / np.sum(volume_series[1:])
    weighted_returns = returns * volume_weights
    
    # Calculate rolling volatility with volume weighting
    window = min(50, len(weighted_returns))
    rolling_vol = pd.Series(weighted_returns).rolling(window).std()
    
    # Current volatility estimate
    current_vol = rolling_vol.iloc[-1]
    
    # VaR calculation
    z_score = stats.norm.ppf(1 - confidence_level)
    var_estimate = current_vol * z_score
    
    return var_estimate
```

#### 15. Liquidity Risk Factor
Measures liquidity risk using order book characteristics:

```python
def liquidity_risk_factor(order_book, trade_volume, time_window=300):
    """
    Calculate liquidity risk based on order book depth and trading activity
    Higher values indicate higher liquidity risk
    """
    # Calculate order book depth
    bid_depth = sum(level.quantity for level in order_book.bid_levels[:10])
    ask_depth = sum(level.quantity for level in order_book.ask_levels[:10])
    total_depth = bid_depth + ask_depth
    
    # Calculate spread
    if order_book.bid_levels[0].price > 0 and order_book.ask_levels[0].price > 0:
        spread = order_book.ask_levels[0].price - order_book.bid_levels[0].price
        mid_price = (order_book.bid_levels[0].price + order_book.ask_levels[0].price) / 2
        relative_spread = spread / mid_price
    else:
        relative_spread = 1.0  # Maximum risk if no quotes
    
    # Calculate volume rate
    avg_volume = np.mean(trade_volume) if len(trade_volume) > 0 else 1
    current_volume = trade_volume[-1] if len(trade_volume) > 0 else 0
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
    
    # Combine factors (higher spread and lower depth/volume = higher risk)
    liquidity_risk = relative_spread * 100 + (1 / max(total_depth, 1)) + (1 / max(volume_ratio, 0.1))
    
    return min(liquidity_risk, 10.0)  # Cap at reasonable maximum
```

### Performance Attribution Factors

#### 16. Order Flow Toxicity
Measures the quality of order flow (toxic vs. benign):

```python
def order_flow_toxicity(trade_prices, trade_volumes, quote_changes, time_window=60):
    """
    Calculate order flow toxicity - measures informed vs. uninformed trading
    Higher toxicity indicates more informed (toxic) flow
    """
    toxicity_scores = []
    
    for i in range(len(trade_prices)):
        if i == 0:
            continue
            
        # Price impact of the trade
        price_impact = abs(trade_prices[i] - trade_prices[i-1]) / trade_prices[i-1]
        
        # Volume factor (larger trades have higher impact)
        volume_factor = np.log(1 + trade_volumes[i] / np.mean(trade_volumes))
        
        # Quote change factor (more quote changes = more toxic flow)
        recent_quote_changes = sum(1 for qc in quote_changes if qc['timestamp'] > time.time() - time_window)
        quote_factor = np.log(1 + recent_quote_changes / 10)
        
        # Combined toxicity score
        toxicity = price_impact * volume_factor * quote_factor
        toxicity_scores.append(toxicity)
    
    return np.mean(toxicity_scores) if toxicity_scores else 0
```

### Implementation Framework

#### Alpha Factor Pipeline
```python
class IndianAlphaFactorEngine:
    """
    Comprehensive alpha factor calculation engine for Indian markets
    """
    
    def __init__(self):
        self.factors = {}
        self.factor_cache = {}
        self.update_frequency = {
            'tick': ['vwoi', 'depth_pressure', 'normalized_spread'],
            'minute': ['momentum', 'mm_activity', 'adverse_selection'],
            'daily': ['sector_spillover', 'regulatory_impact']
        }
    
    def calculate_all_factors(self, market_data, fundamental_data, news_data):
        """Calculate all alpha factors for given market state"""
        factors = {}
        
        # Microstructure factors
        factors['vwoi'] = calculate_vwoi(market_data['order_book'])
        factors['depth_pressure'] = depth_pressure(market_data['order_book'])
        factors['normalized_spread'] = normalized_spread(
            market_data['best_bid'], 
            market_data['best_ask'], 
            market_data['tick_size']
        )
        
        # Flow-based factors
        factors['fii_dii_adjusted'] = calculate_fii_dii_adjusted_imbalance(
            market_data['order_flow'],
            fundamental_data['fii_flow'],
            fundamental_data['dii_flow']
        )
        
        # Momentum factors
        factors['intraday_momentum'] = intraday_momentum_india(
            market_data['prices'],
            market_data['volumes'],
            market_data['timestamps']
        )
        
        # Risk factors
        factors['liquidity_risk'] = liquidity_risk_factor(
            market_data['order_book'],
            market_data['trade_volume']
        )
        
        return factors
    
    def get_factor_importance(self, returns, lookback_days=30):
        """Calculate factor importance scores"""
        # Implementation would use statistical methods to rank factors
        pass
```

This comprehensive alpha factor library provides Indian market-specific predictive signals that can be used in HFT strategies, taking into account the unique characteristics of NSE and BSE markets, regulatory environment, and local institutional flow patterns.