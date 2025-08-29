# Using Machine Learning for Trading with Indian Stock Markets

This guide explains how to adapt the machine learning techniques in this repository for Indian equity markets. All the core concepts, algorithms, and workflows demonstrated throughout the book can be applied to Indian stocks with the appropriate data sources and market-specific considerations.

## Quick Start

**For immediate testing:** Run the quickstart script to verify your setup:
```bash
python indian_market_quickstart.py
```

For an immediate hands-on example, see the [Indian Market Data Demo](02_market_and_fundamental_data/03_data_providers/06_indian_market_data_demo.ipynb) notebook which demonstrates:
- Downloading Indian stock data using yfinance
- Feature engineering with Indian market data
- Applying ML models to predict Indian stock returns
- Market analysis specific to Indian equities

## Data Sources for Indian Markets

### 1. Yahoo Finance (Recommended for Beginners)

**Advantages:**
- Free and reliable
- Works with existing yfinance examples in the repository
- No API keys required
- Covers NSE and BSE stocks

**Usage:**
```python
import yfinance as yf

# NSE stocks (National Stock Exchange)
reliance = yf.download('RELIANCE.NS', start='2020-01-01', end='2023-01-01')
tcs = yf.download('TCS.NS', start='2020-01-01', end='2023-01-01')

# BSE stocks (Bombay Stock Exchange)
reliance_bse = yf.download('RELIANCE.BO', start='2020-01-01', end='2023-01-01')

# Market indices
nifty50 = yf.download('^NSEI', start='2020-01-01', end='2023-01-01')  # NIFTY 50
sensex = yf.download('^BSESN', start='2020-01-01', end='2023-01-01')  # SENSEX
```

**Stock Symbol Format:**
- NSE stocks: Add `.NS` suffix (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`)
- BSE stocks: Add `.BO` suffix (e.g., `RELIANCE.BO`, `500325.BO`)

### 2. Alpha Vantage

**Setup:**
```python
import requests
import pandas as pd

# Get free API key from https://www.alphavantage.co/support/#api-key
api_key = 'your_api_key_here'

def get_indian_stock_data(symbol, api_key):
    # Note: Use BSE symbol format for Alpha Vantage
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}.BSE&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data
```

### 3. Quandl (Now Nasdaq Data Link)

Some Quandl datasets include Indian market data. Check available datasets at https://data.nasdaq.com/

### 4. Professional Data Providers

For production trading systems:
- **Zerodha Kite Connect**: Real-time and historical data API
- **Angel Broking API**: Comprehensive market data
- **NSE/BSE Direct APIs**: Official exchange data
- **Thomson Reuters Eikon**: Professional-grade data
- **Bloomberg Terminal**: Institutional access

## Indian Market Characteristics

### Trading Hours
- **Regular Session:** 9:15 AM - 3:30 PM IST (Monday to Friday)
- **Pre-market:** 9:00 AM - 9:15 AM IST
- **Post-market:** 3:40 PM - 4:00 PM IST

### Market Holidays
Indian markets are closed on national holidays including:
- Republic Day (January 26)
- Independence Day (August 15)
- Gandhi Jayanti (October 2)
- Diwali (varies)
- Holi (varies)
- Good Friday
- And other regional/religious holidays

### Currency and Settlement
- **Currency:** Indian Rupees (INR)
- **Settlement:** T+2 (Trade plus 2 days)
- **Lot Sizes:** Vary by stock for derivatives trading

### Market Indices
- **NIFTY 50** (`^NSEI`): Top 50 companies by market cap
- **SENSEX** (`^BSESN`): Bombay Stock Exchange benchmark (30 stocks)
- **NIFTY BANK** (`^NSEBANK`): Banking sector index
- **NIFTY IT** (`^CNXIT`): Information Technology sector

## Adapting Book Examples for Indian Markets

### Chapter 2: Market & Fundamental Data
- Replace US stock symbols with Indian symbols (add `.NS` or `.BO`)
- Use Indian market indices instead of S&P 500/NASDAQ
- Consider Indian trading hours in data processing

### Chapter 4: Alpha Factor Research
- All technical indicators work the same way
- Consider sector-specific factors (IT export dependency on USD/INR)
- Include India-specific fundamental ratios

### Chapter 7: Linear Models
- Use Indian stock returns as dependent variables
- Include Indian macro indicators (repo rate, inflation, monsoon data)

### Chapter 8: ML4T Workflow
- Adapt Zipline bundle creation for Indian data
- Modify trading calendar for Indian market hours
- Adjust for Indian settlement and transaction costs

### Chapter 11: Random Forests
- The Japanese equity example can be directly adapted for Indian stocks
- Consider using NIFTY sectors instead of Japanese sector classifications

### Chapter 12: Gradient Boosting
- All gradient boosting techniques apply directly
- Consider Indian market microstructure for intraday strategies

## Sample Indian Stock Universe

### Large Cap (NIFTY 50 components)
```python
large_cap_stocks = [
    'RELIANCE.NS',    # Reliance Industries
    'TCS.NS',         # Tata Consultancy Services
    'HDFCBANK.NS',    # HDFC Bank
    'INFY.NS',        # Infosys
    'HINDUNILVR.NS',  # Hindustan Unilever
    'ITC.NS',         # ITC Limited
    'SBIN.NS',        # State Bank of India
    'BHARTIARTL.NS',  # Bharti Airtel
    'KOTAKBANK.NS',   # Kotak Mahindra Bank
    'LT.NS',          # Larsen & Toubro
]
```

### Sector-wise Examples
```python
sectors = {
    'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'TECHM.NS'],
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'LUPIN.NS'],
    'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS']
}
```

## Feature Engineering for Indian Markets

### Market-Specific Features
```python
def indian_market_features(data):
    """Add India-specific features"""
    features = {}
    
    # USD/INR exchange rate impact (important for IT/Pharma exporters)
    usdinr = yf.download('USDINR=X', start=data.index[0], end=data.index[-1])
    features['usdinr_change'] = usdinr['Close'].pct_change()
    
    # Crude oil prices (impacts energy and transportation)
    crude = yf.download('CL=F', start=data.index[0], end=data.index[-1])
    features['crude_change'] = crude['Close'].pct_change()
    
    # NIFTY 50 relative performance
    nifty = yf.download('^NSEI', start=data.index[0], end=data.index[-1])
    features['nifty_returns'] = nifty['Close'].pct_change()
    
    return features
```

### Fundamental Data Considerations
- **P/E Ratios:** Generally higher than developed markets
- **ROE/ROA:** Consider Indian accounting standards
- **Debt-to-Equity:** Important given Indian corporate debt levels
- **Promoter Holdings:** Unique to Indian markets (family/founder ownership)

## Risk Management for Indian Markets

### Market-Specific Risks
- **Currency Risk:** USD/INR volatility affects export companies
- **Regulatory Risk:** SEBI policy changes, tax modifications
- **Liquidity Risk:** Higher impact costs for smaller stocks
- **Political Risk:** Policy uncertainty around elections

### Risk Metrics Adjustments
```python
# Adjust for Indian market volatility patterns
def indian_risk_metrics(returns):
    # Indian markets tend to be more volatile
    volatility_adjustment = 1.2  # Typical multiplier for emerging markets
    
    adjusted_vol = returns.std() * volatility_adjustment
    
    # Consider rupee depreciation trend
    currency_risk = 0.05  # Annual rupee depreciation assumption
    
    return {
        'adjusted_volatility': adjusted_vol,
        'currency_risk': currency_risk
    }
```

## Implementation Examples

### 1. Simple Moving Average Strategy
```python
import yfinance as yf
import pandas as pd

# Download Indian stock data
data = yf.download('RELIANCE.NS', start='2022-01-01', end='2023-01-01')

# Calculate signals
data['SMA_20'] = data['Close'].rolling(20).mean()
data['SMA_50'] = data['Close'].rolling(50).mean()
data['Signal'] = (data['SMA_20'] > data['SMA_50']).astype(int)

# Calculate returns
data['Returns'] = data['Close'].pct_change()
data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']

print(f"Buy and Hold Return: {(data['Close'][-1]/data['Close'][0] - 1)*100:.2f}%")
print(f"Strategy Return: {data['Strategy_Returns'].cumsum()[-1]*100:.2f}%")
```

### 2. Multi-Stock Momentum Strategy
```python
# Download multiple stocks
stocks = ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'TECHM.NS']
data = yf.download(stocks, start='2022-01-01', end='2023-01-01')

# Calculate momentum scores
for stock in stocks:
    data[('Momentum', stock)] = data[('Close', stock)].pct_change(21)  # 1-month momentum

# Rank and create portfolio
latest_momentum = data['Momentum'].iloc[-1].sort_values(ascending=False)
top_stocks = latest_momentum.head(2).index  # Top 2 momentum stocks

print(f"Current top momentum stocks: {list(top_stocks)}")
```

## Backtesting Considerations

### Transaction Costs
- **Brokerage:** 0.05-0.50% per trade depending on broker
- **Securities Transaction Tax (STT):** 0.1% on equity delivery
- **Exchange Charges:** ~0.003% of trade value
- **SEBI Charges:** ₹10 per crore of trade value

### Market Impact
- **Large Caps:** Low impact (similar to US markets)
- **Mid Caps:** Moderate impact (1-2% for large orders)
- **Small Caps:** High impact (5-10% for large orders)

### Example Transaction Cost Model
```python
def indian_transaction_costs(trade_value, stock_category='large_cap'):
    """Calculate transaction costs for Indian markets"""
    brokerage = trade_value * 0.0005  # 0.05%
    stt = trade_value * 0.001  # 0.1%
    exchange_charges = trade_value * 0.00003  # 0.003%
    
    impact_costs = {
        'large_cap': trade_value * 0.001,    # 0.1%
        'mid_cap': trade_value * 0.01,       # 1%
        'small_cap': trade_value * 0.05      # 5%
    }
    
    total_cost = brokerage + stt + exchange_charges + impact_costs[stock_category]
    return total_cost
```

## Getting Started Checklist

1. **✅ Install required packages**
   ```bash
   pip install yfinance pandas numpy scikit-learn matplotlib seaborn
   ```

2. **✅ Run the Indian Market Demo**
   - Execute [06_indian_market_data_demo.ipynb](02_market_and_fundamental_data/03_data_providers/06_indian_market_data_demo.ipynb)

3. **✅ Adapt existing notebooks**
   - Replace US symbols with Indian symbols (add .NS or .BO)
   - Update market indices to NIFTY/SENSEX
   - Adjust trading hours and calendars

4. **✅ Set up data pipeline**
   - Choose primary data source (yfinance for beginners)
   - Set up data download scripts
   - Create Indian market universe

5. **✅ Implement features**
   - Add India-specific technical indicators
   - Include macro-economic features (USD/INR, crude oil)
   - Consider sector-specific factors

6. **✅ Backtest and validate**
   - Use realistic transaction costs
   - Account for market impact
   - Validate with out-of-sample testing

## Resources

### Official Sources
- **NSE**: https://www.nseindia.com/
- **BSE**: https://www.bseindia.com/
- **SEBI**: https://www.sebi.gov.in/
- **RBI**: https://www.rbi.org.in/

### Data Providers
- **Yahoo Finance**: Free, good for beginners
- **Alpha Vantage**: https://www.alphavantage.co/
- **Quandl/Nasdaq Data Link**: https://data.nasdaq.com/
- **Zerodha Kite**: https://kite.trade/
- **Angel Broking**: https://smartapi.angelbroking.com/

### News and Research
- **Economic Times**: https://economictimes.indiatimes.com/
- **Moneycontrol**: https://www.moneycontrol.com/
- **Business Standard**: https://www.business-standard.com/

## Conclusion

All the machine learning techniques demonstrated in this repository can be successfully applied to Indian equity markets. The key is understanding the market-specific characteristics and adapting the data sources accordingly. Start with the demo notebook and gradually build more sophisticated strategies using the frameworks provided throughout the book.

The Indian market offers unique opportunities with its diverse sectors, emerging market dynamics, and growing retail participation. Combined with the ML techniques in this repository, you have a solid foundation for developing data-driven trading strategies for Indian equities.