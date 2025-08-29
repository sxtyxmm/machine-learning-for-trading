# 02 API access to market data

There are several options to access market data via API using Python.

## pandas datareader

The notebook [01_pandas_datareader_demo](01_pandas_datareader_demo.ipynb) presents a few sources built into the pandas library. 
- The `pandas` library enables access to data displayed on websites using the read_html function 
- the related `pandas-datareader` library provides access to the API endpoints of various data providers through a standard interface 

## yfinance: Yahoo! Finance market and fundamental data 

The notebook [yfinance_demo](02_yfinance_demo.ipynb) shows how to use yfinance to download a variety of data from Yahoo! Finance. The library works around the deprecation of the historical data API by scraping data from the website in a reliable, efficient way with a Pythonic API.

## LOBSTER tick data

The notebook [03_lobster_itch_data](03_lobster_itch_data.ipynb) demonstrates the use of order book data made available by LOBSTER (Limit Order Book System - The Efficient Reconstructor), an [online](https://lobsterdata.com/info/WhatIsLOBSTER.php) limit order book data tool that aims to provide easy-to-use, high-quality limit order book data.

Since 2013 LOBSTER acts as a data provider for the academic community, giving access to reconstructed limit order book data for the entire universe of NASDAQ traded stocks. More recently, it started offering a commercial service.

## Qandl

The notebook [03_quandl_demo](03_quandl_demo.ipynb) shows how Quandl uses a very straightforward API to make its free and premium data available. See [documentation](https://www.quandl.com/tools/api) for more details.

## zipline & Qantopian

The notebook [contains the notebook [zipline_data](05_zipline_data.ipynb) briefly introduces the backtesting library `zipline` that we will use throughout this book and show how to access stock price data while running a backtest. For installation please refer to the instructions [here](../../installation).

## Indian Market Data Example

The notebook [06_indian_market_data_demo](06_indian_market_data_demo.ipynb) provides a comprehensive example of how to adapt all the techniques in this book for Indian equity markets, including data access, feature engineering, and machine learning model development.

## Indian Market Data Sources

For trading strategies focused on Indian equity markets, the following data sources and approaches can be used to adapt the techniques shown in this book:

### Yahoo Finance with Indian Stock Symbols

The `yfinance` library demonstrated in [02_yfinance_demo.ipynb](02_yfinance_demo.ipynb) works well with Indian stocks traded on NSE (National Stock Exchange) and BSE (Bombay Stock Exchange):

- **NSE stocks**: Use `.NS` suffix (e.g., 'RELIANCE.NS', 'TCS.NS', 'INFY.NS')
- **BSE stocks**: Use `.BO` suffix (e.g., 'RELIANCE.BO', '500325.BO')

### Alternative Indian Market Data Providers

1. **Alpha Vantage**: Provides Indian stock data through their free and premium APIs
2. **Quandl**: Offers NSE data through various data providers
3. **Zerodha Kite Connect**: API for real-time and historical Indian market data
4. **NSE/BSE Official APIs**: Direct access to exchange data
5. **Economic Times Markets**: Historical data scraping options

### Indian Market Considerations

When adapting the ML techniques for Indian markets, consider:

- **Trading Hours**: IST 9:15 AM - 3:30 PM (Monday-Friday)
- **Market Holidays**: Different from US markets (include Diwali, Holi, etc.)
- **Currency**: All prices in Indian Rupees (INR)
- **Settlement**: T+2 settlement cycle
- **Market Indices**: NIFTY 50, SENSEX instead of S&P 500, NASDAQ

For a complete example of using these techniques with Indian stocks, see the Indian market example notebook.

