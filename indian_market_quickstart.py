#!/usr/bin/env python
"""
Quick start example for Indian market ML4T
Run this script to test your setup and see a basic example
"""

def test_indian_market_setup():
    """Test basic setup for Indian market data access"""
    
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        print("✅ Required packages installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install required packages:")
        print("pip install yfinance pandas numpy matplotlib seaborn scikit-learn")
        return False
    
    try:
        # Test downloading Indian stock data
        print("\n📊 Testing Indian stock data access...")
        
        # Download sample data
        symbol = 'RELIANCE.NS'
        data = yf.download(symbol, period='1mo', progress=False)
        
        if len(data) > 0:
            print(f"✅ Successfully downloaded {len(data)} days of data for {symbol}")
            print(f"   Price range: ₹{data['Close'].min():.2f} - ₹{data['Close'].max():.2f}")
            
            # Calculate basic features
            data['Returns'] = data['Close'].pct_change()
            data['SMA_5'] = data['Close'].rolling(5).mean()
            
            latest_price = data['Close'][-1]
            latest_return = data['Returns'][-1]
            
            print(f"   Latest price: ₹{latest_price:.2f}")
            print(f"   Latest daily return: {latest_return*100:.2f}%")
            
            # Test ML preparation
            feature_data = data[['Close', 'Volume', 'Returns', 'SMA_5']].dropna()
            print(f"   Prepared {len(feature_data)} rows for ML analysis")
            
            return True
        else:
            print(f"❌ No data downloaded for {symbol}")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing Indian market data: {e}")
        print("This could be due to:")
        print("  • Internet connection issues")
        print("  • Yahoo Finance API limitations")
        print("  • Market hours (Indian markets: 9:15 AM - 3:30 PM IST)")
        return False

def show_next_steps():
    """Show next steps for users"""
    print("\n🚀 Next Steps:")
    print("1. Read the comprehensive guide: INDIAN_MARKETS_GUIDE.md")
    print("2. Run the demo notebook: 02_market_and_fundamental_data/03_data_providers/06_indian_market_data_demo.ipynb")
    print("3. Adapt existing ML4T examples with Indian stock symbols:")
    print("   • Replace 'AAPL' with 'RELIANCE.NS'")
    print("   • Replace 'MSFT' with 'TCS.NS'")
    print("   • Replace '^GSPC' with '^NSEI' (S&P 500 → NIFTY 50)")
    print("4. Consider Indian market specific factors:")
    print("   • Trading hours: 9:15 AM - 3:30 PM IST")
    print("   • Currency: Indian Rupees (INR)")
    print("   • Settlement: T+2 cycle")

def main():
    print("Indian Market ML4T Quick Start")
    print("=" * 40)
    
    success = test_indian_market_setup()
    
    if success:
        print("\n🎉 Setup successful! You're ready to use ML4T with Indian markets.")
        show_next_steps()
    else:
        print("\n⚠️  Setup incomplete. Please resolve the issues above.")
        print("For help, see: INDIAN_MARKETS_GUIDE.md")

if __name__ == "__main__":
    main()