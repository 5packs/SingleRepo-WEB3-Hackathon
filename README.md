# 🚀 WEB3 Hackathon - Cryptocurrency Trading System

A comprehensive cryptocurrency trading and analysis system built for Web3 hackathon, featuring automated trading strategies, portfolio optimization, and real-time monitoring capabilities.

## 📋 Project Overview

This project consists of three main modules that work together to provide a complete cryptocurrency trading ecosystem:

- **🔗 Crypto Roostoo API**: Interface with the Roostoo cryptocurrency exchange
- **📊 SMA Prediction**: Advanced trading strategy optimization using Simple Moving Averages
- **🤖 Trading Bot**: Automated monitoring and trading execution

## 🏗️ Project Structure

```
SingleRepo-WEB3-Hackathon/
├── crypto-roostoo-api/          # Exchange API integration
│   ├── balance.py               # Account balance management
│   ├── trades.py               # Trading operations
│   ├── utilities.py            # API utilities and helpers
│   └── manual_api_test.py      # API testing scripts
├── sma-prediction/             # Trading strategy optimization
│   ├── multi_cryptocurrency_optimizer.py  # Main optimizer
│   ├── backtest_sma.py         # Strategy backtesting
│   ├── trading_strategy.py     # Core trading logic
│   ├── prices.py              # Price data fetching
│   └── crypto_data/           # Historical price data (CSV)
├── trading-bot/               # Automated trading bot
│   ├── monitor_bot.py         # Account monitoring
│   ├── purchase_by_value.py   # Value-based purchasing
│   └── logs/                  # Bot execution logs
├── output/                    # Generated analysis results
├── recent_crypto_data/        # Recent market data (JSON)
└── requirements.txt           # Python dependencies
```

## 🔧 Features

### 🔗 Crypto Roostoo API Integration
- **Account Management**: Check balances, view wallet details
- **Trading Operations**: Execute buy/sell orders with authentication
- **Market Data**: Real-time ticker information and exchange details
- **Security**: HMAC-SHA256 signed requests for secure API communication

### 📈 SMA Trading Strategy Optimization
- **Multi-Cryptocurrency Analysis**: Optimize strategies for 24+ cryptocurrencies
- **Parameter Optimization**: Grid search for optimal SMA periods (5-80 periods)
- **Portfolio Allocation**: Intelligent capital distribution based on performance
- **Risk Management**: Sharpe ratio and maximum drawdown analysis
- **Backtesting**: Historical performance validation with detailed metrics

### 🤖 Automated Trading Bot
- **Continuous Monitoring**: Real-time account balance tracking
- **Automated Trading**: Value-based purchase execution
- **Comprehensive Logging**: Detailed execution logs for debugging
- **Error Handling**: Robust error recovery and retry mechanisms

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Roostoo exchange account with API credentials

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd SingleRepo-WEB3-Hackathon
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the `crypto-roostoo-api/` directory:
   ```env
   API_KEY=your_roostoo_api_key
   API_SECRET=your_roostoo_api_secret
   ```

### Usage

#### 🔍 Test API Connection
```bash
cd crypto-roostoo-api
python manual_api_test.py
```

#### 📊 Run Strategy Optimization
```bash
cd sma-prediction
python multi_cryptocurrency_optimizer.py
```

#### 🤖 Start Trading Bot
```bash
cd trading-bot
python monitor_bot.py
```

## 📊 Supported Cryptocurrencies

The system supports analysis and trading for 24+ major cryptocurrencies:
- **Major Coins**: BTC, ETH, BNB, XRP, ADA, SOL, DOGE, DOT, AVAX, LINK
- **DeFi Tokens**: UNI, AAVE, CRV
- **Layer 2**: ARB, APT, SEI
- **And more**: LTC, TRX, TON, NEAR, FIL, ICP, HBAR, FET, SHIB

## 🎯 Key Algorithms

### SMA Strategy Optimization
- **Grid Search**: Tests 56+ parameter combinations per cryptocurrency
- **Performance Metrics**: Return %, Sharpe ratio, maximum drawdown
- **Portfolio Allocation**: Risk-adjusted capital distribution
- **Backtesting**: Historical validation with commission costs

### Risk Management
- **Commission Integration**: 0.1% default trading fees
- **Drawdown Control**: Maximum loss protection
- **Position Sizing**: Portfolio-based allocation strategy

## 📈 Output Files

The system generates several analysis files in the `output/` directory:
- `optimal_sma_parameters.json`: Best SMA parameters per cryptocurrency
- `portfolio_allocation.json`: Recommended capital allocation
- `most_profitable_cryptos.json`: Top performing assets
- `optimized_strategy_parameters.json`: Complete optimization results

## 🔐 Security Features

- **API Authentication**: HMAC-SHA256 signature verification
- **Environment Variables**: Secure credential management
- **Error Handling**: Comprehensive exception management
- **Logging**: Detailed audit trails for all operations

## 📝 Development

### Project Dependencies
- `requests`: HTTP API communication
- `pandas`: Data analysis and manipulation
- `numpy`: Numerical computations
- `matplotlib`: Data visualization
- `python-dotenv`: Environment variable management

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For questions, issues, or contributions, please refer to the individual module READMEs:
- [Trading Bot README](trading-bot/README.md)
- [SMA Optimizer Documentation](sma-prediction/multi_crypto_optimizer_explained.md)

## ⚠️ Disclaimer

This software is for educational and research purposes. Cryptocurrency trading involves significant financial risk. Always:
- Test with small amounts first
- Understand the risks involved
- Never invest more than you can afford to lose
- Verify all trading operations before execution

## 📄 License

This project is developed for the WEB3 Hackathon. Please respect intellectual property and use responsibly.