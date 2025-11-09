# Roostoo API Python Client

A Python client for interacting with the Roostoo cryptocurrency exchange API. This client provides functions for account management, trading, and market data retrieval.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
1. Copy `.env.example` to `.env`
2. Fill in your API credentials:
   ```env
   ROOSTOO_API_KEY=your_api_key_here
   ROOSTOO_API_SECRET=your_api_secret_here
   BASE_URL=https://api.roostoo.com
   ```

### Running the Interactive Menu
```bash
python main.py
```

## 📚 API Functions Documentation

### 🔧 Utility Functions

#### `get_server_timestamp()`
**Module:** `utilities.py`  
**Authentication:** None  
**Description:** Gets the current server timestamp to ensure API requests are synchronized.

```python
from utilities import get_server_timestamp

timestamp = get_server_timestamp()
print(f"Server time: {timestamp}")
```

**Returns:** String timestamp in milliseconds

---

#### `check_server_time()`
**Module:** `utilities.py`  
**Authentication:** None  
**Description:** Retrieves the server time information.

```python
from utilities import check_server_time

server_time = check_server_time()
if server_time:
    print(f"Server time: {server_time.get('ServerTime')}")
```

**Returns:** Dictionary with server time data or None on error

---

#### `get_exchange_info()`
**Module:** `utilities.py`  
**Authentication:** None  
**Description:** Gets general exchange information including available trading pairs and system status.

```python
from utilities import get_exchange_info

exchange_info = get_exchange_info()
if exchange_info:
    print(f"Exchange running: {exchange_info.get('IsRunning')}")
    print(f"Available pairs: {list(exchange_info.get('TradePairs', {}).keys())}")
```

**Returns:** Dictionary with exchange information or None on error

---

### 📊 Market Data Functions

#### `get_ticker(pair=None)`
**Module:** `utilities.py`  
**Authentication:** Required (timestamp signature)  
**Description:** Gets market ticker data for all pairs or a specific trading pair.

```python
from utilities import get_ticker

# Get all tickers
all_tickers = get_ticker()

# Get specific pair ticker
btc_ticker = get_ticker(pair="BTC/USD")
if btc_ticker:
    price = btc_ticker.get('Data', {}).get('BTC/USD', {}).get('LastPrice')
    print(f"BTC/USD Price: {price}")
```

**Parameters:**
- `pair` (optional): Trading pair string (e.g., "BTC/USD")

**Returns:** Dictionary with ticker data or None on error

---

### 💰 Account Functions

#### `get_balance()`
**Module:** `balance.py`  
**Authentication:** Required (API key + signature)  
**Description:** Retrieves account balance information for all available assets.

```python
from balance import get_balance

balance = get_balance()
# Function automatically prints formatted balance information
```

**Returns:** Dictionary with complete balance data or None on error

**Example Output:**
```
BNB Free: 0 Locked: 0.1
BTC Free: 0 Locked: 0
USD Free: 49903.72 Locked: 0
```

---

### 📈 Trading Functions

#### `place_order(pair_or_coin, side, quantity, price=None, order_type=None)`
**Module:** `trades.py`  
**Authentication:** Required (API key + signature)  
**Description:** Places a buy or sell order on the exchange.

```python
from trades import place_order

# Market Buy Order
result = place_order(
    pair_or_coin="BTC",      # Can be "BTC" or "BTC/USD"
    side="BUY",              # "BUY" or "SELL"
    quantity=0.001           # Amount to trade
)

# Limit Sell Order
result = place_order(
    pair_or_coin="BTC/USD",
    side="SELL",
    quantity=0.001,
    price=105000.00          # Price triggers LIMIT order type
)

# Explicit Order Type
result = place_order(
    pair_or_coin="ETH",
    side="BUY", 
    quantity=0.1,
    price=4000.00,
    order_type="LIMIT"       # Explicitly set order type
)
```

**Parameters:**
- `pair_or_coin` (str): Asset to trade (e.g., "BTC" or "BTC/USD")
- `side` (str): "BUY" or "SELL"
- `quantity` (float): Amount to trade
- `price` (float, optional): Price for LIMIT orders
- `order_type` (str, optional): "LIMIT" or "MARKET" (auto-detected if not provided)

**Returns:** API response or None on error

**Notes:**
- Order type is auto-detected: price provided = LIMIT, no price = MARKET
- Coin symbols are automatically converted to pairs (e.g., "BTC" → "BTC/USD")

---

#### `query_order(order_id=None, pair=None, pending_only=None)`
**Module:** `trades.py`  
**Authentication:** Required (API key + signature)  
**Description:** Queries existing orders by ID or trading pair.

```python
from trades import query_order

# Query specific order by ID
order = query_order(order_id="12345")

# Query all orders for a pair
orders = query_order(pair="BTC/USD")

# Query only pending orders for a pair
pending_orders = query_order(pair="BTC/USD", pending_only=True)

if orders and orders.get('Success'):
    order_list = orders.get('OrderMatched', [])
    print(f"Found {len(order_list)} orders")
```

**Parameters:**
- `order_id` (str, optional): Specific order ID to query
- `pair` (str, optional): Trading pair to filter by
- `pending_only` (bool, optional): If True, only returns pending orders

**Returns:** Dictionary with order information or None on error

**Notes:**
- Cannot use `order_id` and `pair` parameters together
- `pending_only` only works when `pair` is specified

---

#### `cancel_order(order_id=None, pair=None)`
**Module:** `trades.py`  
**Authentication:** Required (API key + signature)  
**Description:** Cancels orders by ID, trading pair, or all orders.

```python
from trades import cancel_order

# Cancel specific order
result = cancel_order(order_id="12345")

# Cancel all orders for a specific pair
result = cancel_order(pair="BTC/USD")

# Cancel ALL orders (use with caution!)
result = cancel_order()

if result and result.get('Success'):
    cancelled_list = result.get('CanceledList', [])
    print(f"Cancelled {len(cancelled_list)} orders")
```

**Parameters:**
- `order_id` (str, optional): Specific order ID to cancel
- `pair` (str, optional): Trading pair to cancel orders for
- If neither parameter is provided, cancels ALL orders

**Returns:** Dictionary with cancellation results or None on error

**⚠️ Warning:** Calling `cancel_order()` without parameters will cancel ALL open orders!

---

## 🔐 Authentication

All authenticated functions automatically handle:
- Server timestamp synchronization
- HMAC-SHA256 signature generation
- Required headers (RST-API-KEY, MSG-SIGNATURE)

The client uses server timestamps to avoid time synchronization issues between your local machine and the exchange servers.

## 🚨 Error Handling

All functions include comprehensive error handling:
- Network request failures
- API error responses
- Invalid parameter validation
- Automatic error logging with response details

## 📝 Example Usage

```python
from utilities import get_exchange_info, get_ticker
from balance import get_balance
from trades import place_order, query_order, cancel_order

# Check if exchange is running
info = get_exchange_info()
print(f"Exchange status: {info.get('IsRunning') if info else 'Unknown'}")

# Get current BTC price
ticker = get_ticker("BTC/USD")
if ticker:
    price = ticker.get('Data', {}).get('BTC/USD', {}).get('LastPrice')
    print(f"Current BTC price: ${price}")

# Check account balance
balance = get_balance()

# Place a small buy order
order_result = place_order("BTC", "BUY", 0.001, price=100000)

# Check pending orders
pending = query_order(pair="BTC/USD", pending_only=True)
```

## 🎮 Interactive Menu

Run `python main.py` to access an interactive menu that provides:
- Easy testing of all functions
- Environment variable validation
- User-friendly prompts and error handling
- Step-by-step guidance for complex operations

## 📞 Support

For API documentation and support, visit the official Roostoo API documentation or contact their support team.