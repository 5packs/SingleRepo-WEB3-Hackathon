"""
SMA Trading Strategy Module V2

Enhanced version that combines:
1. Original SMA crossover strategy (from sma-prediction)
2. Market regime detection (from WEB3-Hackathon-PETER copy)
3. Mean reversion strategy for range-bound markets
4. Adaptive decision-making based on market conditions

Built on top of the original trading_strategy.py with minimal layout changes.
"""

import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple


# ============================================================================
# PART 1: MARKET REGIME DETECTION
# ============================================================================

def detect_market_regime(prices: np.ndarray, 
                        ma_window: int = 50,
                        slope_threshold: float = 0.00035,
                        bb_window: int = 20,
                        bb_num_std: float = 2.0,
                        bb_width_threshold: float = 0.014) -> Dict:
    """
    Detect if the market is in a TREND or RANGE regime.
    
    Uses two indicators:
    1. MA Slope: Measures the slope of a moving average
    2. Bollinger Band Width: Measures volatility
    
    Args:
        prices: Array of historical prices
        ma_window: Window for calculating the moving average slope (default: 50)
        slope_threshold: Normalized slope threshold for trend detection
        bb_window: Window for Bollinger Bands calculation (default: 20)
        bb_num_std: Number of standard deviations for BB (default: 2.0)
        bb_width_threshold: Normalized BB width threshold for trend detection
    
    Returns:
        Dictionary with regime information including:
        - regime: 'trend' or 'range'
        - ma_slope_normalized: normalized slope value
        - bb_width_normalized: normalized BB width
        - is_trending_slope: bool
        - is_trending_bb: bool
    """
    
    if len(prices) < ma_window:
        return {
            'regime': 'range',
            'ma_slope_normalized': 0.0,
            'bb_width_normalized': 0.0,
            'is_trending_slope': False,
            'is_trending_bb': False,
            'error': 'Insufficient data'
        }
    
    # Calculate MA Slope
    ma_values = []
    for i in range(len(prices)):
        if i < ma_window - 1:
            ma_values.append(np.nan)
        else:
            window = prices[i - ma_window + 1:i + 1]
            ma_values.append(np.mean(window))
    
    ma_values = np.array(ma_values)
    valid_ma_indices = np.where(~np.isnan(ma_values))[0]
    
    if len(valid_ma_indices) < 20:
        slope_normalized = 0.0
    else:
        recent_ma = ma_values[valid_ma_indices[-20:]]
        x = np.arange(len(recent_ma))
        mean_x = np.mean(x)
        mean_y = np.mean(recent_ma)
        numerator = np.sum((x - mean_x) * (recent_ma - mean_y))
        denominator = np.sum((x - mean_x) ** 2)
        slope = numerator / denominator if denominator > 0 else 0.0
        current_price = prices[-1]
        slope_normalized = abs(slope) / current_price if current_price > 0 else 0.0
    
    # Calculate Bollinger Band Width
    if len(prices) >= bb_window:
        window = prices[-bb_window:]
        middle = np.mean(window)
        std = np.std(window, ddof=1)
        upper = middle + (bb_num_std * std)
        lower = middle - (bb_num_std * std)
        bb_width = upper - lower
        bb_width_normalized = bb_width / prices[-1] if prices[-1] > 0 else 0.0
    else:
        bb_width_normalized = 0.0
    
    # Determine regime
    is_trending_slope = slope_normalized > slope_threshold
    is_trending_bb = bb_width_normalized > bb_width_threshold
    
    regime = 'trend' if (is_trending_slope or is_trending_bb) else 'range'
    
    return {
        'regime': regime,
        'ma_slope_normalized': slope_normalized,
        'bb_width_normalized': bb_width_normalized,
        'is_trending_slope': is_trending_slope,
        'is_trending_bb': is_trending_bb,
        'slope_threshold': slope_threshold,
        'bb_width_threshold': bb_width_threshold
    }


# ============================================================================
# PART 2: MEAN REVERSION STRATEGY (For Range-Bound Markets)
# ============================================================================

def bollinger_band_mean_reversion(prices: np.ndarray,
                                  bb_window: int = 20,
                                  bb_num_std: float = 2.0,
                                  entry_threshold: float = 0.15,
                                  exit_ratio: float = 0.7) -> Dict:
    """
    Mean reversion strategy using Bollinger Bands for range-bound markets.
    
    Strategy Logic:
    - BUY: When price is significantly below lower Bollinger Band
    - SELL: When price recovers to target between middle and upper BB
    - HOLD: Otherwise
    
    Args:
        prices: Array of historical prices
        bb_window: Bollinger Band window (default: 20)
        bb_num_std: Number of standard deviations (default: 2.0)
        entry_threshold: Minimum % below lower BB to trigger BUY (default: 0.15%)
        exit_ratio: Ratio between middle and upper BB for exit (default: 0.7)
    
    Returns:
        Dictionary with trading signal and BB values
    """
    
    if len(prices) < bb_window:
        return {
            'signal': 'HOLD',
            'bb_upper': np.nan,
            'bb_middle': np.nan,
            'bb_lower': np.nan,
            'error': 'Insufficient data'
        }
    
    # Calculate Bollinger Bands
    window = prices[-bb_window:]
    bb_middle = np.mean(window)
    bb_std = np.std(window, ddof=1)
    
    bb_upper = bb_middle + (bb_num_std * bb_std)
    bb_lower = bb_middle - (bb_num_std * bb_std)
    
    current_price = prices[-1]
    exit_target = bb_middle + (exit_ratio * (bb_upper - bb_middle))
    
    distance_to_lower = ((current_price - bb_lower) / bb_lower * 100) if bb_lower > 0 else 0
    
    # Determine signal
    if current_price < bb_lower and distance_to_lower < -entry_threshold:
        signal = 'BUY'
    elif current_price >= exit_target:
        signal = 'SELL'
    else:
        signal = 'HOLD'
    
    return {
        'signal': signal,
        'bb_upper': bb_upper,
        'bb_middle': bb_middle,
        'bb_lower': bb_lower,
        'current_price': current_price,
        'exit_target': exit_target
    }


# ============================================================================
# PART 3: ORIGINAL SMA STRATEGY (Maintained for Trending Markets)
# ============================================================================

def sma_trading_decision(past_prices: List[float], current_price: float, 
                        short_window: int = 25, long_window: int = 45) -> str:
    """
    Make a buy/sell/hold decision based on SMA crossover strategy.
    
    Args:
        past_prices: List of historical prices (should include at least long_window prices)
        current_price: The current price to evaluate
        short_window: Period for short-term SMA (default: 25)
        long_window: Period for long-term SMA (default: 45)
    
    Returns:
        'BUY', 'SELL', or 'HOLD' decision
    """
    # Combine past prices with current price
    all_prices = past_prices + [current_price]
    all_prices = np.array(all_prices, dtype=float)
    
    # Check if we have enough data
    if len(all_prices) < long_window:
        return 'HOLD'
    
    # Calculate SMAs
    short_sma = np.mean(all_prices[-short_window:])
    long_sma = np.mean(all_prices[-long_window:])
    
    # Calculate previous SMAs (if we have enough data)
    if len(all_prices) < long_window + 1:
        return 'HOLD'
    
    prev_short_sma = np.mean(all_prices[-short_window-1:-1])
    prev_long_sma = np.mean(all_prices[-long_window-1:-1])
    
    # Check for crossover signals
    if prev_short_sma <= prev_long_sma and short_sma > long_sma:
        return 'BUY'
    elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
        return 'SELL'
    
    # Additional momentum check
    sma_diff_percent = ((short_sma - long_sma) / long_sma) * 100
    
    if sma_diff_percent > 2.0:
        return 'BUY'
    elif sma_diff_percent < -2.0:
        return 'SELL'
    
    return 'HOLD'


def calculate_sma(prices: List[float], window: int) -> float:
    """
    Calculate Simple Moving Average for a given window.
    
    Args:
        prices: List of price values
        window: Number of periods for the moving average
    
    Returns:
        SMA value as float
    """
    if len(prices) < window:
        return np.mean(prices)
    
    return np.mean(prices[-window:])


def get_sma_signals_info(past_prices: List[float], current_price: float,
                        short_window: int = 10, long_window: int = 50) -> dict:
    """
    Get detailed information about SMA signals and values.
    
    Args:
        past_prices: List of historical prices
        current_price: Current price
        short_window: Short SMA window
        long_window: Long SMA window
    
    Returns:
        Dictionary with SMA values and signal information
    """
    all_prices = past_prices + [current_price]
    all_prices = np.array(all_prices, dtype=float)
    
    if len(all_prices) < long_window + 1:
        return {
            'signal': 'HOLD',
            'short_sma': None,
            'long_sma': None,
            'prev_short_sma': None,
            'prev_long_sma': None,
            'crossover': False,
            'momentum': 0.0,
            'sufficient_data': False
        }
    
    short_sma = np.mean(all_prices[-short_window:])
    long_sma = np.mean(all_prices[-long_window:])
    prev_short_sma = np.mean(all_prices[-short_window-1:-1])
    prev_long_sma = np.mean(all_prices[-long_window-1:-1])
    
    bullish_crossover = prev_short_sma <= prev_long_sma and short_sma > long_sma
    bearish_crossover = prev_short_sma >= prev_long_sma and short_sma < long_sma
    momentum = ((short_sma - long_sma) / long_sma) * 100
    signal = sma_trading_decision(past_prices, current_price, short_window, long_window)
    
    return {
        'signal': signal,
        'short_sma': short_sma,
        'long_sma': long_sma,
        'prev_short_sma': prev_short_sma,
        'prev_long_sma': prev_long_sma,
        'bullish_crossover': bullish_crossover,
        'bearish_crossover': bearish_crossover,
        'momentum': momentum,
        'sufficient_data': True
    }


# ============================================================================
# PART 4: PARAMETER LOADING (Maintained from Original)
# ============================================================================

def _get_output_path(filename: str) -> str:
    """
    Get the absolute path to a file in the project's output directory.
    
    Args:
        filename: Name of the file in the output directory
        
    Returns:
        Absolute path to the file
    """
    # Get the project root directory (two levels up from this file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, "output")
    return os.path.join(output_dir, filename)


def load_optimal_sma_parameters(filepath: str = None) -> Dict:
    """
    Load optimal SMA parameters for all currencies from the optimizer output.
    
    Args:
        filepath: Path to the optimal parameters JSON file (if None, uses default location)
        
    Returns:
        Dictionary containing optimal parameters for each currency
    """
    if filepath is None:
        filepath = _get_output_path("optimal_sma_parameters.json")
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if "parameters" in data:
            print(f"[OK] Loaded optimal SMA parameters for {len(data['parameters'])} currencies")
            return data["parameters"]
        else:
            print("[WARNING] Invalid parameters file format")
            return {}
            
    except FileNotFoundError:
        print(f"[ERROR] Parameters file not found: {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in parameters file: {filepath}")
        return {}


def load_simple_sma_parameters(filepath: str = None) -> Dict:
    """
    Load simplified SMA parameters (just short/long windows) for bot usage.
    
    Args:
        filepath: Path to the simple parameters JSON file (if None, uses default location)
        
    Returns:
        Dictionary with short/long window parameters for each currency
    """
    if filepath is None:
        filepath = _get_output_path("simple_sma_parameters.json")
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"[OK] Loaded simplified SMA parameters for {len(data)} currencies from {filepath}")
        return data
            
    except FileNotFoundError:
        print(f"[ERROR] Simple parameters file not found: {filepath}")
        print(f"[INFO] Using default parameters (25 short, 45 long) for all currencies")
        return {}
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in simple parameters file: {filepath}")
        print(f"[INFO] Using default parameters (25 short, 45 long) for all currencies")
        return {}
    except Exception as e:
        print(f"[ERROR] Unexpected error loading parameters file {filepath}: {e}")
        print(f"[INFO] Using default parameters (25 short, 45 long) for all currencies")
        return {}


def load_optimal_strategy_parameters(filepath: str = None) -> Dict:
    """
    Load optimal strategy parameters including regime detection and mean reversion settings.
    
    Args:
        filepath: Path to the optimal strategy parameters JSON file (if None, uses default location)
        
    Returns:
        Dictionary containing optimal parameters for each currency
    """
    if filepath is None:
        filepath = _get_output_path("optimized_strategy_parameters.json")
    
    print(f"[PARAMS] Loading strategy parameters from: {filepath}")
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if "parameters" in data:
            print(f"[PARAMS] Successfully loaded optimal strategy parameters for {len(data['parameters'])} currencies")
            print(f"[PARAMS] Available currencies: {', '.join(sorted(data['parameters'].keys()))}")
            return data["parameters"]
        else:
            print("[PARAMS] Invalid strategy parameters file format - missing 'parameters' key")
            return {}
            
    except FileNotFoundError:
        print(f"[PARAMS] Strategy parameters file not found: {filepath}")
        print(f"[PARAMS] Will use default regime and mean reversion parameters")
        return {}
    except json.JSONDecodeError as e:
        print(f"[PARAMS] Invalid JSON in strategy parameters file: {filepath}")
        print(f"[PARAMS] JSON Error: {e}")
        return {}
    except Exception as e:
        print(f"[PARAMS] Unexpected error loading strategy parameters: {e}")
        return {}


def get_optimal_parameters_for_currency(currency: str, 
                                       parameters_file: str = None) -> Tuple[int, int]:
    """
    Get optimal SMA parameters for a specific currency.
    
    Args:
        currency: Currency symbol (e.g., 'BTC', 'ETH')
        parameters_file: Path to parameters file (if None, uses default location)
        
    Returns:
        Tuple of (short_window, long_window). Returns (25, 45) as default if not found.
    """
    params = load_simple_sma_parameters(parameters_file)
    
    if currency in params:
        return params[currency]["short"], params[currency]["long"]
    else:
        print(f"[WARNING] No optimal parameters found for {currency}, using defaults (25, 45)")
        return 25, 45


def get_all_strategy_params_for_crypto(currency: str,
                                      parameters_file: str = None) -> Dict:
    """
    Get ALL optimized strategy parameters for a cryptocurrency.
    
    Includes SMA, regime detection, AND mean reversion parameters.
    
    Args:
        currency: Currency symbol (e.g., 'BTC', 'ETH')
        parameters_file: Path to optimized parameters file (if None, uses default location)
    
    Returns:
        Dictionary with all strategy parameters
    """
    params = load_optimal_strategy_parameters(parameters_file)
    
    if currency in params:
        crypto_params = params[currency]
        print(f"[PARAMS] {currency}: Found optimized parameters in JSON file")
        return {
            'slope_threshold': crypto_params.get('slope_threshold', 0.00035),
            'bb_width_threshold': crypto_params.get('bb_width_threshold', 0.014),
            'short_window': crypto_params.get('short_window', 25),
            'long_window': crypto_params.get('long_window', 45),
            'bb_window': crypto_params.get('bb_window', 20),
            'bb_std': crypto_params.get('bb_std', 2.0),
            'entry_threshold': crypto_params.get('entry_threshold', 0.15),
            'exit_ratio': crypto_params.get('exit_ratio', 0.7)
        }
    else:
        # Return defaults if no optimized params found
        print(f"[PARAMS] {currency}: Currency not found in optimized parameters, using defaults")
        return {
            'slope_threshold': 0.00035,
            'bb_width_threshold': 0.014,
            'short_window': 25,
            'long_window': 45,
            'bb_window': 20,
            'bb_std': 2.0,
            'entry_threshold': 0.15,
            'exit_ratio': 0.7
        }


# ============================================================================
# PART 5: ENHANCED ADAPTIVE TRADING DECISION (NEW!)
# ============================================================================

def make_optimized_trading_decision(currency: str, past_prices: List[float], 
                                  current_price: float,
                                  parameters_file: str = None,
                                  strategy_parameters_file: str = None,
                                  use_adaptive: bool = True) -> str:
    """
    Make trading decision using optimized parameters for the specific currency.
    
    ENHANCED VERSION: Now includes regime detection and adaptive strategy selection.
    
    Strategy Logic:
    1. Detect market regime (TREND or RANGE)
    2. If TREND: Use SMA crossover strategy
    3. If RANGE: Use Bollinger Band mean reversion strategy
    4. If not enough data or adaptive disabled: Use standard SMA strategy
    
    Args:
        currency: Currency symbol (e.g., 'BTC', 'ETH')
        past_prices: List of historical prices
        current_price: Current price
        parameters_file: Path to SMA parameters file (if None, uses default location)
        strategy_parameters_file: Path to full strategy parameters file (if None, uses default location)
        use_adaptive: Whether to use adaptive regime-based strategy (default: True)
        
    Returns:
        Trading decision: 'BUY', 'SELL', or 'HOLD'
    """
    
    print(f"[TRADING] Making decision for {currency} (adaptive={use_adaptive})")
    
    # Combine all prices
    all_prices = np.array(past_prices + [current_price], dtype=float)
    print(f"[TRADING] {currency}: Using {len(all_prices)} price points, current price: ${current_price:.2f}")
    
    # If adaptive mode is disabled, use standard SMA strategy
    if not use_adaptive:
        print(f"[TRADING] {currency}: Adaptive mode disabled, using standard SMA strategy")
        short_window, long_window = get_optimal_parameters_for_currency(currency, parameters_file)
        return sma_trading_decision(past_prices, current_price, short_window, long_window)
    
    # Get all strategy parameters for this currency
    print(f"[TRADING] {currency}: Loading optimized strategy parameters...")
    params = get_all_strategy_params_for_crypto(currency, strategy_parameters_file)
    
    # Check if we successfully loaded optimized parameters
    if params and 'slope_threshold' in params and 'bb_width_threshold' in params:
        print(f"[TRADING] {currency}: Successfully loaded optimized parameters from JSON")
        print(f"[TRADING] {currency}: Slope threshold: {params['slope_threshold']}, BB width threshold: {params['bb_width_threshold']}")
    else:
        print(f"[TRADING] {currency}: Using default parameters (optimized parameters not found)")
    
    # Check if we have enough data for regime detection
    min_required = max(params['long_window'], 50)  # Need at least 50 for regime detection
    
    if len(all_prices) < min_required:
        # Not enough data for adaptive strategy, fall back to standard SMA
        print(f"[TRADING] {currency}: Insufficient data ({len(all_prices)} < {min_required}), falling back to standard SMA")
        short_window, long_window = get_optimal_parameters_for_currency(currency, parameters_file)
        return sma_trading_decision(past_prices, current_price, short_window, long_window)
    
    # Step 1: Detect market regime
    print(f"[TRADING] {currency}: Detecting market regime...")
    regime_info = detect_market_regime(
        all_prices,
        ma_window=50,
        slope_threshold=params['slope_threshold'],
        bb_window=params['bb_window'],
        bb_num_std=params['bb_std'],
        bb_width_threshold=params['bb_width_threshold']
    )
    
    regime = regime_info['regime']
    print(f"[TRADING] {currency}: Market regime detected: {regime.upper()}")
    print(f"[TRADING] {currency}: MA slope normalized: {regime_info['ma_slope_normalized']:.6f} (trending: {regime_info['is_trending_slope']})")
    print(f"[TRADING] {currency}: BB width normalized: {regime_info['bb_width_normalized']:.6f} (trending: {regime_info['is_trending_bb']})")
    
    # Step 2: Apply appropriate strategy based on regime
    if regime == 'trend':
        # Use SMA crossover strategy for trending markets
        print(f"[TRADING] {currency}: Using SMA crossover strategy (windows: {params['short_window']}/{params['long_window']})")
        signal = sma_trading_decision(
            past_prices, 
            current_price, 
            params['short_window'], 
            params['long_window']
        )
    else:  # regime == 'range'
        # Use mean reversion strategy for range-bound markets
        print(f"[TRADING] {currency}: Using Bollinger Band mean reversion strategy")
        print(f"[TRADING] {currency}: BB params - window: {params['bb_window']}, std: {params['bb_std']}, entry: {params['entry_threshold']}, exit: {params['exit_ratio']}")
        mean_rev_result = bollinger_band_mean_reversion(
            all_prices,
            bb_window=params['bb_window'],
            bb_num_std=params['bb_std'],
            entry_threshold=params['entry_threshold'],
            exit_ratio=params['exit_ratio']
        )
        signal = mean_rev_result['signal']
    
    print(f"[TRADING] {currency}: Final decision: {signal}")
    return signal


def make_optimized_trading_decision_with_info(currency: str, past_prices: List[float], 
                                            current_price: float,
                                            parameters_file: str = None,
                                            strategy_parameters_file: str = None,
                                            use_adaptive: bool = True) -> Dict:
    """
    Extended version that returns both decision and detailed information.
    
    Returns:
        Dictionary containing:
        - signal: 'BUY', 'SELL', or 'HOLD'
        - regime: 'trend', 'range', or None
        - strategy_used: 'sma', 'mean_reversion', or 'fallback'
        - regime_info: Full regime detection results (if available)
        - additional strategy-specific metrics
    """
    
    all_prices = np.array(past_prices + [current_price], dtype=float)
    params = get_all_strategy_params_for_crypto(currency, strategy_parameters_file)
    min_required = max(params['long_window'], 50)
    
    if len(all_prices) < min_required or not use_adaptive:
        short_window, long_window = get_optimal_parameters_for_currency(currency, parameters_file)
        signal = sma_trading_decision(past_prices, current_price, short_window, long_window)
        return {
            'signal': signal,
            'regime': None,
            'strategy_used': 'fallback',
            'regime_info': None
        }
    
    # Detect regime
    regime_info = detect_market_regime(
        all_prices,
        ma_window=50,
        slope_threshold=params['slope_threshold'],
        bb_window=params['bb_window'],
        bb_num_std=params['bb_std'],
        bb_width_threshold=params['bb_width_threshold']
    )
    
    regime = regime_info['regime']
    
    # Apply strategy
    if regime == 'trend':
        signal = sma_trading_decision(
            past_prices, 
            current_price, 
            params['short_window'], 
            params['long_window']
        )
        strategy_used = 'sma'
    else:
        mean_rev_result = bollinger_band_mean_reversion(
            all_prices,
            bb_window=params['bb_window'],
            bb_num_std=params['bb_std'],
            entry_threshold=params['entry_threshold'],
            exit_ratio=params['exit_ratio']
        )
        signal = mean_rev_result['signal']
        strategy_used = 'mean_reversion'
    
    return {
        'signal': signal,
        'regime': regime,
        'strategy_used': strategy_used,
        'regime_info': regime_info
    }


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SMA Trading Strategy V2 - Enhanced with Adaptive Regime Detection")
    print("=" * 70)
    
    # Example usage
    print("\nExample Usage:")
    print("-" * 70)
    
    # Simulate trending price data
    trending_prices = [100 + i * 0.5 + np.random.randn() * 0.3 for i in range(100)]
    current_price_trend = trending_prices[-1] + 0.5
    
    print("\n1. TRENDING MARKET Example:")
    decision = make_optimized_trading_decision(
        "BTC", 
        trending_prices[:-1], 
        current_price_trend,
        use_adaptive=True
    )
    print(f"   Adaptive Decision: {decision}")
    
    # Get detailed info
    detailed_info = make_optimized_trading_decision_with_info(
        "BTC",
        trending_prices[:-1],
        current_price_trend,
        use_adaptive=True
    )
    print(f"   Regime: {detailed_info['regime']}")
    print(f"   Strategy Used: {detailed_info['strategy_used']}")
    
    # Simulate range-bound price data
    range_prices = [100 + 5 * np.sin(i * 0.1) + np.random.randn() * 0.2 for i in range(100)]
    current_price_range = range_prices[-1]
    
    print("\n2. RANGE-BOUND MARKET Example:")
    decision = make_optimized_trading_decision(
        "ETH",
        range_prices[:-1],
        current_price_range,
        use_adaptive=True
    )
    print(f"   Adaptive Decision: {decision}")
    
    detailed_info = make_optimized_trading_decision_with_info(
        "ETH",
        range_prices[:-1],
        current_price_range,
        use_adaptive=True
    )
    print(f"   Regime: {detailed_info['regime']}")
    print(f"   Strategy Used: {detailed_info['strategy_used']}")
    
    # Test backward compatibility
    print("\n3. BACKWARD COMPATIBILITY Test:")
    print("   Testing standard SMA decision (non-adaptive)...")
    standard_decision = sma_trading_decision(
        trending_prices[:-1],
        current_price_trend,
        short_window=25,
        long_window=45
    )
    print(f"   Standard SMA Decision: {standard_decision}")
    
    print("\n" + "=" * 70)
    print("✅ Trading Strategy V2 tests complete!")
    print("=" * 70)
    print("\nKey Features:")
    print("  • Maintains original file structure and functions")
    print("  • Adds regime detection (trend vs range)")
    print("  • Includes mean reversion for range-bound markets")
    print("  • Uses SMA crossover for trending markets")
    print("  • Fully backward compatible with existing code")
    print("  • Parameter loading from optimized files")
    print("=" * 70)
