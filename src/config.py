# config.py
"""
Configuration file for Chimera Trading Bot
Contains feature definitions and global settings
"""

# Feature columns used by the Causal Engine
# These MUST match exactly what was used during training
FEATURE_COLS_DEFAULT = [
    'Rsi_14',           # Relative Strength Index (14 periods)
    'Macdh_12_26_9',    # MACD Histogram
    'Roc_15',           # Rate of Change (15 periods)
    'Price_vs_sma20',   # Current price vs 20-day SMA ratio
    'Sma20_vs_sma100',  # 20-day SMA vs 100-day SMA ratio
    'Bb_position',      # Bollinger Band position (0-1)
    'Atr_14'            # Average True Range (14 periods) - for volatility
]

# Trading parameters
TRADING_CONFIG = {
    'symbol': 'BTC/USD',
    'symbol_alpaca': 'BTCUSD',
    'min_trade_value': 10.0,  # Minimum trade size in USD
    'history_days': 400,       # Days of history to fetch for indicators
}

# Risk management
RISK_CONFIG = {
    'max_position_ratio': 0.95,   # Max 95% of portfolio in long positions
    'max_short_ratio': 0.50,       # Max 50% of portfolio in short positions
    'allow_shorting': True,
    'max_action_amount': 1.0
}

# API Configuration
API_CONFIG = {
    'base_url': 'https://paper-api.alpaca.markets',
    'api_version': 'v2',
    'rate_limit_pause': 0.2  # Seconds to pause between API calls
}

# Scheduler configuration
SCHEDULER_CONFIG = {
    'run_time': "00:00",  # UTC time to run daily (format: "HH:MM")
    'run_on_startup': True  # Whether to run once immediately on startup
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'update_interval': 30,  # Seconds between dashboard updates
    'chart_days': 7,         # Days to show on price chart
    'output_path': 'live_dashboard.png'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'trader_log': 'trader_activity.log',
    'dashboard_log': 'dashboard_activity.log',
    'stream_log': 'stream_activity.log'
}