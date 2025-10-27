# =============================================================================
# scripts/quant_prepare_training_data.py
#
# Description:
#   This is the definitive data preparation script. It removes ALL dependencies
#   on 'yfinance' and 'pandas-ta', sourcing data from Alpaca and calculating
#   all features manually with pandas/numpy for maximum stability and consistency.
# =============================================================================

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import random
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# --- Configuration ---
TRAINING_HORIZON = 3
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import FEATURE_COLS_DEFAULT

# =========================
# Feature Engineering (with Alpaca & Pure Pandas)
# =========================
def create_features(ticker="BTC/USD", history_days=5*365):
    """
    (PURE PANDAS VERSION)
    Downloads historical data from Alpaca and manually calculates all features
    without relying on the pandas-ta library.
    """
    print("--- Preparing data using Alpaca API & Pure Pandas ---")
    
    load_dotenv()
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    if not API_KEY or not SECRET_KEY:
        raise ValueError("FATAL ERROR: Alpaca API keys must be set in your .env file.")

    api = tradeapi.REST(API_KEY, SECRET_KEY, base_url="https://paper-api.alpaca.markets", api_version='v2')
    
    start_date = (datetime.now() - timedelta(days=history_days)).strftime('%Y-%m-%d')
    
    print(f"Fetching daily data for {ticker} from {start_date} to today...")
    df = api.get_crypto_bars(ticker, '1Day', start=start_date).df

    if df.empty:
        raise ValueError("Failed to fetch data from Alpaca.")
    
    print(f"Successfully fetched {len(df)} data points from Alpaca.")
    df.rename(columns={'Close': 'close'}, inplace=True, errors='ignore')

    # --- MANUAL FEATURE CALCULATION (NO PANDAS-TA) ---
    print("Calculating SLOW features manually...")

    # Feature 1: RSI (Rsi_14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['Rsi_14'] = 100 - (100 / (1 + rs))

    # Feature 2: Rate of Change (Roc_15)
    df['Roc_15'] = df['close'].pct_change(periods=15)

    # Feature 3: MACD Histogram (Macdh_12_26_9)
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['Macdh_12_26_9'] = macd_line - signal_line

    # Features 4 & 5: SMAs and custom ratios
    sma20 = df['close'].rolling(window=20).mean()
    sma100 = df['close'].rolling(window=100).mean()
    df['Price_vs_sma20'] = df['close'] / sma20
    df['Sma20_vs_sma100'] = sma20 / sma100

    # Feature 6: Bollinger Band Position (Bb_position)
    std20 = df['close'].rolling(window=20).std()
    bbu = sma20 + (std20 * 2)
    bbl = sma20 - (std20 * 2)
    bb_range = bbu - bbl
    bb_range[bb_range == 0] = np.nan
    df['Bb_position'] = (df['close'] - bbl) / bb_range
    
    # Feature 7: ATR_14  ---
    high_low = df['high'] - df['low']
    high_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_prev_close = np.abs(df['low'] - df['close'].shift(1))
    tr_df = pd.concat([high_low, high_prev_close, low_prev_close], axis=1)
    true_range = tr_df.max(axis=1)
    df['Atr_14'] = true_range.ewm(alpha=1/14, adjust=False).mean()
    
    # Finalize column names
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True, errors='ignore')
    df.dropna(inplace=True)
    
    print(f"Data with all features shape: {df.shape}")
    return df

# =============================================================================
# --- Training Data Generation & Augmentation ---
# (These functions remain unchanged)
# =============================================================================
def generate_training_data(featured_data, horizon, feature_cols):
    print(f"\nGenerating training data with a {horizon}-day profit horizon...")
    SHORT_RISK_PENALTY = 0.002
    rows = []
    for i in tqdm(range(len(featured_data) - horizon), desc="Generating Samples"):
        context = featured_data.iloc[i]
        future_close = featured_data.iloc[i + horizon]["Close"]
        profit_change = (future_close - context["Close"]) / context["Close"]
        for action_type in ["BUY", "SELL", "SHORT", "HOLD"]:
            action_amount = np.random.rand()
            outcome, code = 0.0, 0
            if action_type == "BUY":
                outcome, code = profit_change * action_amount, 1
            elif action_type == "SHORT":
                base_outcome = -profit_change * action_amount
                outcome, code = base_outcome - (action_amount * SHORT_RISK_PENALTY), -1
            outcome = np.clip(outcome, -0.15, 0.15)
            row = { **context[feature_cols].to_dict(), "Close": context["Close"], "action_type": action_type,
                    "action_amount": action_amount, "outcome_profit_change": outcome, "action_type_code": code }
            rows.append(row)
    return pd.DataFrame(rows)

def augment_training_data(training_df, featured_data, horizon, feature_cols):
    """
    This function augments the training data with high-conviction samples.
    This version is updated to use the new 'slow' features (e.g., Roc_15).
    """
    print("\nAugmenting training data...")
    aug_rows = []

    mask_short = (featured_data["Rsi_14"] > 75) & (featured_data["Roc_15"] > 0)
    indices_short = featured_data[mask_short].index
    
    mask_buy = (featured_data["Rsi_14"] < 25) & (featured_data["Roc_15"] < 0)
    indices_buy = featured_data[mask_buy].index
    
    for pos in tqdm(range(len(featured_data) - horizon), desc="Augmenting Data"):
        profit_change = (featured_data.iloc[pos + horizon]["Close"] - featured_data.iloc[pos]["Close"]) / featured_data.iloc[pos]["Close"]
        context = featured_data.iloc[pos][feature_cols].to_dict()
        if pos in indices_buy and random.random() < 0.20:
            aug_rows.append({ **context, "Close": featured_data.iloc[pos]["Close"], "action_type": "BUY", "action_amount": 1.0, "outcome_profit_change": np.clip(profit_change, -0.15, 0.15), "action_type_code": 1 })
        if pos in indices_short and random.random() < 0.20:
            aug_rows.append({ **context, "Close": featured_data.iloc[pos]["Close"], "action_type": "SHORT", "action_amount": 1.0, "outcome_profit_change": np.clip(-profit_change, -0.15, 0.15), "action_type_code": -1 })
            
    if aug_rows:
        aug_df = pd.DataFrame(aug_rows)
        training_df = pd.concat([training_df, aug_df], ignore_index=True)
        print(f"🔧 Added {len(aug_df)} augmented samples.")
        
    return training_df


# =========================
# Main Execution Block
# =========================
if __name__ == "__main__":
    featured_data = create_features()
    if featured_data is not None:
        df_gen = generate_training_data(featured_data, horizon=TRAINING_HORIZON, feature_cols=FEATURE_COLS_DEFAULT)
        df_aug = augment_training_data(df_gen, featured_data, horizon=TRAINING_HORIZON, feature_cols=FEATURE_COLS_DEFAULT)
        output_path = "causal_training_data_balanced.csv"
        df_aug.to_csv(output_path, index=False)
        print(f"\n✅ Final training data (Pure Alpaca & Pandas) successfully saved to '{output_path}'")