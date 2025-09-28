# =============================================================================
# quant_prepare_training_data.py
# Description:
# This script downloads data, engineers features, and generates a training set.
# This version includes a robust, unified column name standardization process
# to permanently fix all KeyError issues from library inconsistencies.
# =============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from tqdm import tqdm
import os
import sys
import random

# =========================
# Configuration
# =========================
TRAINING_HORIZON = 3

# --- DYNAMIC PATH TO IMPORT FROM 'src' ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import FEATURE_COLS_DEFAULT

# =========================
# Feature Engineering
# =========================
def create_features(ticker="BTC-USD", start_date="2020-01-01", end_date="2025-01-01"):
    """
    Downloads data and calculates all features with a robust standardization workflow.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    print(f"Original data shape: {df.shape}")

    # --- STEP 1: ADD ALL RAW COLUMNS FIRST ---
    # First, let yfinance and pandas_ta create all columns with their native names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print("Calculating all technical indicators...")
    df.ta.rsi(length=14, append=True)
    df.ta.sma(length=10, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.roc(length=5, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # --- STEP 2: UNIFIED COLUMN NAME STANDARDIZATION (THE BULLETPROOF FIX) ---
    # AFTER all columns are created, clean and standardize them all in one go.
    print("Standardizing all column names for consistency...")
    
    # Create a mapping from old names to new, clean names
    new_columns = {}
    for col in df.columns:
        new_col = str(col)
        # Fix potential duplicate suffixes from libraries (e.g., '_2.0_2.0')
        new_col = new_col.replace('_2.0_2.0', '_2.0')
        # Capitalize every word for a consistent format (e.g., 'price_vs_sma10' -> 'Price_vs_sma10')
        new_col = '_'.join([word.capitalize() for word in new_col.split('_')])
        # Final capitalization for single-word columns like 'Open'
        new_col = new_col.capitalize()
        new_columns[col] = new_col
    
    df = df.rename(columns=new_columns)
    print("Cleaned column names:", df.columns.tolist())
    # --- END OF FIX ---
        
    print("Engineering relational features...")
    # Now we can safely use the standardized, predictable names
    try:
        df['Price_vs_sma10'] = (df['Close'] - df['Sma_10']) / df['Sma_10']
        df['Sma10_vs_sma50'] = (df['Sma_10'] - df['Sma_50']) / df['Sma_50']
        df['Bb_position'] = (df['Close'] - df['Bbl_20_2.0']) / (df['Bbu_20_2.0'])
    except KeyError as e:
        print(f"FATAL ERROR: A required column is missing after standardization: {e}")
        return None

    df = df.dropna().copy()
    print(f"Data with all features shape: {df.shape}")
    return df

# =========================
# Training Data Generation & Augmentation
# =========================
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
    print("\nAugmenting training data...")
    aug_rows = []
    # Using the standardized column names
    mask_short = (featured_data["Rsi_14"] > 75) & (featured_data["Roc_5"] < 0)
    indices_short = featured_data[mask_short].index
    mask_buy = (featured_data["Rsi_14"] < 25) & (featured_data["Roc_5"] > 0)
    indices_buy = featured_data[mask_buy].index
    for pos in tqdm(range(len(featured_data) - horizon), desc="Augmenting Data"):
        profit_change = (featured_data.iloc[pos + horizon]["Close"] - featured_data.iloc[pos]["Close"]) / featured_data.iloc[pos]["Close"]
        context = featured_data.iloc[pos][feature_cols].to_dict()
        if pos in indices_buy and random.random() < 0.20:
            aug_rows.append({ **context, "Close": featured_data.iloc[pos]["Close"], "action_type": "BUY", "action_amount": 1.0,
                               "outcome_profit_change": np.clip(profit_change, -0.15, 0.15), "action_type_code": 1 })
        if pos in indices_short and random.random() < 0.20:
             aug_rows.append({ **context, "Close": featured_data.iloc[pos]["Close"], "action_type": "SHORT", "action_amount": 1.0,
                                "outcome_profit_change": np.clip(-profit_change, -0.15, 0.15), "action_type_code": -1 })
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
        print(f"\n✅ Final training data (Bulletproof Version) successfully saved to '{output_path}'")