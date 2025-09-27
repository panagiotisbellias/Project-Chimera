# scripts/prepare_training_data.py

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm

def create_features(ticker="BTC-USD"):
    """Fetches historical data and enriches it with technical analysis features."""
    print(f"Fetching data for {ticker}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print("Original data shape:", df.shape)
    print("Calculating technical indicators...")
    
    df.ta.rsi(length=14, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.bbands(length=20, append=True)
    
    df.dropna(inplace=True)
    print("Data with features shape:", df.shape)
    return df

def generate_training_data(featured_data: pd.DataFrame, lookahead_days: int = 10, num_samples_per_day: int = 5):
    """Generates a training dataset by simulating random actions on historical data."""
    print(f"\nGenerating training data with a {lookahead_days}-day profit horizon...")
    
    future_returns = featured_data['Close'].shift(-lookahead_days) / featured_data['Close'] - 1
    featured_data['profit_change'] = future_returns
    featured_data.dropna(inplace=True)

    training_rows = []
    
    # --- ‼️ BUG FIX STARTS HERE ‼️ ---
    # feature_cols will ONLY be used to select features for the model.
    feature_cols = [col for col in featured_data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'profit_change']]
    # --- ‼️ BUG FIX ENDS HERE ‼️ ---

    for i in tqdm(range(len(featured_data)), desc="Simulating Random Trades"):
        for _ in range(num_samples_per_day):
            
            # Context for the model uses only the feature columns
            context = featured_data.iloc[i][feature_cols].to_dict()
            
            action_type = np.random.choice([1, -1])
            action_amount = np.random.uniform(0.1, 1.0)
            outcome = featured_data.iloc[i]['profit_change'] * action_type
            
            row = {
                **context,
                # --- ‼️ BUG FIX STARTS HERE ‼️ ---
                # We add 'Close' price to the final training data row for the Guardian's use.
                'Close': featured_data.iloc[i]['Close'],
                # --- ‼️ BUG FIX ENDS HERE ‼️ ---
                'action_type': action_type,
                'action_amount': action_amount,
                'outcome_profit_change': outcome
            }
            training_rows.append(row)
            
    training_df = pd.DataFrame(training_rows)
    print(f"\nGenerated {len(training_df)} training samples.")
    return training_df


if __name__ == "__main__":
    featured_data = create_features()
    training_df = generate_training_data(featured_data)
    
    output_path = "causal_training_data.csv"
    training_df.to_csv(output_path, index=False)
    print(f"\n✅ Training data successfully saved to '{output_path}'")
    
    print("\n--- Sample of the final training data ---")
    print(training_df.head())