# tests/final_integration_test.py

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import MarketSimulatorV1, SymbolicGuardianV5, CausalEngineV7_Quant

def run_full_system_test():
    """
    Runs an end-to-end integration test of the full Chimera architecture for Quant Trading.
    The agent uses the Causal Engine to think, the Guardian to check, and the Simulator to act.
    """
    print("--- CHIMERA QUANT AGENT: FULL INTEGRATION TEST ---")

    # --- 1. SETUP: Prepare data and initialize all components ---
    print("\n[SETUP] Preparing market data with features...")
    ticker = "BTC-USD"; end_date = datetime.now(); start_date = end_date - timedelta(days=5*365)
    
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.ta.rsi(length=14, append=True); df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True); df.ta.bbands(length=20, append=True)
    df.dropna(inplace=True)
    print("✅ Market data is ready.")
    
    print("\n[SETUP] Initializing components...")
    simulator = MarketSimulatorV1(market_data=df, initial_capital=100_000)
    guardian = SymbolicGuardianV5()
    # This will take a moment as it trains the model upon initialization
    causal_engine = CausalEngineV7_Quant(data_path="causal_training_data.csv")
    print("✅ All components initialized.")

    # --- 2. SIMULATION: Run the agent's decision loop ---
    num_days_to_simulate = 100
    print(f"\n[SIMULATION] Running for {num_days_to_simulate} days...")
    
    for day in tqdm(range(num_days_to_simulate), desc="Simulating Agent Decisions"):
        # a. OBSERVE: Get the current state of the market and portfolio
        current_state = simulator.get_state()
        market_context = {
            feature: current_state['market_data'][feature] 
            for feature in causal_engine.feature_cols
            if feature in current_state['market_data']
        }

        # b. THINK: Use the Causal Engine to evaluate options
        # For this test, we evaluate a simple 50% BUY vs. 50% SELL
        buy_action = {'type': 'BUY', 'amount': 0.5}
        sell_action = {'type': 'SELL', 'amount': 0.5}

        buy_effect = causal_engine.estimate_causal_effect(buy_action, market_context)
        sell_effect = causal_engine.estimate_causal_effect(sell_action, market_context)
        
        # The agent's hypothesis is the action with the higher predicted profit
        best_action = buy_action if buy_effect > sell_effect else sell_action
        
        # c. CHECK: Submit the best action to the Guardian for validation
        validation_report = guardian.validate_action(best_action, current_state)
        
        # d. ACT: If valid, execute the action. If not, hold.
        if validation_report['is_valid']:
            final_action = best_action
        else:
            # If the best idea is illegal, the safe default is to do nothing.
            final_action = {'type': 'HOLD', 'amount': 0.0}

        # Execute the final, validated action in the simulator
        simulator.step(final_action)
    
    print("\n[SIMULATION] Simulation complete.")

    # --- 3. RESULTS: Display the final outcome ---
    print("\n--- FINAL RESULTS ---")
    final_state = simulator.get_state()
    initial_capital = simulator.initial_capital
    final_value = final_state['portfolio_value']
    total_profit = final_value - initial_capital
    total_return_pct = (total_profit / initial_capital) * 100

    print(f"Initial Portfolio Value: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value:   ${final_value:,.2f}")
    print(f"Total Net Profit:        ${total_profit:,.2f}")
    print(f"Total Return:            {total_return_pct:.2f}%")
    print("---------------------")


if __name__ == "__main__":
    run_full_system_test()