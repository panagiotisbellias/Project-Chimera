# tests/test_market_sim.py

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Add the project root to the Python path to allow imports from `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import both the Simulator and our new Guardian
from src.components import MarketSimulatorV1, SymbolicGuardianV5

def run_test():
    """
    Tests the integration of MarketSimulatorV1 and SymbolicGuardianV5.
    This version uses a definitive set of test cases guaranteed to test the boundary conditions.
    """
    print("--- MarketSimulatorV1 & SymbolicGuardianV5 Integration Test (v3 - Definitive) ---")

    # 1. Fetch market data (Same as before)
    ticker = "BTC-USD"; end_date = datetime.now(); start_date = end_date - timedelta(days=2*365)
    print(f"Fetching data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    try:
        market_df = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(market_df.columns, pd.MultiIndex):
            market_df.columns = market_df.columns.get_level_values(0)
        market_df.dropna(inplace=True)
        print(f"✅ Data successfully fetched. Total days: {len(market_df)}.")
    except Exception as e:
        print(f"❌ ERROR: Failed to fetch data. Error: {e}")
        return

    # 2. Initialize both the Simulator and the Guardian
    simulator = MarketSimulatorV1(market_data=market_df, initial_capital=100_000)
    # The Guardian's code is correct, no changes needed there.
    guardian = SymbolicGuardianV5() 
    print("\n--- Guardian Initialized with Config ---")
    for key, value in guardian.cfg.items(): print(f"  {key}: {value}")

    # 3. Define a definitive list of test cases that forces a rule violation
    test_actions = [
        {"desc": "Test 1: A large initial BUY (80% of cash)", "action": {'type': 'BUY', 'amount': 0.80}},
        {"desc": "Test 2: Another large BUY that MUST BE INVALID due to max_position_ratio", "action": {'type': 'BUY', 'amount': 0.80}},
        {"desc": "Test 3: An invalid action with amount > 1.0", "action": {'type': 'BUY', 'amount': 1.5}},
        {"desc": "Test 4: A full SELL to exit the position", "action": {'type': 'SELL', 'amount': 1.0}}
    ]

    # 4. Loop through test cases
    for test in test_actions:
        print(f"\n--- {test['desc']} ---")
        action_to_test = test['action']
        current_state = simulator.get_state()
        validation_report = guardian.validate_action(action_to_test, current_state)
        print(f"Guardian Report: [VALID: {validation_report['is_valid']}] - Message: {validation_report['message']}")
        if validation_report['is_valid']:
            print(">>> Action is VALID. Proceeding with simulation step.")
            simulator.step(action_to_test)
            print("New Portfolio Value:", f"${simulator.get_state()['portfolio_value']:,.2f}")
        else:
            print(">>> Action is INVALID. SKIPPING simulation step as intended.")
            
    print("\n--- Integration Test Complete ---")


if __name__ == "__main__":
    run_test()