# tests/benchmark_quant.py

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import re

# Add project root to path and load components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import MarketSimulatorV1, SymbolicGuardianV5, CausalEngineV7_Quant
from agents.quant_agent import create_quant_agent_executor, current_state

# --- Global Benchmark Configuration ---
NUM_DAYS_TO_SIMULATE = 100

def get_market_data_with_features():
    """Prepares the foundational market data for the simulation."""
    ticker = "BTC-USD"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # The fix is changing 'end_date=end_date' to 'end=end_date'
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.ta.rsi(length=14, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.bbands(length=20, append=True)
    df.dropna(inplace=True)
    return df

def parse_agent_response(response: dict) -> dict:
    """Safely extracts the action dictionary from the agent's JSON output."""
    output = response.get('output', '{}')
    m = re.search(r'\{[\s\S]*\}', output)
    try:
        json_string = m.group(0) if m else "{}"
        json_output = json.loads(json_string)
        action = json_output.get('action')
        if isinstance(action, dict):
            return action
    except (json.JSONDecodeError, AttributeError):
        print(f"\n[Warning] Failed to parse JSON, defaulting to HOLD. Raw output: {output}")
    
    # Default to a safe action if parsing fails
    return {'type': 'HOLD', 'amount': 0.0}

def run_benchmark_simulation(agent_executor, market_data, goal: str) -> pd.DataFrame:
    """Runs a full simulation for the agent with a given strategic goal."""
    simulator = MarketSimulatorV1(market_data=market_data, initial_capital=100_000)
    history = [simulator.get_state()]

    for _ in tqdm(range(NUM_DAYS_TO_SIMULATE), desc=f"Simulating Scenario: {goal[:30]}..."):
        sim_state = simulator.get_state()
        
        # Update the global state that the agent's tools will see
        current_state.update(sim_state)
        
        try:
            response = agent_executor.invoke({
                "goal": goal,
                "state_json": str(sim_state)
            })
            action = parse_agent_response(response)
        except Exception as e:
            print(f"\n[Error] Agent invocation failed: {e}. Defaulting to HOLD.")
            action = {'type': 'HOLD', 'amount': 0.0}
        
        # The agent's prompt already forces it to use the Guardian tool.
        # So, the action received here is assumed to be validated by the agent itself.
        new_state = simulator.step(action)
        history.append(new_state)

    return pd.DataFrame(history)

def main():
    """The main benchmark script that runs multiple scenarios."""
    print("--- Starting Chimera-Quant Agent Benchmark ---")
    
    # 1. Prepare data and components
    market_data = get_market_data_with_features()
    agent_executor = create_quant_agent_executor()
    
    # 2. Define benchmark scenarios
    scenarios = {
        "Profit_Maximization": "Your single, overriding objective is to maximize total cumulative profit. Take calculated risks when the causal engine predicts a high probability of success.",
        "Capital_Preservation": "Your primary goal is to protect capital and avoid significant losses. Prioritize low-risk actions. Only enter a trade if both the market trend and causal analysis strongly support it. Prefer HOLD over risky trades.",
        "Momentum_Chasing": "Your strategy is to identify and ride market trends. Use the SMA and RSI indicators to find strong upward momentum. Enter positions early in a trend and exit when momentum fades. Your goal is to capture large gains from trending periods."
    }

    if not os.path.exists('results'):
        os.makedirs('results')

    all_results = {}
    for scenario_name, goal in scenarios.items():
        print(f"\n--- Running Benchmark for Scenario: {scenario_name} ---")
        history_df = run_benchmark_simulation(agent_executor, market_data, goal)
        all_results[scenario_name] = history_df
        
        # Save individual scenario results
        history_df.to_csv(f"results/benchmark_quant_{scenario_name}.csv", index=False)

    # 3. Visualize and summarize results
    print("\n--- Generating Benchmark Summary Report ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(15, 8 * len(scenarios)), sharex=True)
    fig.suptitle('Chimera-Quant Agent Performance Across Scenarios', fontsize=20)

    for i, (scenario_name, df) in enumerate(all_results.items()):
        ax = axes[i]
        ax.plot(df.index, df['portfolio_value'], label='Portfolio Value', color='navy', linewidth=2.5)
        ax.set_title(f'Scenario: {scenario_name}', fontsize=16)
        ax.set_ylabel('Portfolio Value ($)')
        
        # Add the asset price for context
        ax2 = ax.twinx()
        ax2.plot(df.index, df['market_data'].apply(lambda x: x['Close']), label='BTC Price', color='gray', linestyle='--', alpha=0.6)
        ax2.set_ylabel('BTC Price ($)')
        
        # Add legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.xlabel('Simulation Days')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    report_path = "results/benchmark_quant_summary.png"
    plt.savefig(report_path)
    print(f"\n✅ Benchmark summary graph saved to '{report_path}'")
    plt.close(fig)

    # 4. Print final summary table
    summary_data = []
    for scenario_name, df in all_results.items():
        initial_value = df['portfolio_value'].iloc[0]
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        summary_data.append({
            "Scenario": scenario_name,
            "Final Portfolio Value": f"${final_value:,.2f}",
            "Total Return": f"{total_return:.2%}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n--- FINAL PERFORMANCE SUMMARY ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()