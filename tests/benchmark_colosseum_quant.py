# tests/benchmark_colosseum_quant.py

import os
import sys
import pandas as pd
import numpy as np
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
from agents.quant_agent import SYSTEM_PROMPT, create_quant_agent_executor, current_state, check_action_validity, estimate_profit_impact

# LangChain specific imports for creating different agent types
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Global Benchmark Configuration ---
NUM_DAYS_TO_SIMULATE = 100
AGENT_TYPES_TO_COMPETE = ["Chimera (Full)", "LLM + Symbolic", "LLM-Only"]

def get_market_data_with_features():
    """Prepares the foundational market data for the simulation."""
    # Using a fixed date range for reproducible benchmarks
    start_date = "2023-01-01"
    end_date = "2024-01-01" # 1 year of data
    
    df = yf.download("BTC-USD", start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.ta.rsi(length=14, append=True); df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True); df.ta.bbands(length=20, append=True)
    df.dropna(inplace=True)
    return df

def create_competitive_agent_executor(agent_type: str, causal_engine: CausalEngineV7_Quant) -> AgentExecutor:
    """Creates a LangChain agent executor based on the specified type."""
    tools = []
    if agent_type == "Chimera (Full)":
        tools = [check_action_validity, estimate_profit_impact]
    elif agent_type == "LLM + Symbolic":
        tools = [check_action_validity]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Strategic Goal: {goal}\n\nCurrent State:\n{state_json}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

def parse_agent_response(response: dict) -> dict:
    """Safely extracts the action dictionary from the agent's JSON output."""
    output = response.get('output', '{}')
    m = re.search(r'\{[\s\S]*\}', output)
    try:
        json_string = m.group(0) if m else "{}"
        action = json.loads(json_string).get('action', {})
        if isinstance(action, dict): return action
    except (json.JSONDecodeError, AttributeError):
        pass
    return {'type': 'HOLD', 'amount': 0.0}

def run_colosseum():
    print("--- Starting Chimera-Quant Colosseum Benchmark ---")
    
    # 1. Prepare data and components for all agents
    market_data = get_market_data_with_features().iloc[:NUM_DAYS_TO_SIMULATE]
    causal_engine = CausalEngineV7_Quant(data_path="causal_training_data.csv")
    
    agent_executors = {
        agent_type: create_competitive_agent_executor(agent_type, causal_engine)
        for agent_type in AGENT_TYPES_TO_COMPETE
    }
    
    simulators = {
        agent_type: MarketSimulatorV1(market_data=market_data, initial_capital=100_000)
        for agent_type in AGENT_TYPES_TO_COMPETE
    }

    history = {agent_type: [] for agent_type in AGENT_TYPES_TO_COMPETE}
    
    # 2. Run the simulation loop for all agents in parallel
    for day in tqdm(range(len(market_data)), desc="Simulating the Colosseum"):
        for agent_type in AGENT_TYPES_TO_COMPETE:
            simulator = simulators[agent_type]
            agent_executor = agent_executors[agent_type]
            
            sim_state = simulator.get_state()
            current_state.update(sim_state)
            history[agent_type].append(sim_state)

            try:
                response = agent_executor.invoke({
                    "goal": "Maximize profit.",
                    "state_json": str(sim_state)
                })
                action = parse_agent_response(response)
            except Exception as e:
                action = {'type': 'HOLD', 'amount': 0.0}

            simulator.step(action)

    # 3. Analyze and visualize the results
    history_dfs = {
        name: pd.DataFrame(hist) for name, hist in history.items()
    }
    analyze_and_visualize(history_dfs, market_data)

def analyze_and_visualize(results: dict, market_data: pd.DataFrame):
    """Creates a rich visual report from the benchmark results."""
    print("\n--- Generating Colosseum Performance Report ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- Data Preparation ---
    portfolio_values = pd.DataFrame({
        name: df['portfolio_value'] for name, df in results.items()
    })
    
    # --- Plot 1: Main Performance Chart ---
    fig, axes = plt.subplots(4, 1, figsize=(15, 25), sharex=True)
    fig.suptitle('Chimera Quant Colosseum: Agent Performance Analysis', fontsize=20, weight='bold')

    ax1 = axes[0]
    for agent_name in portfolio_values.columns:
        ax1.plot(portfolio_values.index, portfolio_values[agent_name], label=agent_name, linewidth=2)
    ax1.set_title('Portfolio Value Over Time', fontsize=16)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1_twin = ax1.twinx()
    ax1_twin.plot(market_data.index, market_data['Close'], label='BTC Price', color='gray', linestyle='--', alpha=0.5)
    ax1_twin.set_ylabel('BTC Price ($)')
    ax1_twin.legend(loc='upper right')

    # --- Plot 2: Cumulative Alpha vs. Buy & Hold ---
    ax2 = axes[1]
    buy_hold_return = (market_data['Close'] / market_data['Close'].iloc[0]) - 1
    for agent_name in portfolio_values.columns:
        agent_return = (portfolio_values[agent_name] / portfolio_values[agent_name].iloc[0]) - 1
        alpha = (agent_return - buy_hold_return) * 100 # In percentage points
        ax2.plot(alpha.index, alpha, label=f"{agent_name} Alpha", linewidth=2)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Cumulative Alpha (Outperformance vs. Buy & Hold)', fontsize=16)
    ax2.set_ylabel('Alpha (%)')
    ax2.legend()
    
    # --- Plot 3: Strategy Map (Buy/Sell Decisions) ---
    ax3 = axes[2]
    # (Implementation for this plot can be added later for brevity)
    ax3.set_title('Strategy Map (Placeholder)', fontsize=16)

    # --- Plot 4: Risk/Volatility (Daily Returns Distribution) ---
    ax4 = axes[3]
    daily_returns = portfolio_values.pct_change().dropna()
    daily_returns.plot(kind='box', ax=ax4, vert=False)
    ax4.set_title('Distribution of Daily Returns (Volatility)', fontsize=16)
    ax4.set_xlabel('Daily Return (%)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    report_path = "results/benchmark_colosseum_quant_report.png"
    plt.savefig(report_path)
    print(f"\n✅ Full visual report saved to '{report_path}'")
    plt.close(fig)

    # --- Final Summary Table ---
    summary_data = []
    for name, df in results.items():
        returns = df['portfolio_value'].pct_change().dropna()
        initial = df['portfolio_value'].iloc[0]; final = df['portfolio_value'].iloc[-1]
        total_return = (final - initial) / initial
        # Sharpe Ratio (annualized, assuming daily data)
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        summary_data.append({
            "Agent": name,
            "Final Value": f"${final:,.2f}",
            "Total Return": f"{total_return:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}"
        })
    summary_df = pd.DataFrame(summary_data)
    print("\n--- FINAL COLOSSEUM SUMMARY ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_colosseum()