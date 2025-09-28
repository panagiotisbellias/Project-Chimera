# =============================================================================
# tests/run_quant_backtest.py (Standalone Backtest - Final Debug Version)
#
# Description:
#   This is the definitive, standalone backtesting script for the finalized
#   Chimera-Quant agent. It is designed for rigorous, focused testing and
#   includes the most effective prompt engineering and component architecture
#   developed throughout the project. The exception handling around the agent
#   call is intentionally removed to expose any underlying errors for debugging.
# =============================================================================

import os
import sys
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv

# --- 1. Setup Project Environment ---
# Add project root to path to allow imports from 'src' and 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import MarketSimulatorV2, SymbolicGuardianV6, CausalEngineV7_Quant
from src.config import FEATURE_COLS_DEFAULT
from scripts.quant_prepare_training_data import create_features

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 2. Configuration ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY must be set in your environment or a .env file.")

SIMULATION_DAYS = 200
AGENT_MODEL = "gpt-4o"
INITIAL_CAPITAL = 100000.0

# --- 3. Initialize Core Components & Global State ---
print("--- Initializing Core Components ---")
guardian = SymbolicGuardianV6()
causal_engine = CausalEngineV7_Quant(data_path="causal_training_data_balanced.csv")
current_state = {} # This global state will be updated by the main simulation loop for the tools to use

# --- 4. Define Agent Tools ---
@tool
def check_action_validity(action_type: str, amount: float) -> str:
    """
    Checks if a proposed trading action is valid according to fundamental risk management rules.
    Use this BEFORE estimating the effect of an action.
    Action type must be 'BUY', 'SELL', 'SHORT', or 'HOLD'. Amount is a ratio from 0.0 to 1.0.
    """
    action = {'type': action_type.upper(), 'amount': amount}
    return json.dumps(guardian.validate_action(action, current_state))

@tool
def estimate_profit_impact(action_type: str, amount: float) -> str:
    """
    Estimates the causal profit impact of a VALID trading action based on historical data patterns.
    Only use this for actions that have been validated by 'check_action_validity'.
    """
    action = {'type': action_type.upper(), 'amount': amount}
    market_context = {f: current_state.get('market_data', {}).get(f) for f in FEATURE_COLS_DEFAULT
                      if pd.notna(current_state.get('market_data', {}).get(f))}
    if not market_context: return json.dumps({'error': 'Market context is empty.'})
    
    effect = causal_engine.estimate_causal_effect(action, market_context)
    return json.dumps({'predicted_profit_impact': effect})

# --- 5. Define The "Ultimate Chimera" Prompt ---
SYSTEM_PROMPT = """You are "Chimera-Quant", a world-class autonomous trading agent.
Your decision-making process is a strict, non-negotiable workflow.

**MANDATORY WORKFLOW:**

1.  **Analyze State & Goal:** Deeply analyze the current market state to form a market thesis.
2.  **Brainstorm 4 Hypotheses:** Based on your thesis, create FOUR diverse and actionable hypotheses (e.g., BUY 50%, SELL 25%). Do not be passive.
3.  **Mandatory Validation:** You MUST validate EACH of your four hypotheses using the `check_action_validity` tool.
4.  **Causal Estimation:** For VALID hypotheses, use `estimate_profit_impact` to predict their profitability.
5.  **Synthesize & Decide:** Review the valid options and their predicted impacts. Select the single best action.
6.  **Final Output:** Provide your final decision as a single, clean JSON object.

**EXAMPLE THOUGHT PROCESS:**
*Thought:*
The market RSI is low (oversold). My thesis is that a rebound is likely.

*Hypotheses:*
1.  H1: Aggressive BUY with 80%. `{{'type': 'BUY', 'amount': 0.8}}`
2.  H2: Moderate SHORT with 40%. `{{'type': 'SHORT', 'amount': 0.4}}`
3.  H3: Cautious SELL of 10%. `{{'type': 'SELL', 'amount': 0.1}}`
4. H4: Aggresive SHORT with 80%. `{{'type': 'SHORT', 'amount': 0.8}}`

*Validation:*
-   Checking H1: `check_action_validity(action_type='BUY', amount=0.8)`. Result: `{{'is_valid': True, ...}}`. H1 is valid.
-   Checking H2: `check_action_validity(action_type='SHORT', amount=0.4)`. Result: `{{'is_valid': True, ...}}`. H2 is valid.
-   Checking H3: `check_action_validity(action_type='SELL', amount=0.1)`. Result: `{{'is_valid': True, ...}}`. H3 is valid.
-   Checking H4: `check_action_validity(action_type='SHORT', amount=0.8)`. Result: `{{'is_valid': True, ...}}`. H4 is valid.

*Estimation:*
-   Estimating H1: `estimate_profit_impact(action_type='BUY', amount=0.8)`. Result: `{{'predicted_profit_impact': 0.0521}}`.
-   Estimating H2: `estimate_profit_impact(action_type='SHORT', amount=0.4)`. Result: `{{'predicted_profit_impact': -0.0315}}`.
-   Estimating H3: `estimate_profit_impact(action_type='SELL', amount=0.1)`. Result: `{{'predicted_profit_impact': -0.0250}}`.
-   Estimating H4: `estimate_profit_impact(action_type='SHORT', amount=0.8)`. Result: `{{'predicted_profit_impact': -0.0550}}`.

*Decision:*
H1 has the highest predicted profit. I will choose H1.

*Final Output:*
```json
{{
  "commentary": "Based on oversold conditions, an 80% BUY action offers the highest potential return of +5.21% as estimated by the causal engine. The action was validated as safe.",
  "action": {{
    "type": "BUY",
    "amount": 0.8
  }}
}}
"""

# =============================================================================
# Add these two new helper functions to your script
# =============================================================================

def parse_final_json(agent_output: str) -> dict:
    """Extracts the final JSON block from the agent's string output."""
    # This regex is designed to find a JSON block, even with markdown ```json ... ```
    json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', agent_output, re.DOTALL)
    if not json_match:
        # Fallback for cases where markdown is missing
        json_match = re.search(r'(\{[\s\S]*\})$', agent_output)
    
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            print(f"\nWarning: Failed to decode JSON from agent output.")
            return {}
            
    print(f"\nWarning: Could not find any JSON block in agent output.")
    return {}

def print_daily_report(day, sim_state_after_action, agent_response, action_data):
    """Prints a beautifully formatted summary of the agent's decision for the day."""
    
    # Extract data for the report
    market_data = sim_state_after_action.get('market_data', {})
    portfolio_value = sim_state_after_action.get('portfolio_value', 0)
    cash = sim_state_after_action.get('cash', 0)
    shares = sim_state_after_action.get('shares_held', 0)
    commentary = action_data.get('commentary', 'No commentary provided.')
    action = action_data.get('action', {})
    action_type = action.get('type', 'N/A')
    action_amount = action.get('amount', 0.0)

    # --- Terminal Output Formatting ---
    # Using ANSI escape codes for colors. \033[1m is bold, \033[0m resets.
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    print("\n" + "="*80)
    print(f"{BOLD}{HEADER}=============== CHIMERA DAILY REPORT: DAY {day+1}/{SIMULATION_DAYS} ==============={ENDC}")
    print("="*80)
    
    print(f"\n--- {BOLD}1. Market Snapshot{ENDC} ---")
    print(f"  - BTC Price: ${market_data.get('Close', 0):,.2f} | RSI: {market_data.get('Rsi_14', 0):.2f} | MACD Hist: {market_data.get('Macdh_12_26_9', 0):.2f}")

    print(f"\n--- {BOLD}2. Agent's Reasoning & Commentary{ENDC} ---")
    print(f"  > {commentary}")

    print(f"\n--- {BOLD}3. Final Action & Portfolio Status (After Action){ENDC} ---")
    print(f"  - Final Decision: {BOLD}{CYAN}{action_type}{ENDC} (Amount: {action_amount:.2f})")
    print(f"  - Portfolio Value: {BOLD}{GREEN}${portfolio_value:,.2f}{ENDC}")
    print(f"  - Cash: ${cash:,.2f} | Shares Held: {shares:.4f} BTC")
    
    print("="*80 + "\n")

# =============================================================================
# 6. Reporting and Visualization Function
# =============================================================================

def analyze_and_report(history_df, actions_df, market_data_df, initial_capital):
    """
    Generates and saves a professional, multi-panel performance report dashboard.
    """
    print("\n--- Generating Final Performance Report ---")
    
    # --- Data Preparation for Plotting ---
    report_df = history_df.copy()
    report_df['day'] = range(len(report_df))
    report_df['btc_price'] = market_data_df['Close'].values
    report_df['daily_return'] = report_df['portfolio_value'].pct_change().fillna(0)
    
    # --- Calculate Advanced Metrics ---
    final_value = report_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital) - 1
    
    # Buy & Hold return
    buy_hold_return = (report_df['btc_price'].iloc[-1] / report_df['btc_price'].iloc[0]) - 1
    
    # Sharpe Ratio (annualized)
    risk_free_rate = 0.0
    mean_daily_return = report_df['daily_return'].mean()
    std_daily_return = report_df['daily_return'].std()
    sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return * np.sqrt(252) if std_daily_return != 0 else 0
    
    # Max Drawdown
    cumulative_roll_max = report_df['portfolio_value'].cummax()
    drawdown = report_df['portfolio_value'] / cumulative_roll_max - 1.0
    max_drawdown = drawdown.min()
    
    # Trade Statistics
    trades = actions_df[actions_df['type'] != 'HOLD']
    num_trades = len(trades)
    
    # --- Plotting Dashboard ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 28))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    fig.suptitle('Chimera-Quant: Final Backtest Performance Analysis', fontsize=28, weight='bold')

    # Panel 1: Cumulative Returns vs Buy & Hold
    ax1 = fig.add_subplot(gs[0, :])
    report_df['cumulative_strategy'] = (1 + report_df['daily_return']).cumprod()
    report_df['cumulative_bh'] = report_df['btc_price'] / report_df['btc_price'].iloc[0]
    ax1.plot(report_df['day'], report_df['cumulative_strategy'], label='Strategy Cumulative Return', color='darkgreen', lw=2.5)
    ax1.plot(report_df['day'], report_df['cumulative_bh'], label='Buy & Hold Cumulative Return', color='gray', linestyle='--', lw=2)
    ax1.set_title('1. Cumulative Returns: Strategy vs Buy & Hold', fontsize=18, weight='bold')
    ax1.set_ylabel('Cumulative Growth (x)', fontsize=14)
    ax1.set_xlabel('Trading Day', fontsize=14)
    ax1.legend(loc='upper left')

    # Panel 2: Strategy & Action Map on Price
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.plot(report_df['day'], report_df['btc_price'], color='lightgray', linestyle='-', alpha=0.9, lw=2, label='BTC Price')
    action_colors = {"BUY": "#2ca02c", "SELL": "#d62728", "SHORT": "#9467bd"}
    for _, action in trades.iterrows():
        marker = '^' if action['type'] == 'BUY' else 'v' if action['type'] == 'SELL' else 'o'
        color = action_colors[action['type']]
        ax2.scatter(action['day'], report_df.loc[action['day'], 'btc_price'], 
                    color=color, marker=marker, s=action['amount']*500 + 150, 
                    edgecolors='white', linewidth=1.5, zorder=10, label=action['type'])
    ax2.set_title('2. Strategy Map: Agent Decisions on Price Chart', fontsize=18, weight='bold')
    ax2.set_ylabel('BTC Price ($)', fontsize=14)
    ax2.set_xlabel('Trading Day', fontsize=14)
    
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper left', ncol=4)

    # Panel 3: Key Performance Indicators
    ax3 = fig.add_subplot(gs[2, 0])
    metrics_text = (
        f"----- Performance -----\n"
        f"Total Return: {total_return:.2%}\n"
        f"Buy & Hold Return: {buy_hold_return:.2%}\n\n"
        f"----- Risk -----\n"
        f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}\n"
        f"Max Drawdown: {max_drawdown:.2%}\n\n"
        f"----- Activity -----\n"
        f"Total Trades: {num_trades}\n"
    )
    ax3.text(0.5, 0.5, metrics_text, fontsize=18, va='center', ha='center', linespacing=1.8,
             bbox=dict(boxstyle="round,pad=0.5", fc="#D3D3D3", alpha=1))
    ax3.set_title('3. Key Performance Indicators', fontsize=18, weight='bold')
    ax3.axis('off')

    # Panel 4: Daily Returns Distribution
    ax4 = fig.add_subplot(gs[2, 1])
    sns.histplot(data=report_df, x='daily_return', ax=ax4, color='orchid', bins=30, kde=True)
    ax4.set_title('4. Daily Returns Distribution', fontsize=18, weight='bold')
    ax4.set_xlabel('Daily Return (%)', fontsize=14)
    ax4.axvline(0, color='white', linestyle='--', lw=1)

    # Panel 5: Portfolio Growth vs Market
    ax5 = fig.add_subplot(gs[3, :])
    ax5.plot(report_df['day'], report_df['portfolio_value'], color='darkgreen', label='Chimera Portfolio Value', lw=2.5, zorder=5)
    ax5.set_ylabel('Portfolio Value ($)', color='darkgreen', fontsize=14)
    ax5.set_title('5. Portfolio Growth vs. Market', fontsize=18, weight='bold')
    ax5.tick_params(axis='y', labelcolor='darkgreen')
    ax5.axhline(initial_capital, color='white', linestyle='--', lw=1, alpha=0.7)
    
    ax5_twin = ax5.twinx()
    ax5_twin.plot(report_df['day'], report_df['btc_price'], color='gray', linestyle=':', label='BTC Price', lw=1.5, alpha=0.8)
    ax5_twin.set_ylabel('BTC Price ($)', color='gray', fontsize=14)
    ax5_twin.tick_params(axis='y', labelcolor='gray')
    
    # Combine legends
    lines, labels = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5_twin.legend(lines + lines2, labels + labels2, loc='upper left')

    # Save the final report
    os.makedirs("results/quant", exist_ok=True)
    output_path = os.path.join("results/quant", "final_backtest_report.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Professional performance report saved to '{output_path}'")
    plt.show()


# =============================================================================
# 7. Main Execution Block
# =============================================================================
def run_backtest():
    """
    Main function to run the simulation and generate the report.
    """
    market_data = create_features()
    if market_data is None or len(market_data) < SIMULATION_DAYS:
        raise ValueError("Failed to prepare sufficient market data.")
    simulation_data = market_data.tail(SIMULATION_DAYS).reset_index(drop=True)

    print("\n--- Creating Agent Executor ---")
    tools = [check_action_validity, estimate_profit_impact]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Strategic Goal: {goal}\n\nCurrent State:\n{state_json}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = ChatOpenAI(model=AGENT_MODEL, temperature=0.2)
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors="raise")
    
    print(f"\n--- Starting {SIMULATION_DAYS}-Day Backtest ---")
    simulator = MarketSimulatorV2(market_data=simulation_data, initial_capital=INITIAL_CAPITAL)
    history, actions_log = [], []

# --- Main Simulation Loop ---
    for day in tqdm(range(SIMULATION_DAYS), desc="Simulating Trading Days"):
        sim_state_before = simulator.get_state()
        history.append(sim_state_before)
        current_state.update(sim_state_before)

        # We keep the try/except block for long runs to prevent a single error
        # from crashing the whole 200-day simulation.
        try:
            goal = "Maximize long-term portfolio value by following the mandatory workflow."
            response = executor.invoke({
                "goal": goal,
                "state_json": json.dumps(sim_state_before, default=str)
            })
            
            action_data = parse_final_json(response['output'])
            action = action_data.get("action", {})
            final_action_type = action.get('type', 'HOLD')
            final_amount = float(action.get('amount', 0.0))

        except Exception as e:
            print(f"\n---! An error occurred during agent invocation on day {day+1}: {e} !---")
            print("Defaulting to HOLD action for safety.")
            response = {} # No response to show in report
            action_data = {} # No action data to show in report
            final_action_type, final_amount = 'HOLD', 0.0
        
        # Execute the action in the simulation
        simulator.step({'type': final_action_type, 'amount': final_amount})
        
        # Get the state AFTER the action was taken for the report
        sim_state_after = simulator.get_state()
        
        # Call our new beautiful reporting function
        print_daily_report(day, sim_state_after, response, action_data)

        # Log the action for the final graph
        actions_log.append({'day': day, 'type': final_action_type, 'amount': final_amount})

    print("\n--- Backtest Complete ---")
    history_df = pd.DataFrame(history)
    actions_df = pd.DataFrame(actions_log)
    
    # --- CSV Saving ---
    os.makedirs("results", exist_ok=True)
    history_df.to_csv("results/quant/final_backtest_history.csv", index=False)
    actions_df.to_csv("results/quant/final_backtest_actions.csv", index=False)
    print("✅ Raw history and action logs saved to 'results/quant/' directory.")

    # Call the new professional reporting function
    analyze_and_report(history_df, actions_df, simulation_data, INITIAL_CAPITAL)

# This is what runs when you execute the script
if __name__ == '__main__':
    run_backtest()