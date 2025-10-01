# =============================================================================
# live_paper_trader.py (v6 - FINAL PRESENTATION VERSION)
#
# Author: Aytug Akarlar & Gemini
#
# Description:
#   This script represents the final, fully operational version of the
#   Project Chimera live paper trading bot. It orchestrates the entire
#   "Observe -> Think -> Act" cycle in a live environment:
#
#   1.  OBSERVE: Connects to the Alpaca API and fetches the latest portfolio
#       status and market data for the target asset (BTC/USD).
#   2.  THINK: Feeds this live state into the multi-layered Chimera agent
#       (Neuro-Symbolic-Causal), which analyzes the data, validates hypotheses
#       with the Guardian, estimates outcomes with the Causal Engine, and
#       makes a final, rational trading decision.
#   3.  ACT: Executes the agent's decision by submitting a real paper trading
#       order to the Alpaca exchange.
#   4.  REPORT: Generates a beautiful, detailed report of the entire cycle
#       for clear monitoring and analysis.
#
# This script is the culmination of the "ilmek ilmek isleme" philosophy,
# resulting in a robust, intelligent, and autonomous trading agent.
# =============================================================================

# --- Section 1: Standard Library and Third-Party Imports ---

import os
import json
import re  # Regular expressions for parsing the agent's final JSON output
import time  # For adding strategic delays (e.g., waiting for orders to fill)
import sys
from datetime import datetime, timedelta

import numpy as np  # For numerical operations, especially handling NaN values
import pandas as pd  # The core library for data manipulation and analysis
import alpaca_trade_api as tradeapi  # The official Alpaca API client
from dotenv import load_dotenv  # For loading secret keys from the .env file

# --- Section 2: Project-Specific Chimera Component Imports ---

# Add the project's root directory to the Python path.
# This allows us to import our custom modules from the 'src' folder.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import FEATURE_COLS_DEFAULT  # The single source of truth for feature names
from src.components import SymbolicGuardianV6, CausalEngineV7_Quant  # Our core logic modules

# --- Section 3: LangChain Imports for the AI Agent ---

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# =============================================================================
# --- Section 4: Configuration and Initialization ---
# =============================================================================

print("--- Initializing Live Trader with Chimera Agent (v6 - FINAL) ---")

# Load secret keys (like API keys) from the .env file into environment variables.
load_dotenv()

# Read API credentials from the loaded environment variables.
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # The URL for Alpaca's Paper Trading environment.

# Proactive check to ensure all required keys are present before proceeding.
if not API_KEY or not SECRET_KEY or not os.getenv("OPENAI_API_KEY"):
    raise ValueError("FATAL ERROR: Alpaca and OpenAI API keys must be set in your .env file.")

# Create the API client object. This is our gateway to all Alpaca services.
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
print("✅ Successfully connected to Alpaca API.")

# Initialize the core components of our agent's brain.
print("--- Initializing Core Chimera Components ---")
guardian = SymbolicGuardianV6()
causal_engine = CausalEngineV7_Quant(data_path="causal_training_data_balanced.csv")

# A global dictionary to hold the most recent state. The agent's tools
# will read from this variable to get their context.
current_state_for_tools = {}
print("✅ Guardian and Causal Engine are ready.")

# =============================================================================
# --- Section 5: Definition of Agent Tools ---
# =============================================================================

@tool
def check_action_validity(action_type: str, amount: float) -> str:
    """
    Checks if a proposed trading action is valid according to the hard-coded
    risk management rules within the Symbolic Guardian. Use this BEFORE
    estimating the effect of an action. Action type must be one of
    ['BUY', 'SELL', 'SHORT', 'HOLD']. Amount must be a ratio from 0.0 to 1.0.
    """
    action = {'type': action_type.upper(), 'amount': amount}
    return json.dumps(guardian.validate_action(action, current_state_for_tools))

@tool
def estimate_profit_impact(action_type: str, amount: float) -> str:
    """
    Estimates the causal profit impact of a VALIDATED trading action based on
    historical data patterns learned by the Causal Forest model. Only use this
    for actions that have been validated by 'check_action_validity'.
    """
    action = {'type': action_type.upper(), 'amount': amount}
    # Create the market context using only the features the model was trained on.
    market_context = {f: current_state_for_tools.get('market_data', {}).get(f) for f in FEATURE_COLS_DEFAULT
                      if pd.notna(current_state_for_tools.get('market_data', {}).get(f))}
    
    # Proactive check to ensure the context is not empty.
    if not market_context or len(market_context) < len(FEATURE_COLS_DEFAULT):
        return json.dumps({'error': 'Market context is missing required features.'})
    
    # Get the profit prediction from the Causal Engine.
    effect = causal_engine.estimate_causal_effect(action, market_context)
    return json.dumps({'predicted_profit_impact': effect})

# =============================================================================
# --- Section 6: Data Fetching and Feature Engineering ---
# =============================================================================

def create_live_features(symbol: str = 'BTC/USD', timeframe: str = '1Day', history_days: int = 300) -> dict:
    """
    (BULLETPROOF VERSION - NO PANDAS-TA)
    This function performs the critical task of fetching raw market data and
    manually calculating all the features the agent needs, using only the
    robust pandas & numpy libraries.
    """
    # 1. Fetch raw market data from Alpaca, explicitly defining a start date
    #    to ensure a sufficient amount of historical data is retrieved.
    start_date = (datetime.now() - timedelta(days=history_days)).strftime('%Y-%m-%d')
    bars_df = api.get_crypto_bars(symbol, timeframe, start=start_date).df

    # 2. Proactively check if the API returned enough data for our longest indicator (SMA50).
    if bars_df.empty or len(bars_df) < 50:
        raise ValueError(f"CRITICAL ERROR: Insufficient data from Alpaca for {symbol}. "
                         f"Expected at least 50 bars, got {len(bars_df)}.")

    # 3. Standardize column names to lowercase for consistency in calculations.
    bars_df.rename(columns={'Close': 'close'}, inplace=True, errors='ignore')
    
    # 4. Manually calculate each of the 6 required features.
    
    # Feature 1: RSI (Relative Strength Index)
    delta = bars_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    bars_df['Rsi_14'] = 100 - (100 / (1 + rs))

    # Feature 2: Rate of Change
    bars_df['Roc_5'] = bars_df['close'].pct_change(periods=5)

    # Feature 3: MACD Histogram
    ema_fast = bars_df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = bars_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    bars_df['Macdh_12_26_9'] = macd_line - signal_line

    # Features 4 & 5: SMAs and custom ratios
    sma10 = bars_df['close'].rolling(window=10).mean()
    sma50 = bars_df['close'].rolling(window=50).mean()
    bars_df['Price_vs_sma10'] = bars_df['close'] / sma10
    bars_df['Sma10_vs_sma50'] = sma10 / sma50

    # Feature 6: Bollinger Band Position
    sma20 = bars_df['close'].rolling(window=20).mean()
    std20 = bars_df['close'].rolling(window=20).std()
    bbu = sma20 + (std20 * 2)
    bbl = sma20 - (std20 * 2)
    bb_range = bbu - bbl
    bb_range[bb_range == 0] = np.nan # Avoid division-by-zero errors
    bars_df['Bb_position'] = (bars_df['close'] - bbl) / bb_range

    # 5. Final proactive check for NaN (Not a Number) values in the features.
    final_features = bars_df.iloc[-1]
    if final_features[FEATURE_COLS_DEFAULT].isnull().any():
        print("WARNING: NaN value detected in the final feature set for the agent.")
        print(final_features[FEATURE_COLS_DEFAULT])

    # 6. Return the final, complete feature set for the latest bar as a dictionary.
    return final_features.to_dict()

# /live_paper_trader.py içindeki get_live_state fonksiyonunu güncelle

def get_live_state() -> dict:
    """
    (PNL AWARE VERSION)
    This version is updated to include the current position's entry price
    and unrealized Profit/Loss (PNL) information, giving the agent "memory".
    """
    symbol_for_position = 'BTCUSD'
    symbol_for_data = 'BTC/USD'
    
    account = api.get_account()
    
    # --- NEW: Initialize PNL variables with defaults ---
    shares_held = 0.0
    entry_price = 0.0
    unrealized_pnl = 0.0
    unrealized_pnl_pct = 0.0
    
    try:
        position = api.get_position(symbol_for_position)
        # --- NEW: Extract PNL data from the position object ---
        shares_held = float(position.qty)
        entry_price = float(position.avg_entry_price)
        unrealized_pnl = float(position.unrealized_pl)
        # Convert PNL percentage from a value like 0.05 to 5.0 for easier reading
        unrealized_pnl_pct = float(position.unrealized_plpc) * 100 
        
    except tradeapi.rest.APIError:
        # This means we have no open position, so all PNL values remain 0.
        pass

    raw_market_data = create_live_features(symbol_for_data)
    
    # Translate keys to the format our components expect
    formatted_market_data = raw_market_data.copy()
    if 'open' in formatted_market_data: formatted_market_data['Open'] = formatted_market_data.pop('open')
    if 'high' in formatted_market_data: formatted_market_data['High'] = formatted_market_data.pop('high')
    if 'low' in formatted_market_data: formatted_market_data['Low'] = formatted_market_data.pop('low')
    if 'close' in formatted_market_data: formatted_market_data['Close'] = formatted_market_data.pop('close')
    if 'volume' in formatted_market_data: formatted_market_data['Volume'] = formatted_market_data.pop('volume')
    
    # Assemble the final state object, now including PNL data
    state = {
        "market_data": formatted_market_data,
        "portfolio": { # Group portfolio data for clarity
            "portfolio_value": float(account.equity),
            "cash": float(account.cash),
            "shares_held": shares_held,
            "entry_price": entry_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct
        }
    }
    return state

# =============================================================================
# --- Section 7: Trade Execution and Reporting ---
# =============================================================================

def execute_trade(decision: dict):
    """
    Takes the agent's final JSON decision and places the corresponding
    order on Alpaca. Our strategy is to "clear then enter": first, close all
    existing positions, then enter the new one. This simplifies state management.
    """
    action = decision.get('action', {})
    action_type = action.get('type', 'HOLD').upper()
    amount_ratio = float(action.get('amount', 0.0))
    symbol = 'BTCUSD'  # Use the non-slash version for trading
    order_result = None

    # For any action, our default strategy is to start with a clean slate.
    try:
        api.close_all_positions()
        # Wait a few seconds for the closing order to be processed.
        time.sleep(3)
    except Exception:
        pass  # This will fail if there are no positions to close, which is fine.

    # If the decision is BUY or SHORT, place the new order.
    if action_type in ['BUY', 'SHORT'] and amount_ratio > 0:
        account = api.get_account()
        equity = float(account.equity)
        notional_value = equity * amount_ratio # The dollar value of the trade
        side = 'buy' if action_type == 'BUY' else 'sell'
        
        try:
            order_result = api.submit_order(
                symbol=symbol,
                notional=notional_value,
                side=side,
                type='market',
                time_in_force='gtc'  # 'gtc' is required for 24/7 crypto markets
            )
        except Exception as e:
            return e  # Return the error object for reporting
            
    # Return the successful order object, or the error, or None if no trade was made.
    return order_result

def print_cycle_report(decision: dict, agent_steps: list, order_result: object):
    """
    Prints a beautiful, detailed summary of the entire agent cycle, from
    reasoning and scenarios to execution and final portfolio status.
    """
    # ANSI escape codes for adding color to the terminal output
    HEADER = '\033[95m\033[1m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    # --- Helper function to parse and format the agent's thought process ---
    def format_scenario(step):
        action, observation = step
        tool_input = action.tool_input
        action_type = tool_input.get('action_type', 'N/A')
        amount = tool_input.get('amount', 0.0)
        obs_data = json.loads(observation)
        
        if action.tool == 'check_action_validity':
            is_valid = obs_data.get('is_valid', False)
            message = obs_data.get('message', '')
            status = f"{GREEN}✅ Valid{ENDC}" if is_valid else f"{RED}❌ Invalid{ENDC} ({message})"
            return f"   - H: {action_type} {amount:.2%} -> {status}"
        
        if action.tool == 'estimate_profit_impact':
            profit = obs_data.get('predicted_profit_impact', 0.0)
            profit_color = GREEN if profit > 0 else RED
            return f" -> Est. Profit: {profit_color}{profit:+.2%}{ENDC}"
        return ""

    # --- Print Report Sections ---
    print("\n" + "="*80)
    print(f"| {HEADER}CHIMERA CYCLE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{ENDC}".ljust(89) + "|")
    print("="*80)

    # Section 1: Final Decision
    action = decision.get('action', {})
    commentary = decision.get('commentary', 'No commentary provided.')
    action_type = action.get('type', 'HOLD')
    amount = action.get('amount', 0.0)
    action_color = GREEN if action_type == 'BUY' else RED if action_type in ['SELL', 'SHORT'] else YELLOW
    
    print(f"\n{BOLD}[1] AGENT'S FINAL DECISION{ENDC}")
    print("-" * 80)
    print(f"   - Commentary: {commentary}")
    print(f"   - Final Action: {action_color}{BOLD}{action_type} {amount:.2%}{ENDC}")

    # Section 2: Scenario Analysis from agent's intermediate steps
    print(f"\n{BOLD}[2] SCENARIO ANALYSIS{ENDC}")
    print("-" * 80)
    scenarios = {}
    for step in agent_steps:
        hypothesis_key = str(step[0].tool_input)
        if hypothesis_key not in scenarios: scenarios[hypothesis_key] = format_scenario(step)
        else: scenarios[hypothesis_key] += format_scenario(step)
    for line in scenarios.values(): print(line)

    # Section 3: Trade Execution results
    print(f"\n{BOLD}[3] TRADE EXECUTION DETAILS{ENDC}")
    print("-" * 80)
    if isinstance(order_result, Exception):
        print(f"   - Status:         {RED}❌ FAILED{ENDC}")
        print(f"   - Error:          {str(order_result)}")
    elif order_result:
        time.sleep(3) # Wait for order fill data to populate
        order = api.get_order(order_result.id)
        print(f"   - Status:         {GREEN}✅ SUCCESS{ENDC}")
        print(f"   - Order ID:       {order.id}")
        print(f"   - Symbol:         {order.symbol}")
        print(f"   - Side:           {'BUY' if order.side == 'buy' else 'SELL'}")
        print(f"   - Quantity:       {CYAN}{float(order.filled_qty):.4f} BTC{ENDC}")
        print(f"   - Avg Fill Price: {CYAN}${float(order.filled_avg_price):,.2f}{ENDC}")
        print(f"   - Total Value:    {CYAN}${float(order.notional):,.2f}{ENDC}")
    else:
        print(f"   - Status:         {YELLOW}ℹ️ NO TRADE EXECUTED{ENDC}")

    # Section 4: Final portfolio status after the trade
    print(f"\n{BOLD}[4] POST-TRADE PORTFOLIO STATUS{ENDC}")
    print("-" * 80)
    final_account = api.get_account()
    try:
        final_position = api.get_position('BTCUSD')
        pos_qty = f"{float(final_position.qty):.4f} BTC"
    except tradeapi.rest.APIError:
        pos_qty = "None"
    print(f"   - Portfolio Value: {YELLOW}${float(final_account.equity):,.2f}{ENDC}")
    print(f"   - Cash Balance:    {YELLOW}${float(final_account.cash):,.2f}{ENDC}")
    print(f"   - BTC Position:    {YELLOW}{pos_qty}{ENDC}")
    
    print("="*80 + "\n")

# =============================================================================
# --- Section 8: Main Orchestration Block ---
# =============================================================================

def run_agent_once():
    """
    This is the main orchestrator function. It runs one full cycle of the
    agent's operation, from fetching data to final reporting.
    """
    print("\n" + "="*50)
    print("      STARTING NEW AGENT CYCLE")
    print("="*50)
    
    try:
        # Step 1: Observe the environment
        print("\n[Step 1/5] Fetching live market and portfolio state...")
        live_state = get_live_state()
        global current_state_for_tools
        current_state_for_tools.update(live_state)
        print("✅ State fetched successfully.")
        
        # Step 2: Let the agent think
        print("\n[Step 2/5] Chimera agent is thinking...")
        SYSTEM_PROMPT = SYSTEM_PROMPT = """You are "Chimera-Quant", a world-class autonomous trading agent. Your sole objective is to maximize the portfolio's long-term value by making intelligent, data-driven decisions.

        ### 1. How to Read Your State
        You will be given the current state as a JSON object with two main keys: "market_data" and "portfolio".
        
        - **`market_data`**: Contains the latest technical indicators for the market.
        - **`portfolio`**: Contains your current holdings and performance. This is your memory.
          - `shares_held`: The amount of the asset you currently own. Can be 0.
          - `entry_price`: The average price at which you bought your current shares.
          - `unrealized_pnl_pct`: The current profit or loss on your position, as a percentage (e.g., 5.5 for +5.5%).
        
        ### 2. Critical Rules for Using Tools
        - **The `amount` parameter MUST ALWAYS be a decimal ratio between 0.0 and 1.0.** To use 50% of your assets, you MUST use `amount=0.5`. NEVER use absolute dollar values.
        
        ### 3. Core Strategy & PNL Management
        - **If `shares_held` is 0 (You are OUT of the market):** Your goal is to find a good entry. Analyze the `market_data` to form a `BUY` or `SHORT` thesis.
        - **If `shares_held` > 0 (You are IN a position):** Your priority is managing this position.
          - **Take Profit:** If `unrealized_pnl_pct` is high (e.g., > 4.0), a `SELL` hypothesis to lock in profits should be your primary consideration.
          - **Stop Loss:** If `unrealized_pnl_pct` is too low (e.g., < -2.5), a `SELL` hypothesis to cut your losses is mandatory.
        
        ### 4. Mandatory Workflow
        1.  **Analyze State & Goal:** Deeply analyze the current `market_data` AND your `portfolio`'s PNL status.
        2.  **Brainstorm 4 Hypotheses:** Based on your analysis and the Core Strategy rules, create diverse and actionable hypotheses. If you are in a profitable position, one hypothesis MUST be to SELL and take profit.
        3.  **Mandatory Validation:** You MUST validate EACH of your four hypotheses using the `check_action_validity` tool.
        4.  **Causal Estimation:** For VALID hypotheses, use `estimate_profit_impact` to predict their profitability.
        5.  **Synthesize & Decide:** Review all the data and select the single best action that aligns with your Core Strategy.
        6.  **Final Output:** Provide your final decision as a single, clean JSON object.
        
        ### 5. Final Output Format
        Your final output MUST be a single JSON code block. The JSON object must contain two keys:
        1.  `commentary`: A brief, clear explanation of your reasoning.
        2.  `action`: A dictionary containing the `type` and `amount` of your chosen action.
        
        ### EXAMPLE THOUGHT PROCESS (WITH PNL)
        *Thought:*
        I am currently holding 0.5 BTC. My `unrealized_pnl_pct` is 5.5, which is above my 4.0 take-profit threshold. The market's MACD is also showing signs of weakening. My main thesis is to lock in this profit.
        
        *Hypotheses:*
        1.  H1: Take Profit. SELL 100% of my current position. `{{'type': 'SELL', 'amount': 1.0}}`
        2.  H2: Partial Profit. SELL 50% of my position. `{{'type': 'SELL', 'amount': 0.5}}`
        3.  H3: Hold position, hoping for more gains. `{{'type': 'HOLD', 'amount': 0.0}}`
        4.  H4: Add to position, being greedy. BUY 20%. `{{'type': 'BUY', 'amount': 0.2}}`
        
        *Validation & Estimation:*
        - I will validate all hypotheses. H1, H2, H3, H4 are all valid.
        - I will estimate their impact. The `estimate_profit_impact` for SELL actions might be negative if the market is still predicted to go up, but my Core Strategy says taking profit is the priority.
        
        *Decision:*
        The most rational decision according to my rules is H1: Sell 100% and take profit.
        
        *Final Output:*
        ```json
        {{
          "commentary": "My current position has an unrealized profit of +5.5%, which is above the take-profit threshold. To prudently lock in gains, I will sell 100% of my position.",
          "action": {{
            "type": "SELL",
            "amount": 1.0
          }}
        }}
        """
        
        tools = [check_action_validity, estimate_profit_impact]
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "Strategic Goal: Maximize long-term portfolio value.\n\nCurrent State:\n{state_json}"), MessagesPlaceholder(variable_name="agent_scratchpad")])
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        
        # We set verbose=False to keep the console clean for our custom report.
        # return_intermediate_steps=True is CRITICAL to get the agent's thought process for our report.
        agent = create_openai_tools_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False, return_intermediate_steps=True)
        
        response = executor.invoke({"state_json": json.dumps(live_state, default=str)})
        print("✅ Agent has finished its thought process.")

        # Step 3: Parse the agent's final decision
        print("\n[Step 3/5] Parsing final decision...")
        decision = {}
        final_json_str = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response['output'])
        if final_json_str:
            decision = json.loads(final_json_str.group(1))
            print("✅ Decision parsed successfully.")
        else:
            # If parsing fails, default to a safe HOLD action to prevent errors.
            print("❌ ERROR: Could not parse a final JSON decision. Defaulting to HOLD.")
            decision = {"commentary": "Agent output parsing failed. Defaulting to safe HOLD.", "action": {"type": "HOLD", "amount": 0.0}}

        # Step 4: Execute the decision
        print("\n[Step 4/5] Executing the agent's decision...")
        order_result = execute_trade(decision)

        # Step 5: Generate the final report
        print("\n[Step 5/5] Generating final cycle report...")
        print_cycle_report(decision, response.get('intermediate_steps', []), order_result)
    
    except Exception as e:
        # Catch any unexpected errors during the cycle and print them cleanly.
        print("\n" + "!"*80)
        print(f"A CRITICAL ERROR OCCURRED DURING THE AGENT CYCLE: {e}")
        import traceback
        traceback.print_exc()
        print("!"*80)

    print("AGENT CYCLE COMPLETE.")


# This is the entry point when the script is run directly.
if __name__ == '__main__':
    run_agent_once()