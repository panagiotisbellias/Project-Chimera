# =============================================================================
# live_paper_trader.py (v7 - STRATEGY SYNCHRONIZED)
#
# Description:
#   This version synchronizes the live bot's logic completely with the
#   successful "Slow + ATR" strategy from the final backtest. It now uses the
#   exact same feature engineering and agent prompt, ensuring consistency
#   between testing and live execution.
# =============================================================================

# --- Section 1: Imports ---
import os
import sys
import json
import re
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import logging
from alpaca_trade_api.rest import APIError

# --- Section 2: Project-Specific Imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import SymbolicGuardianV6, CausalEngineV7_Quant
from src.config import FEATURE_COLS_DEFAULT # Now includes Atr_14

# --- Section 3: LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# =============================================================================
# --- Section 4: Configuration & Initialization ---
# =============================================================================

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='trader_activity.log', # Tüm aktiviteleri bu dosyaya yazacak
                    filemode='a')


print("--- Initializing Live Trader (Slow+ATR Strategy) ---")
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

if not API_KEY or not SECRET_KEY or not os.getenv("OPENAI_API_KEY"):
    raise ValueError("FATAL ERROR: Alpaca and OpenAI API keys must be set in your .env file.")

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
print("✅ Successfully connected to Alpaca API.")

print("--- Initializing Core Chimera Components ---")
guardian = SymbolicGuardianV6()
causal_engine = CausalEngineV7_Quant(data_path="causal_training_data_balanced.csv")
current_state_for_tools = {}
print("✅ Guardian and Causal Engine are ready.")


# =============================================================================
# --- Section 5: Definition of Agent Tools ---
# =============================================================================

@tool
def check_action_validity(action_type: str, amount: float) -> str:
    """
    (CORRECTED & SIMPLIFIED)
    Checks if a proposed trading action is valid. It now directly passes the
    global 'current_state_for_tools', which has the correct flat structure
    that the SymbolicGuardian expects.
    """
    action = {'type': action_type.upper(), 'amount': amount}
    
    # The global state now perfectly matches the structure the Guardian needs.
    # No transformations or recombinations are necessary.
    return json.dumps(guardian.validate_action(action, current_state_for_tools))

@tool
def estimate_profit_impact(action_type: str, amount: float) -> str:
    """Estimates the causal profit impact of a VALID trading action."""
    action = {'type': action_type.upper(), 'amount': amount}
    market_context = current_state_for_tools.get('market_data', {})
    if not market_context: return json.dumps({'error': 'Market context is empty.'})
    
    # Ensure all features are present before estimating
    if any(feat not in market_context for feat in FEATURE_COLS_DEFAULT):
        return json.dumps({'error': f'Market context is missing required features. Expected: {FEATURE_COLS_DEFAULT}'})

    effect = causal_engine.estimate_causal_effect(action, market_context)
    return json.dumps({'predicted_profit_impact': effect})

# =============================================================================
# --- Section 6: Data Fetching and Feature Engineering ---
# =============================================================================

def create_live_features(symbol: str = 'BTC/USD', history_days: int = 400) -> dict:
    """
    (BULLETPROOF & SYNCHRONIZED VERSION)
    Fetches historical data from Alpaca and manually calculates all core features,
    including the "Slow Strategy" indicators and ATR.
    """
    start_date = (datetime.now() - timedelta(days=history_days)).strftime('%Y-%m-%d')
    bars_df = api.get_crypto_bars(symbol, '1Day', start=start_date).df
    
    if bars_df.empty or len(bars_df) < 100:
        raise ValueError(f"CRITICAL ERROR: Insufficient data from Alpaca.")
    
    print(f"[Data Prep] Calculating {len(FEATURE_COLS_DEFAULT)} SLOW core features...")
    
    # Proactive Fix: Ensure all necessary columns are lowercase for calculation
    bars_df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True, errors='ignore')

    # --- MANUAL FEATURE CALCULATION ---
    # (Existing calculations remain the same)
    delta = bars_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    bars_df['Rsi_14'] = 100 - (100 / (1 + rs))
    
    ema_fast = bars_df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = bars_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    bars_df['Macdh_12_26_9'] = macd_line - signal_line
    
    bars_df['Roc_15'] = bars_df['close'].pct_change(periods=15)
    
    sma20 = bars_df['close'].rolling(window=20).mean()
    sma100 = bars_df['close'].rolling(window=100).mean()
    bars_df['Price_vs_sma20'] = bars_df['close'] / sma20
    bars_df['Sma20_vs_sma100'] = sma20 / sma100
    
    std20 = bars_df['close'].rolling(window=20).std()
    bbu = sma20 + (std20 * 2)
    bbl = sma20 - (std20 * 2)
    bb_range = bbu - bbl
    bb_range[bb_range == 0] = np.nan
    bars_df['Bb_position'] = (bars_df['close'] - bbl) / bb_range
    
    # --- Manuel ATR Calculation ---
    print("[Data Prep] Calculating ATR for volatility analysis...")
    
    # First, calculate the three components of True Range
    high_low = bars_df['high'] - bars_df['low']
    high_prev_close = np.abs(bars_df['high'] - bars_df['close'].shift(1))
    low_prev_close = np.abs(bars_df['low'] - bars_df['close'].shift(1))
    
    # Combine them into a temporary DataFrame
    tr_df = pd.concat([high_low, high_prev_close, low_prev_close], axis=1)
    
    # True Range is the maximum of the three components for each day
    true_range = tr_df.max(axis=1)
    
    # Calculate the Average True Range (ATR) using a 14-period EMA (Exponential Moving Average)
    bars_df['Atr_14'] = true_range.ewm(alpha=1/14, adjust=False).mean()
    
    # Finalize column names to match components' expectations
    bars_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True, errors='ignore')
    bars_df.dropna(inplace=True)
    
    print("✅ SLOW STRATEGY + ATR data preparation complete.")
    return bars_df

def get_live_state() -> dict:
    """
    (SERVER-READY FINAL VERSION V1.1)
    Fetches live data and correctly handles the 'position does not exist' APIError,
    which is a normal state for an account with no open positions.
    """
    try:
        logging.info("Attempting to fetch account data...")
        account = api.get_account()
        logging.info("Account data fetched successfully.")

        shares_held, entry_price, unrealized_pnl, unrealized_pnl_pct = 0.0, 0.0, 0.0, 0.0
        try:
            position = api.get_position('BTCUSD')
            shares_held = float(position.qty)
            entry_price = float(position.avg_entry_price)
            unrealized_pnl = float(position.unrealized_pl)
            unrealized_pnl_pct = float(position.unrealized_plpc) * 100
        except APIError as e:
            # --- CRITICAL FIX: Correctly check for the "position does not exist" message ---
            error_message = str(e).lower()
            if "position does not exist" in error_message:
                # This is a normal, expected condition if we have no open position.
                logging.info("No active position found for BTCUSD. Setting shares_held to 0.")
                pass # Continue with shares_held = 0.0
            else:
                # Any other API error is unexpected and should be logged as an error.
                logging.error(f"An unexpected APIError occurred while fetching position: {e}")
                raise # Re-raise the exception to stop the script.

        logging.info("Fetching and calculating market features...")
        market_data_df = create_live_features('BTC/USD')
        latest_market_data = market_data_df.iloc[-1].to_dict()
        logging.info("Market features calculated.")

        state = {
            "market_data": latest_market_data,
            "portfolio_value": float(account.equity),
            "cash": float(account.cash),
            "shares_held": shares_held,
            "entry_price": entry_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct
        }
        logging.info(f"State successfully created. Cash: {state['cash']:.2f}, Equity: {state['portfolio_value']:.2f}")
        return state

    except Exception as e:
        logging.critical(f"A critical error occurred in get_live_state: {e}", exc_info=True)
        raise

# Mevcut execute_trade fonksiyonunu silip bunu yapıştır

def execute_trade(decision: dict, live_state: dict):
    """
    (INTELLIGENT EXECUTION V2)
    Executes a trade by calculating the difference (delta) between the current
    position and the target position dictated by the agent. This minimizes
    unnecessary trades and reduces transaction costs.
    """
    # Get current state from the live_state dictionary
    portfolio_value = live_state.get('portfolio_value', 0.0)
    current_qty = live_state.get('shares_held', 0.0)
    current_price = live_state.get('market_data', {}).get('Close', 0.0)
    
    if portfolio_value == 0 or current_price == 0:
        logging.warning("[Execution] Cannot execute trade due to invalid portfolio or price data.")
        return None

    # Get agent's target decision
    action = decision.get('action', {})
    action_type = action.get('type', 'HOLD').upper()
    amount_ratio = float(action.get('amount', 0.0))
    symbol = 'BTCUSD'
    order_result = None

    # --- INTELLIGENT DELTA CALCULATION ---

    # 1. Calculate TARGET quantity in BTC
    target_value = portfolio_value * amount_ratio
    target_qty = target_value / current_price
    
    # For SHORT, the target quantity is negative
    if action_type == 'SHORT':
        target_qty = -target_qty
    # For SELL or HOLD, the target is zero
    elif action_type in ['SELL', 'HOLD']:
        target_qty = 0.0

    # 2. Calculate the DELTA (the difference to trade)
    delta_qty = target_qty - current_qty
    
    logging.info(f"[Execution] Current Qty: {current_qty:.4f} BTC. Target Qty: {target_qty:.4f} BTC. Delta to trade: {delta_qty:.4f} BTC.")

    # 3. Execute trade based on the delta
    if abs(delta_qty * current_price) < 1.0: # Ignore trades smaller than $1
        logging.info("[Execution] Delta is too small to trade. Holding position.")
        return None

    try:
        if delta_qty > 0: # We need to BUY
            side = 'buy'
            qty_to_trade = round(delta_qty, 6) # Round for API precision
            logging.info(f"[Execution] Placing BUY order for {qty_to_trade:.6f} BTC.")
            order_result = api.submit_order(
                symbol=symbol, qty=qty_to_trade, side=side, type='market', time_in_force='gtc'
            )
        elif delta_qty < 0: # We need to SELL
            side = 'sell'
            qty_to_trade = round(abs(delta_qty), 6) # Quantity is always positive
            logging.info(f"[Execution] Placing SELL order for {qty_to_trade:.6f} BTC.")
            order_result = api.submit_order(
                symbol=symbol, qty=qty_to_trade, side=side, type='market', time_in_force='gtc'
            )
    except Exception as e:
        logging.error(f"Failed to submit order to Alpaca: {e}", exc_info=True)
        return e
            
    return order_result

def save_report_for_dashboard(decision: dict, agent_steps: list):
    """Saves the latest agent analysis to a JSON file for the dashboard to read."""
    
    # Process agent steps into a clean list of dictionaries
    scenarios_data = []
    scenarios = {}
    for step in agent_steps:
        tool_input = step[0].tool_input
        action_type = tool_input.get('action_type', 'N/A')
        amount = tool_input.get('amount', 0.0)
        hypothesis_key = f"{action_type} {amount:.2%}"
        
        tool_name = step[0].tool
        obs_data = json.loads(step[1])

        if hypothesis_key not in scenarios:
            scenarios[hypothesis_key] = {'validation': 'Pending', 'message': '', 'impact': None}

        if tool_name == 'check_action_validity':
            scenarios[hypothesis_key]['validation'] = "Valid" if obs_data.get('is_valid') else "Invalid"
            scenarios[hypothesis_key]['message'] = obs_data.get('message', '')
        elif tool_name == 'estimate_profit_impact':
            scenarios[hypothesis_key]['impact'] = obs_data.get('predicted_profit_impact')

    for key, value in scenarios.items():
        scenarios_data.append({'hypothesis': key, **value})

    # Prepare final data object
    report_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "final_action": decision.get('action', {}),
        "commentary": decision.get('commentary', 'No commentary provided.'),
        "scenarios": scenarios_data
    }

    # Save to file
    try:
        with open("last_cycle_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        print("✅ Analysis report saved for dashboard.")
    except Exception as e:
        print(f"❌ Failed to save dashboard report: {e}")


def print_cycle_report(decision: dict, agent_steps: list, order_result: object):
    """
    (ENHANCED VERSION)
    Prints a beautiful, detailed summary of the entire agent cycle, including
    the Causal Engine's profit impact estimates for each valid scenario.
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
    def format_scenario_step(step):
        action, observation = step
        tool_input = action.tool_input
        tool_name = action.tool
        obs_data = json.loads(observation)

        if tool_name == 'check_action_validity':
            is_valid = obs_data.get('is_valid', False)
            message = obs_data.get('message', '')
            status = f"{GREEN}✅ Valid{ENDC}" if is_valid else f"{RED}❌ Invalid{ENDC} ({message})"
            return ('validation', status)

        if tool_name == 'estimate_profit_impact':
            profit = obs_data.get('predicted_profit_impact', 0.0)
            profit_color = GREEN if profit > 0 else RED if profit < 0 else YELLOW
            profit_str = f"-> Est. Causal Impact: {profit_color}{profit:+.2%}{ENDC}"
            return ('estimation', profit_str)
        return ('unknown', '')

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
    
    # Process and combine steps for each hypothesis
    scenarios = {}
    for step in agent_steps:
        tool_input = step[0].tool_input
        action_type = tool_input.get('action_type', 'N/A')
        amount = tool_input.get('amount', 0.0)
        hypothesis_key = f"{action_type} {amount:.2%}"
        
        step_type, step_result = format_scenario_step(step)

        if hypothesis_key not in scenarios:
            scenarios[hypothesis_key] = {'validation': '', 'estimation': ''}
        
        if step_type in scenarios[hypothesis_key]:
            scenarios[hypothesis_key][step_type] = step_result

    # Print the combined results
    for hypothesis, results in scenarios.items():
        validation_str = results.get('validation', '')
        estimation_str = results.get('estimation', '')
        print(f"   - H: {hypothesis.ljust(12)} -> {validation_str.ljust(45)} {estimation_str}")


    # Section 3: Trade Execution results (This part remains the same)
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

    # Section 4: Final portfolio status (This part remains the same)
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
    
    save_report_for_dashboard(decision, agent_steps)

# =============================================================================
# --- Section 8: Main Orchestration Block ---
# =============================================================================

# Mevcut run_agent_once fonksiyonunu silip bunu yapıştır

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
        SYSTEM_PROMPT = SYSTEM_PROMPT = """You are "Chimera-Quant", a world-class autonomous trading agent.
        Your decision-making process is a strict, non-negotiable workflow.
        
        **STRATEGIC RULES:**
        **1. Core Analysis:** Your primary goal is to analyze market trends using RSI, MACD, SMAs, etc.
        **2. Volatility Awareness (ATR):** `Atr_14` measures market volatility. A sudden, massive spike in ATR during a sell-off can signal panic and a potential trend reversal (a "V-shaped" recovery). If your other indicators suggest a SHORT, but ATR is extremely high or spiking, **BE CAUTIOUS**. A smaller position size or waiting for confirmation might be a wiser move than an aggressive SHORT.

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
        logging.info("\n[Step 3/5] Parsing final decision...")
        decision = {}
        final_json_str = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response['output'])
        if final_json_str:
            decision = json.loads(final_json_str.group(1))
            logging.info("✅ Decision parsed successfully.")
        else:
            logging.warning("❌ ERROR: Could not parse a final JSON decision. Defaulting to HOLD.")
            decision = {"commentary": "Agent output parsing failed. Defaulting to safe HOLD.", "action": {"type": "HOLD", "amount": 0.0}}

        # Step 4: Execute the decision
        logging.info("\n[Step 4/5] Executing the agent's decision...")
        # --- CRITICAL CHANGE: Pass live_state to the execution function ---
        order_result = execute_trade(decision, live_state)

        # Step 5: Generate the final report
        logging.info("\n[Step 5/5] Generating final cycle report...")
        print_cycle_report(decision, response.get('intermediate_steps', []), order_result)
    
    except Exception as e:
        logging.critical(f"A CRITICAL ERROR OCCURRED DURING THE AGENT CYCLE: {e}", exc_info=True)

    logging.info("AGENT CYCLE COMPLETE.")


# This is the entry point when the script is run directly.
if __name__ == '__main__':
    run_agent_once()