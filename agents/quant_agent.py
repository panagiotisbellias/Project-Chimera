# agents/quant_agent.py

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add project root to path and load components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import SymbolicGuardianV5, CausalEngineV7_Quant

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 1. Load Environment and Core Components ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file or environment variables. Please set it.")

# Initialize our battle-tested components
guardian = SymbolicGuardianV5()
causal_engine = CausalEngineV7_Quant(data_path="causal_training_data.csv")
# This global state will be updated by the main simulation loop for the tools to use
current_state = {}

# --- 2. Define Tools for the LangChain Agent ---

@tool
def check_action_validity(action_type: str, amount: float) -> str:
    """
    Checks if a proposed trading action is valid according to fundamental risk management rules.
    Use this BEFORE estimating the effect of an action.
    Action type must be 'BUY', 'SELL', or 'HOLD'. Amount is a ratio from 0.0 to 1.0.
    """
    action = {'type': action_type.upper(), 'amount': amount}
    return str(guardian.validate_action(action, current_state))

@tool
def estimate_profit_impact(action_type: str, amount: float) -> str:
    """
    Estimates the long-term causal profit impact of a VALID trading action.
    Only use this for actions that have been validated by 'check_action_validity'.
    Action type must be 'BUY' or 'SELL'. Amount is a ratio from 0.0 to 1.0.
    """
    action = {'type': action_type.upper(), 'amount': amount}
    # For causal engine, context is the market features, not the full state
    market_context = {
        feature: current_state.get('market_data', {}).get(feature)
        for feature in causal_engine.feature_cols
        if current_state.get('market_data', {}).get(feature) is not None
    }
    effect = causal_engine.estimate_causal_effect(action, market_context)
    return f"{{'predicted_profit_impact': {effect:.4f}}}"

SYSTEM_PROMPT = """You are "Chimera-Quant", a world-class autonomous trading agent.
Your decision-making process is a strict, non-negotiable workflow designed for maximum safety and profitability.

**MANDATORY WORKFLOW:**

1.  **Analyze Goal & State:** First, deeply analyze the user's strategic goal and the current market state (RSI, SMA, Bollinger Bands, etc.). Form an initial market thesis.
2.  **Brainstorm 3 Hypotheses:** Based on your thesis, create THREE diverse, distinct, and actionable hypotheses. A hypothesis is a trading action, like BUY 50% or SELL 25%. Do not be passive.
3.  **Mandatory Validation:** You MUST validate EACH of your three hypotheses using the `check_action_validity` tool. In your thought process, state the action and the validation result for each. If a hypothesis is INVALID, discard it and explain why.
4.  **Causal Estimation:** For the hypotheses that were found to be VALID, and ONLY for those, use the `estimate_profit_impact` tool to predict their future profitability.
5.  **Synthesize & Decide:** Review the valid options and their predicted impacts. Select the single best action that aligns with the user's strategic goal.
6.  **Final Output:** Provide your final decision as a single, clean JSON object. Do not add any text before or after the JSON block.

**EXAMPLE THOUGHT PROCESS:**
*Thought:*
The user wants to be aggressive. The market RSI is low (oversold) and the price is near the lower Bollinger Band. My thesis is that a rebound is likely.

*Hypotheses:*
1.  H1: Aggressive BUY with 80% of cash. `{{'type': 'BUY', 'amount': 0.8}}`
2.  H2: Moderate BUY with 40% of cash. `{{'type': 'BUY', 'amount': 0.4}}`
3.  H3: Cautious SELL of 10% of shares, anticipating a further drop before the rebound. `{{'type': 'SELL', 'amount': 0.1}}`

*Validation:*
-   Checking H1: `check_action_validity(action_type='BUY', amount=0.8)`. Result: `{{'is_valid': True, ...}}`. H1 is valid.
-   Checking H2: `check_action_validity(action_type='BUY', amount=0.4)`. Result: `{{'is_valid': True, ...}}`. H2 is valid.
-   Checking H3: `check_action_validity(action_type='SELL', amount=0.1)`. Result: `{{'is_valid': True, ...}}`. H3 is valid.

*Estimation:*
-   Estimating H1: `estimate_profit_impact(action_type='BUY', amount=0.8)`. Result: `{{'predicted_profit_impact': 0.0521}}`.
-   Estimating H2: `estimate_profit_impact(action_type='BUY', amount=0.4)`. Result: `{{'predicted_profit_impact': 0.0315}}`.
-   Estimating H3: `estimate_profit_impact(action_type='SELL', amount=0.1)`. Result: `{{'predicted_profit_impact': -0.0250}}`.

*Decision:*
H1 has the highest predicted profit and aligns with the aggressive goal. I will choose H1.

*Final Output:*
```json
{{
  "commentary": "Based on the oversold market conditions (RSI below 30) and the user's aggressive goal, an 80% BUY action offers the highest potential return of +5.21% as estimated by the causal engine. The action was validated as safe by the Guardian.",
  "action": {{
    "type": "BUY",
    "amount": 0.8
  }}
}}
"""


def create_quant_agent_executor():
    """Builds and returns the complete LangChain agent executor."""
    tools = [check_action_validity, estimate_profit_impact]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Strategic Goal: {goal}\n\nCurrent State:\n{state_json}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor

if __name__ == '__main__':
    # This is a simple test run to see the agent think
    print("--- Creating Quant Agent Executor ---")
    agent_executor = create_quant_agent_executor()
    print("✅ Agent Executor created.")

    # --- FOOLPROOF TEST STATE GENERATION ---
    print("\n--- Generating a foolproof test state from the actual training data ---")
    # Take the first row of the data the Causal Engine was trained on.
    first_row_of_data = causal_engine.training_df.iloc[0]
    
    # Construct the market_data dict from the feature columns
    market_data_for_test = {col: first_row_of_data[col] for col in causal_engine.feature_cols}
    
    # --- ‼️ BUG FIX STARTS HERE ‼️ ---
    # The Guardian needs the 'Close' price, which is not in 'feature_cols'.
    # We must add it to the market_data for the test.
    if 'Close' in first_row_of_data:
        market_data_for_test['Close'] = first_row_of_data['Close']
    # --- ‼️ BUG FIX ENDS HERE ‼️ ---

    # Build the final state object for the test
    state_for_test = {
        "week": 1,
        "market_data": market_data_for_test,
        "cash": 100000, 
        "shares_held": 0.0, 
        "portfolio_value": 100000
    }
    print("✅ Test state generated successfully.")
    
    # Update the global state for the tools to use
    current_state.update(state_for_test)

    print("\n--- Running a Test Invocation ---")
    user_goal = "The market conditions are as provided. My goal is maximum profitability. Analyze the situation and propose the best action."
    
    response = agent_executor.invoke({
        "goal": user_goal,
        "state_json": str(state_for_test)
    })

    print("\n--- Agent Final Response ---")
    print(response['output'])
