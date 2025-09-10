# app.py

import json
import re
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool
from langchain_community.callbacks import StreamlitCallbackHandler
import os

from components import (
    EcommerceSimulatorV5, SymbolicGuardianV3, CausalEngineV5, 
    COST_PER_ITEM, DEFAULT_TRUST_VALUE_MULTIPLIER
)

# --- 1. HELPER FUNCTIONS ---

def get_dynamic_trust_multiplier(goal: str, openai_api_key: str) -> float:
    """A small, fast agent to determine the TRUST_VALUE_MULTIPLIER from the user's goal."""
    st.info(f"Interpreting strategy to set dynamic trust multiplier...")
    
    interpreter_prompt_text = """You are a strategy interpreter. Your job is to read the user's strategic goal and assign a 'Trust Value Multiplier'.
        - If the goal is EXTREME profit maximization, choose a low value like 100000.
        - If the goal is a BALANCED approach, choose a medium value like 200000.
        - If the goal is an EXTREME brand trust focus, choose a high value like 500000.
    Your output must be a single JSON object: {{"trust_multiplier": <number>}}"""

    interpreter_prompt = ChatPromptTemplate.from_messages([
        ("system", interpreter_prompt_text),
        ("human", "{goal}")
    ])
    
    interpreter_llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
    interpreter_chain = interpreter_prompt | interpreter_llm
    
    try:
        response = interpreter_chain.invoke({"goal": goal})
        result = json.loads(response.content)
        multiplier = float(result.get("trust_multiplier", DEFAULT_TRUST_VALUE_MULTIPLIER))
        st.info(f"-> Dynamic multiplier for this goal set to: {multiplier:,.0f}")
        return multiplier
    except Exception as e:
        st.warning(f"Could not determine dynamic multiplier, using default. Error: {e}")
        return DEFAULT_TRUST_VALUE_MULTIPLIER

def get_state_key(state: dict) -> tuple:
    """Converts a continuous state into a discrete key for caching."""
    rounded_price = round(state.get("price", 0))
    rounded_trust = round(state.get("brand_trust", 0), 2)
    rounded_ad_spend = round(state.get("weekly_ad_spend", 0) / 50) * 50
    return (rounded_price, rounded_trust, rounded_ad_spend)

def get_decision_from_response(response: dict):
    """Safely extracts the full decision dictionary from the agent's JSON output."""
    output = response.get('output', '{}')
    m = re.search(r'\{[\s\S]*\}', output)
    try:
        json_string = m.group(0) if m else "{}"
        json_output = json.loads(json_string)
        json_output.setdefault('action', {'price_change': 0.0, 'ad_spend': 0.0})
        json_output.setdefault('predicted_profit_change', 0.0) # For compatibility
        return json_output
    except (json.JSONDecodeError, AttributeError):
        return {
            "commentary": "**Analysis Error:** Agent returned invalid JSON.",
            "action": {"price_change": 0.0, "ad_spend": 0.0},
            "predicted_profit_change": 0.0
        }

def apply_decision_and_prepare_experience(decision: dict):
    sim: EcommerceSimulatorV5 = st.session_state.simulator
    guardian: SymbolicGuardianV3 = st.session_state.guardian
    before = sim.state.copy()
    safe_action, report = guardian.repair_action(decision.get("action", {}), before)
    after = sim.take_action(safe_action)
    exp = {
        "initial_price": float(before["price"]), "initial_brand_trust": float(before["brand_trust"]),
        "initial_ad_spend": float(before["weekly_ad_spend"]), "price_change": float(safe_action["price_change"]),
        "ad_spend": float(safe_action["ad_spend"]), "profit_change": float(after["profit"] - before["profit"]),
        "sales_change": float(after["sales_volume"] - before["sales_volume"]),
        "trust_change": float(after["brand_trust"] - before["brand_trust"]),
    }
    return {"safe_action": safe_action, "guardian_report": report, "before": before, "after": after, "experience": exp}

def generate_feedback_on_last_action(last_decision_info: dict) -> str:
    if not last_decision_info or "safe_action_applied" not in last_decision_info: return ""
    proposed = last_decision_info.get("action", {})
    applied = last_decision_info.get("safe_action_applied", {})
    if proposed != applied:
        return f"\n--- FEEDBACK ON LAST ACTION ---\nYou proposed: {json.dumps(proposed)}\nSystem applied: {json.dumps(applied)}\nPlease consider this adjustment.\n---------------------------\n"
    return "\n(Feedback: Your previous action was applied exactly as proposed.)\n"

# --- 2. AGENT SETUP ---

# @st.cache_resource
def setup_agent_and_tools(agent_type: str):
    """Sets up the agent and its tools based on the selected type."""
    guardian: SymbolicGuardianV3 = st.session_state.guardian
    causal_engine: CausalEngineV5 = st.session_state.causal_engine

    @tool
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Checks if a proposed action violates business rules."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        result = guardian.validate_action(action, st.session_state.simulator.state)
        return json.dumps(result)

    @tool
    def estimate_profit_impact(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Estimates the causal impact of an action on a long-term value score."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        report = causal_engine.estimate_causal_effect(action, st.session_state.simulator.state)
        return json.dumps(report)

    tools_map = {
        "Full Neuro-Symbolic-Causal": [check_business_rules, estimate_profit_impact],
        "LLM + Symbolic": [check_business_rules], "LLM-Only": []
    }
    tools = tools_map.get(agent_type, [])
    
    if agent_type == "LLM-Only":
        SYSTEM_PROMPT = """You are an AI business strategist. Based on the current market state, propose a single action. You have NO tools. Use your intuition. Your final output MUST be a single JSON object with "commentary", "action" (with "price_change" and "ad_spend"), and "predicted_profit_change" (use 0.0 as you cannot predict). Example: {{"commentary": "I think a small price increase is safe.", "action": {{"price_change": 0.02, "ad_spend": 0.0}}, "predicted_profit_change": 0.0}}"""
    elif agent_type == "LLM + Symbolic":
        SYSTEM_PROMPT = """You are an AI business strategist with a rule-checking tool. 1. Brainstorm a hypothesis. 2. You MUST use the `check_business_rules` tool to verify your hypothesis. 3. If invalid, try again. 4. Once valid, present your final decision. Your final output MUST be a single JSON object with "commentary", "action", and "predicted_profit_change" (use 0.0 as you cannot predict). Example: {{"commentary": "This action is valid according to the rules.", "action": {{"price_change": -0.10, "ad_spend": 500.0}}, "predicted_profit_change": 0.0}}"""

    else: # Full Neuro-Symbolic-Causal Agent
        SYSTEM_PROMPT = """
        You are a world-class AI business strategist. Your reasoning process MUST follow this strict, non-negotiable workflow:

        **Step 1: Brainstorm Initial Hypotheses**
        - Formulate THREE diverse initial hypotheses. These are just first drafts.

        **Step 2: Mandatory Validation**
        - You MUST validate EACH of your initial hypotheses using the `check_business_rules` tool.
        - In your thought process, show the result of each check.
        - **If a hypothesis is INVALID, you MUST discard it immediately and create a new, valid replacement.**
        - Your goal in this step is to end up with a list of THREE FULLY VALIDATED hypotheses.

        **Step 3: Causal Impact Estimation (on Valid Hypotheses Only)**
        - Once you have three confirmed-valid hypotheses, and only then, use the `estimate_profit_impact` tool on EACH of them.

        **Step 4: Analysis and Final JSON Output**
        - Create a markdown table comparing your three VALID and EVALUATED hypotheses.
        - Write a final rationale explaining your choice.
        - Your final output MUST be a single JSON object containing your full analysis in the 'commentary' field, and the chosen action. Do not write any text after the JSON object.
        
        Here is an example of the required thought process and final output format:

        *Thought Process Example:*
        "**Step 1: Brainstorm Initial Hypotheses**
        - H1: Increase price by 60% (price_change=0.6).
        - H2: Decrease price by 10% (price_change=-0.1), increase ad spend by $500.
        - H3: Keep price same, increase ad spend by $1000.

        **Step 2: Mandatory Validation**
        - Checking H1: `check_business_rules(price_change=0.6)` -> Result: Invalid, price increase cannot exceed 50%.
        - H1 is invalid. I must replace it. New H1: Increase price by 40% (price_change=0.4).
        - Checking New H1: `check_business_rules(price_change=0.4)` -> Result: Valid.
        - Checking H2: `check_business_rules(price_change=-0.1, ad_spend=500)` -> Result: Valid.
        - Checking H3: `check_business_rules(price_change=0.0, ad_spend=1000)` -> Result: Valid.
        - I now have three valid hypotheses.

        **Step 3: Causal Impact Estimation**
        - Now I will estimate the impact for the valid H1, H2, and H3..."

        *Final JSON Output Example:*
        ```json
        {{
          "commentary": "### Step 1 & 2: Validated Hypotheses\nAfter brainstorming and validating, my three valid strategies are:\n- H1: A 40% price increase.\n- H2: A 10% discount with a $500 ad spend.\n- H3: A $1000 ad spend increase with no price change.\n\n### Step 3: Comparison Table\n| Hypothesis | Price Change | Ad Spend | Predicted Profit ($) |\n|:---|:---:|:---:|:---:|\n| H1 | +0.40 | 0 | 8500.0 |\n| H2 | -0.10 | 500 | 6200.0 |\n| H3 | 0.0 | 1000| 7100.0 |\n\n### Step 4: Final Rationale\nHypothesis 1 provides the highest profit impact by a significant margin. Although aggressive, it remains within the business rules and is the optimal choice for short-term profit maximization.",
          "action": {{
            "price_change": 0.40,
            "ad_spend": 0.0
          }},
          "predicted_profit_change": 8500.0
        }}
        ```
        """


    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}"), MessagesPlaceholder(variable_name="agent_scratchpad")])
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.session_state.openai_api_key)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# --- 3. STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="Project Chimera", layout="wide")
st.title("🧠 Project Chimera: The Adaptive Strategy Lab")

# Initialize session state
if "app_initialized" not in st.session_state:
    st.session_state.guardian = SymbolicGuardianV3()
    st.session_state.simulator = EcommerceSimulatorV5(seed=123)
    st.session_state.causal_engine = CausalEngineV5(force_regenerate=False)
    st.session_state.history = [st.session_state.simulator.state.copy()]
    st.session_state.experience_history = []
    st.session_state.last_decision_info = None
    st.session_state.decision_cache = {}
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    st.session_state.app_initialized = True

with st.sidebar:
    st.header("Configuration")
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
    if not st.session_state.openai_api_key:
        st.info("Please enter your OpenAI API key to begin.")
        st.stop()

    st.header("Agent Control")
    agent_choice = st.selectbox("Select Agent Type:", ("Full Neuro-Symbolic-Causal", "LLM + Symbolic", "LLM-Only"))

    st.header("Simulation Control")
    if st.button("Reset Simulation"):
        st.session_state.simulator = EcommerceSimulatorV5(seed=123)
        st.session_state.history = [st.session_state.simulator.state.copy()]
        st.session_state.experience_history = []
        st.session_state.last_decision_info = None
        st.session_state.decision_cache = {}
        st.toast("Simulation has been reset!")
        st.rerun()

agent_executor = setup_agent_and_tools(agent_type=agent_choice)

state = st.session_state.simulator.state
st.subheader("Current Market State")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Week", state['week'])
col2.metric("Price ($)", f"{state['price']:.2f}")
col3.metric("Brand Trust", f"{state['brand_trust']:.3f}")
col4.metric("Weekly Ad Spend ($)", f"{state['weekly_ad_spend']:,.2f}")
col5.metric("Last Week's Profit ($)", f"{state['profit']:,.2f}")

if len(st.session_state.history) > 1:
    st.subheader("Weekly Performance Trends")
    history_df = pd.DataFrame(st.session_state.history).set_index('week')
    st.line_chart(history_df[['profit', 'sales_volume']])
    chart_col1, chart_col2 = st.columns(2)
    chart_col1.line_chart(history_df[['price']])
    chart_col2.line_chart(history_df[['brand_trust']])

st.subheader("Define Your Strategic Goal")
user_goal = st.text_area("Describe your objective:", "Our primary objective is to achieve an exceptional brand trust score above 0.95, but in a more capital-efficient way. " 
         "Instead of relying solely on deep, continuous price cuts, your main strategy should be to use high and sustained advertising to build trust. "
         "Use moderate discounts as a supporting tool, not the only tool. "
         "The goal is to reach >0.95 trust while keeping the business as profitable as possible.", height=100)

if st.button("Develop Strategy 🚀", use_container_width=True):
    if user_goal and st.session_state.openai_api_key:
        state_key = get_state_key(state)
        cached_decision = st.session_state.decision_cache.get(state_key)

        if cached_decision:
            st.info(f"💡 Using a cached decision for a similar past state to save resources. (State Key: {state_key})")
            st.session_state.last_decision_info = cached_decision
            st.rerun()
        else:
            dynamic_multiplier = get_dynamic_trust_multiplier(user_goal, st.session_state.openai_api_key)
            st.session_state.causal_engine.trust_multiplier = dynamic_multiplier
            
            with st.spinner(f"The '{agent_choice}' agent is thinking with its new strategic focus..."):
                feedback = generate_feedback_on_last_action(st.session_state.last_decision_info)
                full_input = (f"Current Market State:\n{json.dumps(state)}\n\n{feedback}\nOur objective: {user_goal}")
                with st.expander("See Agent's Thought Process"):
                    st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                    response = agent_executor.invoke({"input": full_input}, {"callbacks": [st_callback]})
                
                st.session_state.last_decision_info = get_decision_from_response(response)
                st.session_state.decision_cache[state_key] = st.session_state.last_decision_info
                st.rerun()

if st.session_state.last_decision_info:
    st.divider()
    st.subheader("📈 Agent's Final Decision & Rationale")
    st.markdown(st.session_state.last_decision_info.get("commentary", "No commentary provided."))
    
    is_decision_applied = "actual_profit_change" in st.session_state.last_decision_info

    if is_decision_applied:
        st.button("✅ Week Advanced! Please Develop a New Strategy Above", use_container_width=True, disabled=True)
    else:
        if st.button("Apply Decision & Advance to Next Week", use_container_width=True, type="primary"):
            result = apply_decision_and_prepare_experience(st.session_state.last_decision_info)
            st.session_state.history.append(result["after"])
            st.session_state.experience_history.append(result["experience"])

            st.session_state.last_decision_info["actual_profit_change"] = result["after"]['profit'] - result["before"]['profit']
            st.session_state.last_decision_info["safe_action_applied"] = result["safe_action"]
            st.session_state.last_decision_info["guardian_validation"] = result["guardian_report"]

            current_week = result["after"]['week']
            if (current_week - 1) % 10 == 0 and current_week > 1 and agent_choice == "Full Neuro-Symbolic-Causal":
                with st.spinner(f"Week {current_week - 1}: Agent is learning..."):
                    st.session_state.causal_engine.retrain(st.session_state.experience_history)
                    st.session_state.decision_cache = {}
                    st.toast("🧠 Agent has learned! Decision cache cleared.")
            st.rerun()

    if is_decision_applied:
        st.divider()
        st.subheader("Results Analysis: Prediction vs. Reality")
        pred = st.session_state.last_decision_info.get("predicted_profit_change", 0.0)
        actual = st.session_state.last_decision_info.get("actual_profit_change", 0.0)

        col_pred, col_actual, col_diff = st.columns(3)
        col_pred.metric("Predicted Value", f"${pred:,.2f}")
        col_actual.metric("Actual Profit Change", f"${actual:,.2f}")

        diff_percent = None
        if pred is not None and abs(pred) > 0.01:
            # Note: We compare 'actual profit change' to 'predicted long term value'
            diff_val = actual - pred
            diff_percent = ((actual - pred) / abs(pred)) * 100
            col_diff.metric("Difference", f"${diff_val:,.2f}", f"{diff_percent:.1f}%")
        else:
            col_diff.metric("Difference", "N/A")
        
        if diff_percent is not None:
            if abs(diff_percent) <= 30:
                st.success(f"✅ Prediction Alignment: Good! The predicted value and actual profit change were reasonably aligned.")
            else:
                st.warning(f"⚠️ Prediction Alignment: Noteable difference between predicted value and actual profit change.")

        st.subheader("Safety & Compliance")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Guardian Validation Report**")
            validation_report = st.session_state.last_decision_info.get("guardian_validation", {})
            is_valid = validation_report.get("is_valid", False)
            message = validation_report.get("message", "No report found.")
            if is_valid:
                st.success(f"✅ Compliant: Agent's action was fully compliant.")
            else:
                st.warning(f"⚠️ Auto-Corrected: Guardian adjusted the action. Reason: \"{message}\"")
        with cols[1]:
            st.markdown("**Safe Action Applied**")
            safe_action = st.session_state.last_decision_info.get("safe_action_applied", {})
            price_change = safe_action.get("price_change", 0.0)
            ad_spend = safe_action.get("ad_spend", 0.0)
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Final Price Change", f"{price_change * 100:+.1f}%")
            metric_col2.metric("Final Ad Spend", f"${ad_spend:,.2f}")