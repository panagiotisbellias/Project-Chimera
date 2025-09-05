# ==============================================================================
#                 Neuro-Symbolic AI Agent - V7.2 (FINAL STABLE APP)
# ==============================================================================
# This is the final, stable, fully-featured version of the interactive app.
# It includes the structured output agent, the learning mechanism,
# the agent selector, and all previous bug fixes and enhancements.
# ==============================================================================

# --- PART 1: LIBRARIES & COMPONENT IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool
from pydantic.v1 import BaseModel, Field
from langchain_community.callbacks import StreamlitCallbackHandler

# --- Import core components from the dedicated file ---
from components import EcommerceSimulatorV4, SymbolicGuardianV2, CausalEngineV4, COST_PER_ITEM

# --- PART 2: AGENT SETUP & HELPER FUNCTIONS ---

@st.cache_resource
def setup_agent_and_tools(agent_type: str):
    """Sets up the agent based on the selected type by importing components."""
    print(f"Setting up '{agent_type}' agent...")
    symbolic_guardian = SymbolicGuardianV2()
    # CausalEngine is managed in session_state, but we ensure it's created if this is the first run for the Full Agent.
    if agent_type == "Full Neuro-Symbolic-Causal" and 'causal_engine' not in st.session_state:
        st.session_state.causal_engine = CausalEngineV4()

    class ActionInput(BaseModel):
        price_change: float = Field(0.0, description="The price change as a decimal.")
        ad_spend: float = Field(0.0, description="The additional ad spend.")

    @tool(args_schema=ActionInput)
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Checks if a proposed action violates business rules."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        result = symbolic_guardian.validate_action(action, st.session_state.simulator.state)
        return json.dumps(result)

    @tool(args_schema=ActionInput)
    def estimate_profit_impact(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Estimates the causal impact of an action on profit."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        if 'causal_engine' in st.session_state:
            report = st.session_state.causal_engine.estimate_causal_effect(action, st.session_state.simulator.state)
            return json.dumps(report)
        return json.dumps({"error": "Causal Engine not available for this agent type."})
    
    tools_map = {
        "Full Neuro-Symbolic-Causal": [check_business_rules, estimate_profit_impact],
        "LLM + Symbolic": [check_business_rules],
        "LLM-Only": []
    }
    tools = tools_map.get(agent_type, [])
    
    # --- Define SYSTEM PROMPTs for each agent type ---
    # Simplified agents are instructed to also use the JSON format for consistency
    if agent_type == "LLM-Only":
        SYSTEM_PROMPT = """You are an AI business strategist. Based on the current market state, propose a single action. You have NO tools. Use your intuition. Your final output MUST be a single JSON object with "commentary", "action" (with "price_change" and "ad_spend"), and "predicted_profit_change" (use 0.0 as you cannot predict). Example: {{"commentary": "I think a small price increase is safe.", "action": {{"price_change": 0.02, "ad_spend": 0.0}}, "predicted_profit_change": 0.0}}"""
    elif agent_type == "LLM + Symbolic":
        SYSTEM_PROMPT = """You are an AI business strategist with a rule-checking tool. 1. Brainstorm a hypothesis. 2. You MUST use the `check_business_rules` tool to verify your hypothesis. 3. If invalid, try again. 4. Once valid, present your final decision. Your final output MUST be a single JSON object with "commentary", "action", and "predicted_profit_change" (use 0.0 as you cannot predict). Example: {{"commentary": "This action is valid according to the rules.", "action": {{"price_change": -0.10, "ad_spend": 500.0}}, "predicted_profit_change": 0.0}}"""
    else: # Full Neuro-Symbolic-Causal Agent
        SYSTEM_PROMPT = """
        You are a world-class AI business strategist. Your reasoning process MUST follow these steps:
        Step 1: Brainstorm at least THREE diverse hypotheses.
        Step 2: Evaluate EACH hypothesis using your tools.
        Step 3: Create a markdown table comparing the results.
        Step 4: Analyze the table and select the best hypothesis.
        
        **VERY IMPORTANT**: Your final output MUST be a single JSON object. Do NOT write any text before or after the JSON object.
        The JSON object must have the following structure:
        {{
          "commentary": "A markdown-formatted text containing your full analysis, the comparison table, and your final rationale.",
          "action": {{
            "price_change": <float>,
            "ad_spend": <float>
          }},
          "predicted_profit_change": <float>
        }}
        Fill the `predicted_profit_change` with the value from the winning hypothesis in your table.
        """

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}"), MessagesPlaceholder(variable_name="agent_scratchpad")])
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=st.session_state.openai_api_key)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

def get_decision_from_response(response: dict):
    """Safely extracts and parses the JSON block from the agent's raw output."""
    try:
        output = response.get('output', '{}')
        json_start_index = output.find('{')
        json_end_index = output.rfind('}')
        if json_start_index != -1 and json_end_index != -1:
            json_string = output[json_start_index : json_end_index + 1]
            json_output = json.loads(json_string)
            json_output.setdefault('commentary', 'No commentary provided.')
            json_output.setdefault('action', {'price_change': 0.0, 'ad_spend': 0.0})
            json_output['action'].setdefault('price_change', 0.0)
            json_output['action'].setdefault('ad_spend', 0.0)
            json_output.setdefault('predicted_profit_change', 0.0)
            return json_output
        else:
            raise json.JSONDecodeError("No JSON object found in the output.", output, 0)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON response: {e}. Raw output: {response.get('output')}")
        return {
            "commentary": f"**Analysis Error:** The agent returned an invalid JSON response. Please try again.\n\n**Raw Output:**\n```\n{response.get('output')}\n```",
            "action": {"price_change": 0.0, "ad_spend": 0.0},
            "predicted_profit_change": 0.0
        }

# --- PART 4: STREAMLIT APPLICATION ---
st.set_page_config(page_title="Neuro-Symbolic AI Agent", layout="wide")
st.title("🧠 Neuro-Symbolic E-Commerce Strategy Lab")

# Initialize all stateful components
if 'causal_engine' not in st.session_state:
    st.session_state.causal_engine = CausalEngineV4()
if 'simulator' not in st.session_state:
    st.session_state.simulator = EcommerceSimulatorV4()
    st.session_state.history = [st.session_state.simulator.state.copy()]
    st.session_state.last_decision_info = None

with st.sidebar:
    st.header("Configuration")
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not st.session_state.openai_api_key: st.info("Please enter your OpenAI API key to begin."); st.stop()
    
    st.header("Agent Control")
    agent_choice = st.selectbox("Select Agent Type:", ("Full Neuro-Symbolic-Causal", "LLM + Symbolic", "LLM-Only"))
    
    st.header("Simulation Control")
    if st.button("Reset Simulation"):
        st.session_state.causal_engine = CausalEngineV4()
        st.session_state.simulator = EcommerceSimulatorV4()
        st.session_state.history = [st.session_state.simulator.state.copy()]
        st.session_state.last_decision_info = None
        st.rerun()

agent_executor = setup_agent_and_tools(agent_type=agent_choice)

state = st.session_state.simulator.state
st.subheader("Current Market State")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Week", state['week']); col2.metric("Price ($)", f"{state['price']:.2f}"); col3.metric("Brand Trust", f"{state['brand_trust']:.2f}"); col4.metric("Last Week's Profit ($)", f"{state['profit']:,.2f}")

if len(st.session_state.history) > 1:
    st.subheader("Weekly Performance Trends")
    history_df = pd.DataFrame(st.session_state.history).set_index('week')
    st.write("Profit Trend ($)"); st.line_chart(history_df[['profit']])
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1: st.write("Price Trend ($)"); st.line_chart(history_df[['price']])
    with chart_col2: st.write("Brand Trust Trend"); st.line_chart(history_df[['brand_trust']])

st.subheader("Define Your Strategic Goal")
user_goal = st.text_area("Describe your objective for the agent:", "Propose a balanced strategy that maximize profit and brand trust.", height=100)

if st.button("Develop Strategy 🚀", use_container_width=True):
    if user_goal:
        with st.spinner(f"The '{agent_choice}' agent is thinking..."):
            full_input = (f"Current Market State:\n{st.session_state.simulator.get_state_string()}\n\n" 
                          f"Contextual Information:\nThe cost to produce one unit of our product is ${COST_PER_ITEM:.2f}.\n\n"
                          f"Our objective: {user_goal}")
            with st.expander("See Agent's Thought Process"):
                st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = agent_executor.invoke({"input": full_input}, {"callbacks": [st_callback]})
            
            st.session_state.last_decision_info = get_decision_from_response(response)
            st.rerun()

if st.session_state.last_decision_info:
    st.divider()
    st.subheader("📈 Agent's Final Decision & Rationale")
    st.markdown(st.session_state.last_decision_info.get("commentary", "No commentary provided."))
    
    action_to_take = st.session_state.last_decision_info.get("action", {})
    is_analysis_visible = "actual_profit_change" in st.session_state.last_decision_info
    
    if st.button("Apply Decision & Advance to Next Week", use_container_width=True, type="primary", disabled=is_analysis_visible):
        previous_profit = state['profit']
        st.session_state.simulator.take_action(action_to_take)
        new_profit = st.session_state.simulator.state['profit']
        actual_profit_change = new_profit - previous_profit
        
        st.session_state.last_decision_info["actual_profit_change"] = actual_profit_change
        st.session_state.history.append(st.session_state.simulator.state.copy())

        current_week = st.session_state.simulator.state['week']
        if (current_week - 1) % 10 == 0 and current_week > 1 and agent_choice == "Full Neuro-Symbolic-Causal":
            with st.spinner(f"Week {current_week - 1}: Agent is learning from past experiences... (Retraining Causal Engine)"):
                st.session_state.causal_engine.retrain(st.session_state.history)
        
        st.rerun()

if st.session_state.last_decision_info and "actual_profit_change" in st.session_state.last_decision_info:
    st.divider()
    st.subheader("Results Analysis: Prediction vs. Reality")
    pred = st.session_state.last_decision_info.get("predicted_profit_change", 0.0)
    actual = st.session_state.last_decision_info.get("actual_profit_change", 0.0)
    
    col_pred, col_actual, col_diff = st.columns(3)
    if agent_choice == "Full Neuro-Symbolic-Causal":
        col_pred.metric("Predicted Profit Change", f"${pred:,.2f}")
    else:
        col_pred.metric("Predicted Profit Change", "N/A (No Causal Engine)")
        
    col_actual.metric("Actual Profit Change", f"${actual:,.2f}")
    
    if agent_choice == "Full Neuro-Symbolic-Causal" and pred is not None and pred != 0:
        diff_percent = ((actual - pred) / abs(pred) * 100) 
        col_diff.metric("Difference", f"${actual - pred:,.2f}", f"{diff_percent:.1f}%")
        if abs(diff_percent) > 30:
            st.warning("Warning: Significant difference between predicted and actual results.")
        else:
            st.success("Success: The model's prediction was reasonably close.")
    else:
        col_diff.metric("Difference", "N/A")