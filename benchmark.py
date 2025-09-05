# benchmark.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import re

# --- Component and LangChain Imports ---
from components import EcommerceSimulatorV4, SymbolicGuardianV2, CausalEngineV4, COST_PER_ITEM
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool
from pydantic.v1 import BaseModel, Field

# --- Helper Function (copied from app.py) ---
def parse_action_from_decision(decision_text: str) -> dict:
    """Parses action parameters from the agent's text output."""
    price_change_match = re.search(r"price_change=(-?\d+\.?\d*)", decision_text)
    ad_spend_match = re.search(r"ad_spend=(-?\d+\.?\d*)", decision_text)
    price_change = float(price_change_match.group(1)) if price_change_match else 0.0
    ad_spend = float(ad_spend_match.group(1)) if ad_spend_match else 0.0
    return {"price_change": price_change, "ad_spend": ad_spend}

# --- Agent Factory (adapted from app.py, without Streamlit dependencies) ---
def create_agent_executor(agent_type: str, openai_api_key: str):
    """Creates an agent executor based on the selected type."""
    
    symbolic_guardian = SymbolicGuardianV2()
    if agent_type == "Full Neuro-Symbolic-Causal":
        causal_engine = CausalEngineV4()

    class ActionInput(BaseModel):
        price_change: float = Field(0.0, description="The price change as a decimal.")
        ad_spend: float = Field(0.0, description="The additional ad spend.")

    _simulator_state = {}
    def get_current_state():
        return _simulator_state

    @tool(args_schema=ActionInput)
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Checks if a proposed action violates business rules."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        result = symbolic_guardian.validate_action(action, get_current_state())
        return json.dumps(result)

    @tool(args_schema=ActionInput)
    def estimate_profit_impact(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Estimates the causal impact of an action on profit."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        if 'causal_engine' in locals():
            report = causal_engine.estimate_causal_effect(action, get_current_state())
            return json.dumps(report)
        return json.dumps({"error": "Causal Engine not available for this agent type."})
    
    # --- Dynamic setup based on agent_type ---
    if agent_type == "LLM-Only":
        tools = []
        SYSTEM_PROMPT = """You are an AI business strategist. Based on the current market state and contextual information provided, propose a single, optimal action to achieve the user's objective. You have NO tools to verify rules or predict outcomes. You must use your general economic intuition to guess the best action. Your final output MUST be a rationale explaining your choice, followed by the specific numeric values for the action in parenthesis, for example: This is my decision because it balances risk and reward, (price_change=0.05) and (ad_spend=100.0)."""
    elif agent_type == "LLM + Symbolic":
        tools = [check_business_rules]
        SYSTEM_PROMPT = """You are an AI business strategist with a tool to check business rules. Your goal is to find a valid and effective strategy. Your process must be: 1. Analyze the current market state and the user's objective. 2. Brainstorm a strategic hypothesis (a proposed action). 3. You MUST use the `check_business_rules` tool to verify that your hypothesis is valid. 4. If your hypothesis is invalid, you MUST formulate a new one and check it again until you find a valid action. 5. Once you have found a valid action, present it as your final decision. Your final output MUST be a rationale explaining your choice, followed by the specific numeric values for the action in parenthesis, for example: After checking the rules, I found this to be a valid action, (price_change=-0.10) and (ad_spend=500.0)."""
    else: # Full Neuro-Symbolic-Causal Agent
        tools = [check_business_rules, estimate_profit_impact]
        # !!! CRITICAL UPDATE HERE: Added the rule to force the agent to be proactive !!!
        SYSTEM_PROMPT = """
        You are a world-class AI business strategist for an e-commerce company. Your goal is to identify the **optimal** strategy to achieve the user's objective.

        **Crucial Dynamics of Our Simulated World:**
        This simulation has unique economic principles you must internalize:
        - Advertising has a **direct, immediate impact on sales volume** this week, in addition to a smaller, long-term impact on brand trust. It's a powerful tool.
        - Aggressive price increases (>10%) negatively impact 'brand_trust', which can hurt long-term profitability.
        - Strategic discounts (>5%) positively impact 'brand_trust', and can sometimes lead to higher long-term profits.
        - Your ultimate goal is long-term, sustainable profit.

        **Key Business Rules for Brainstorming:**
        - The cost per item is 50.0.
        - The maximum additional ad spend per week is 500.0 (`ad_spend`).
        - The maximum discount is 40% (`price_change` > -0.40).
        - The price cannot exceed 150.0.

        Your reasoning process MUST follow these rigorous steps:

        **Step 1: Brainstorm Multiple Hypotheses**
        - Based on the user's goal AND the crucial dynamics of our world, formulate at least THREE diverse and distinct strategic hypotheses.
        - **CRITICAL RULE: A 'do nothing' hypothesis (price_change=0.0 and ad_spend=0.0) is NOT a valid strategic option and must be avoided. You must propose meaningful actions.**
        - **At least one hypothesis MUST combine multiple actions (e.g., a moderate price change AND an ad spend increase).**
        - Label them clearly (e.g., Hypothesis A, B, C).

        **Step 2: Evaluate Each Hypothesis Systematically**
        - For EACH hypothesis, perform the LOGICAL CHECK (`check_business_rules`) and the CAUSAL CHECK (`estimate_profit_impact`).

        **Step 3: Compare and Conclude**
        - Create a summary markdown table comparing the hypotheses.
        - Analyze the table, considering both immediate profit and long-term effects.
        - Select the **single best hypothesis**.

        **Step 4: Final Recommendation**
        - State your final, optimal decision clearly.
        - Justify your choice by explaining WHY it is superior to the other hypotheses.
        - Your final output must contain the specific numeric values for the chosen action, for example: (price_change=0.10) and (ad_spend=200.0).
        """

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}"), MessagesPlaceholder(variable_name="agent_scratchpad")])
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) # verbose=False for cleaner benchmark logs
    
    return agent_executor, _simulator_state

def run_simulation(agent_type, num_weeks, openai_api_key):
    """Runs a full simulation for a given agent type and returns the history."""
    print(f"\n--- Running Benchmark for: {agent_type} ---")
    
    simulator = EcommerceSimulatorV4()
    agent_executor, agent_state_awareness = create_agent_executor(agent_type, openai_api_key)
    
    history = [simulator.state.copy()]
    
    for _ in tqdm(range(num_weeks), desc=f"Simulating {agent_type}"):
        current_state = simulator.state
        agent_state_awareness.update(current_state)
        
        goal = "Propose a balanced strategy to increase our profit this week while protecting brand trust."
        full_input = (f"Current Market State:\n{json.dumps(current_state)}\n\n"
                      f"Contextual Information:\nThe cost to produce one unit is ${COST_PER_ITEM:.2f}.\n\n"
                      f"Our objective: {goal}")
        
        try:
            response = agent_executor.invoke({"input": full_input})
            action = parse_action_from_decision(response['output'])
        except Exception as e:
            print(f"  - Agent failed on turn, taking no action. Error: {e}")
            action = {"price_change": 0.0, "ad_spend": 0.0}
            
        simulator.take_action(action)
        history.append(simulator.state.copy())
        
    return pd.DataFrame(history)

def main():
    """The main benchmark script."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        print("In your terminal, run: export OPENAI_API_KEY='sk-...'")
        return

    num_weeks_to_simulate = 52
    agent_types_to_test = ["Full Neuro-Symbolic-Causal", "LLM + Symbolic", "LLM-Only"]

    results = {}
    for agent_type in agent_types_to_test:
        results[agent_type] = run_simulation(agent_type, num_weeks_to_simulate, openai_api_key)

    # --- Visualize and Save Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Agent Performance Comparison Over {num_weeks_to_simulate} Weeks', fontsize=16)

    for agent_type, history_df in results.items():
        axes[0].plot(history_df['week'], history_df['profit'], label=agent_type)
        axes[1].plot(history_df['week'], history_df['brand_trust'], label=agent_type)
        axes[2].plot(history_df['week'], history_df['price'], label=agent_type)

    axes[0].set_ylabel('Total Profit ($)')
    axes[0].set_title('Profit Over Time')
    axes[0].legend()
    
    axes[1].set_ylabel('Brand Trust')
    axes[1].set_title('Brand Trust Over Time')
    axes[1].legend()

    axes[2].set_ylabel('Price ($)')
    axes[2].set_title('Price Over Time')
    axes[2].legend()
    
    axes[2].set_xlabel('Week')
    
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/benchmark_comparison.png')
    print("\nBenchmark graph saved to 'results/benchmark_comparison.png'")
    
    # --- Print Final Summary Table ---
    print("\n--- Final Results (Week 52) ---")
    summary_data = []
    for agent_type, history_df in results.items():
        final_state = history_df.iloc[-1]
        summary_data.append({
            "Agent Type": agent_type,
            "Final Profit": f"${final_state['profit']:,.2f}",
            "Final Brand Trust": f"{final_state['brand_trust']:.3f}",
            "Final Price": f"${final_state['price']:.2f}"
        })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()