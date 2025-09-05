# benchmark_learning.py

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
    price_change_match = re.search(r"price_change=(-?\d+\.?\d*)", decision_text)
    ad_spend_match = re.search(r"ad_spend=(-?\d+\.?\d*)", decision_text)
    price_change = float(price_change_match.group(1)) if price_change_match else 0.0
    ad_spend = float(ad_spend_match.group(1)) if ad_spend_match else 0.0
    return {"price_change": price_change, "ad_spend": ad_spend}

# --- Agent Factory (adapted from benchmark.py) ---
def create_agent_executor(openai_api_key: str):
    """Creates a standard Full Neuro-Symbolic-Causal agent."""
    symbolic_guardian = SymbolicGuardianV2()
    
    class ActionInput(BaseModel):
        price_change: float = Field(0.0, description="The price change as a decimal.")
        ad_spend: float = Field(0.0, description="The additional ad spend.")

    _simulator_state = {}
    _causal_engine_instance = [None] 

    def get_current_state(): return _simulator_state

    # !!! DÜZELTME: Eksik olan docstring'ler eklendi !!!
    @tool(args_schema=ActionInput)
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Checks if a proposed action violates business rules."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        result = symbolic_guardian.validate_action(action, get_current_state())
        return json.dumps(result)

    # !!! DÜZELTME: Eksik olan docstring'ler eklendi !!!
    @tool(args_schema=ActionInput)
    def estimate_profit_impact(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Estimates the causal impact of an action on profit."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        report = _causal_engine_instance[0].estimate_causal_effect(action, get_current_state())
        return json.dumps(report)

    tools = [check_business_rules, estimate_profit_impact]
    
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
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    return agent_executor, _simulator_state, _causal_engine_instance

# MODIFIED: run_simulation now accepts an `enable_learning` flag
def run_simulation(agent_version: str, num_weeks: int, openai_api_key: str, enable_learning: bool = False):
    """Runs a full simulation for a given agent version."""
    print(f"\n--- Running Benchmark for: {agent_version} ---")
    
    simulator = EcommerceSimulatorV4()
    causal_engine = CausalEngineV4()
    agent_executor, agent_state_awareness, causal_engine_awareness = create_agent_executor(openai_api_key)
    
    causal_engine_awareness[0] = causal_engine
    history = [simulator.state.copy()]
    
    for week in tqdm(range(1, num_weeks + 1), desc=f"Simulating {agent_version}"):
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
        
        # LEARNING MECHANISM TRIGGER
        if enable_learning and week % 10 == 0 and week > 1:
            print(f"\nWeek {week}: Learning Agent is updating its Causal Engine...")
            causal_engine.retrain(history)
            print("Retraining complete.")
            
    return pd.DataFrame(history)

def main():
    """The main A/B test script."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        return

    num_weeks_to_simulate = 52
    agent_versions_to_test = ["Learning Agent", "Static Agent"]

    results = {}
    for version in agent_versions_to_test:
        enable_learning_flag = True if version == "Learning Agent" else False
        results[version] = run_simulation(version, num_weeks_to_simulate, openai_api_key, enable_learning=enable_learning_flag)

    # --- Visualize and Save Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Learning vs. Static Agent Performance Over {num_weeks_to_simulate} Weeks', fontsize=16)

    for version, history_df in results.items():
        axes[0].plot(history_df['week'], history_df['profit'], label=version)
        axes[1].plot(history_df['week'], history_df['brand_trust'], label=version)
        axes[2].plot(history_df['week'], history_df['price'], label=version)

    axes[0].set_title('Profit Over Time'); axes[0].set_ylabel('Total Profit ($)'); axes[0].legend()
    axes[1].set_title('Brand Trust Over Time'); axes[1].set_ylabel('Brand Trust'); axes[1].legend()
    axes[2].set_title('Price Over Time'); axes[2].set_ylabel('Price ($)'); axes[2].legend()
    axes[2].set_xlabel('Week')
    
    if not os.path.exists('results'): os.makedirs('results')
    plt.savefig('results/learning_agent_comparison.png')
    print("\nBenchmark graph saved to 'results/learning_agent_comparison.png'")
    
    # --- Print Final Summary Table ---
    print("\n--- Final Results (Week 52) ---")
    summary_data = []
    for version, history_df in results.items():
        final_state = history_df.iloc[-1]
        summary_data.append({
            "Agent Version": version,
            "Final Profit": f"${final_state['profit']:,.2f}",
            "Final Brand Trust": f"{final_state['brand_trust']:.3f}"
        })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()