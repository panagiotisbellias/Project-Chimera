# benchmark.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import re
from typing import Optional, Dict, Any

# --- Component and LangChain Imports ---
from components import EcommerceSimulatorV5, SymbolicGuardianV3, CausalEngineV6, DEFAULT_TRUST_VALUE_MULTIPLIER
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool


def get_decision_from_response(response: dict):
    """
    Safely extracts the action dictionary from the agent's JSON output.
    This version is robust against malformed 'action' fields.
    """
    output = response.get('output', '{}')
    m = re.search(r'\{[\s\S]*\}', output)
    try:
        json_string = m.group(0) if m else "{}"
        json_output = json.loads(json_string)

        action = json_output.get('action')
        if isinstance(action, dict):
            action.setdefault('price_change', 0.0)
            action.setdefault('ad_spend', 0.0)
            return action
        else:
            tqdm.write(f"\n[Warning] Parsed JSON, but 'action' key is not a valid dictionary. Defaulting to no-op. Value was: {action}")
            return {"price_change": 0.0, "ad_spend": 0.0}

    except (json.JSONDecodeError, AttributeError):
        tqdm.write(f"\n[Warning] Failed to parse JSON, defaulting to no-op action. Raw output: {output}")
        return {"price_change": 0.0, "ad_spend": 0.0}

def get_dynamic_trust_multiplier(goal: str, openai_api_key: str) -> float:
    """
    A small, fast agent to determine the TRUST_VALUE_MULTIPLIER from the user's goal.
    """
    tqdm.write(f"-> Interpreting strategy to set dynamic trust multiplier...")
    
    # UPDATE: New optimized values ​​set
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
        tqdm.write(f"   - Dynamic multiplier set to: {multiplier:,.0f}")
        return multiplier
    except Exception as e:
        tqdm.write(f"   - Could not determine dynamic multiplier, using default. Error: {e}")
        return DEFAULT_TRUST_VALUE_MULTIPLIER

def create_agent_executor(agent_type: str, openai_api_key: str, current_state_provider, causal_engine: Optional[CausalEngineV6]):
    guardian = SymbolicGuardianV3()
    
    @tool
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Checks if a proposed action violates business rules."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        result = guardian.validate_action(action, current_state_provider())
        return json.dumps(result)

    @tool
    def estimate_profit_impact(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Estimates the causal impact of an action on a long-term value score."""
        if causal_engine:
            action = {"price_change": price_change, "ad_spend": ad_spend}
            report = causal_engine.estimate_causal_effect(action, current_state_provider())
            return json.dumps(report)
        return json.dumps({"error": "Causal Engine is not available for this agent."})

    tools_map = { "Full Neuro-Symbolic-Causal": [check_business_rules, estimate_profit_impact], "LLM + Symbolic": [check_business_rules], "LLM-Only": [] }
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
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    return agent_executor


def run_simulation(agent_type: str, num_weeks: int, openai_api_key: str, goal: str, trust_multiplier: Optional[float] = None):
    simulator = EcommerceSimulatorV5(seed=42)
    guardian = SymbolicGuardianV3()
    causal_engine = None
    if agent_type == "Full Neuro-Symbolic-Causal":
        tqdm.write("--> Causal Engine for Full Agent is being built from scratch...")
        causal_engine = CausalEngineV6(force_regenerate=True, trust_multiplier=trust_multiplier)

    class StateProvider:
        def __init__(self): self.state = {}
        def __call__(self): return self.state
        def update(self, new_state): self.state = new_state
            
    state_provider = StateProvider()
    agent_executor = create_agent_executor(agent_type, openai_api_key, state_provider, causal_engine)
    
    history = [simulator.state.copy()]
    experience_history = []
    
    for _ in tqdm(range(num_weeks), desc=f"Simulating {agent_type}"):
        current_state = simulator.state.copy()
        state_provider.update(current_state)
        
        full_input = (f"Current Market State:\n{json.dumps(current_state)}\n\nOur objective: {goal}")
        
        try:
            response = agent_executor.invoke({"input": full_input})
            action = get_decision_from_response(response)
        except Exception as e:
            tqdm.write(f"  - Agent failed on turn, taking no action. Error: {e}")
            action = {"price_change": 0.0, "ad_spend": 0.0}
            
        safe_action, _ = guardian.repair_action(action, current_state)
        
        state_before = current_state 
        simulator.take_action(safe_action)
        state_after = simulator.state.copy()
        history.append(state_after)

        if agent_type == "Full Neuro-Symbolic-Causal" and causal_engine:
            exp = {
                "initial_price": float(state_before["price"]), "initial_brand_trust": float(state_before["brand_trust"]),
                "initial_ad_spend": float(state_before["weekly_ad_spend"]), "price_change": float(safe_action["price_change"]),
                "ad_spend": float(safe_action["ad_spend"]), "profit_change": float(state_after["profit"] - state_before["profit"]),
                "sales_change": float(state_after["sales_volume"] - state_before["sales_volume"]),
                "trust_change": float(state_after["brand_trust"] - state_before["brand_trust"]),
                "season_phase": int(state_before["season_phase"]),
            }
            experience_history.append(exp)

            current_week = state_after['week']
            if (current_week - 1) % 10 == 0 and current_week > 1:
                tqdm.write(f"\n--- [Week {current_week-1}] Retraining Causal Engine with {len(experience_history)} total experiences... ---")
                causal_engine.retrain(experience_history)
                tqdm.write("--- Retraining complete. ---")
            
        tqdm.write(
            f"[{agent_type} - Week {state_after['week']}] Decision -> "
            f"Price: ${state_after['price']:.2f}, "
            f"Ad Spend: ${state_after['weekly_ad_spend']:.2f}"
        )

    return pd.DataFrame(history)


def main():
    """The main benchmark script that runs multiple scenarios."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        return

    num_weeks_to_simulate = 52
    agent_types_to_test = ["Full Neuro-Symbolic-Causal", "LLM + Symbolic", "LLM-Only"]

    # --- UPDATE: Added missing "Brand_Trust_Focus" scenario ---
    scenarios = {
        "1_Brand_Trust_Focus": {
            "goal": ("Our primary objective is to achieve an exceptional brand trust score above 0.95, but in a more capital-efficient way." 
                     "Instead of relying solely on deep, continuous price cuts, your main strategy should be to use high and sustained advertising to build trust."
                     "Use moderate discounts as a supporting tool, not the only tool."
                     "The goal is to reach >0.95 trust while keeping the business as profitable as possible."),
            "filename": "results/benchmark_trust_focus.png"
        },
        "2_Profit_Maximization": {
            "goal": "Your single, overriding objective is to maximize total cumulative profit. Prioritize the final profit number above all else.",
            "filename": "results/benchmark_profit_focus.png"
        },
        "3_Balanced_Strategy": {
            "goal": ("Your goal is a balanced strategy. Prioritize STABLE, STEADY PROFIT GROWTH. "
                     "Avoid extreme actions and volatility. Use ADVERTISING as your primary tool for growth, "
                     "while keeping the PRICE relatively stable, ideally between $90 and $120. "
                     "Large price cuts are discouraged unless necessary to maintain stability. "
                     "Aim for a final brand trust score between 0.70 and 0.80."),
            "filename": "results/benchmark_balanced.png"
        }
    }

    if not os.path.exists('results'): os.makedirs('results')
    
    # Before each full set of benchmarks, deleting the old pkl file ensures the cleanest start.
    if os.path.exists("initial_causal_data.pkl"):
        print("Deleting old causal data file for a fresh start...")
        os.remove("initial_causal_data.pkl")

    for scenario_name, config in scenarios.items():
        print(f"\n\n{'='*80}\nRUNNING SCENARIO: {scenario_name}\n{'='*80}")

        dynamic_multiplier = get_dynamic_trust_multiplier(config["goal"], openai_api_key)
        
        results = {}
        for agent_type in agent_types_to_test:
            results[agent_type] = run_simulation(
                agent_type=agent_type, num_weeks=num_weeks_to_simulate, 
                openai_api_key=openai_api_key, goal=config["goal"],
                trust_multiplier=dynamic_multiplier if agent_type == "Full Neuro-Symbolic-Causal" else None
            )

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f'Scenario: {scenario_name}\nAgent Performance Over {num_weeks_to_simulate} Weeks', fontsize=18)
        ax = axes.flatten()
        for agent_type, history_df in results.items():
            ax[0].plot(history_df['week'], history_df['profit'], label=agent_type, linewidth=2.5)
            ax[1].plot(history_df['week'], history_df['brand_trust'], label=agent_type, linewidth=2.5)
            ax[2].plot(history_df['week'], history_df['price'], label=agent_type, linewidth=2.5)
            ax[3].plot(history_df['week'], history_df['weekly_ad_spend'], label=agent_type, linewidth=2.5)
        ax[0].set_title('Profit Over Time', fontsize=14); ax[0].set_ylabel('Weekly Profit ($)')
        ax[1].set_title('Brand Trust Over Time', fontsize=14); ax[1].set_ylabel('Brand Trust')
        ax[2].set_title('Price Over Time', fontsize=14); ax[2].set_ylabel('Price ($)'); ax[2].set_xlabel('Week')
        ax[3].set_title('Ad Spend Over Time', fontsize=14); ax[3].set_ylabel('Weekly Ad Spend ($)'); ax[3].set_xlabel('Week')
        for i in range(4): ax[i].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(config["filename"])
        print(f"\nBenchmark graph for {scenario_name} saved to '{config['filename']}'")
        plt.close(fig)

        summary_data = []
        for agent_type, history_df in results.items():
            final_state = history_df.iloc[-1]
            total_profit = history_df['profit'].sum()
            avg_weekly_profit = history_df['profit'].mean()
            summary_data.append({
                "Agent Type": agent_type, "Total Profit (Cumulative)": f"${total_profit:,.2f}",
                "Average Weekly Profit": f"${avg_weekly_profit:,.2f}", "Final Brand Trust": f"{final_state['brand_trust']:.3f}",
                "Final Price": f"${final_state['price']:.2f}", "Final Ad Spend": f"${final_state['weekly_ad_spend']:.2f}"
            })
        summary_df = pd.DataFrame(summary_data)
        try:
            summary_df['Sort_Key'] = summary_df['Total Profit (Cumulative)'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
            summary_df = summary_df.sort_values(by='Sort_Key', ascending=False).drop(columns='Sort_Key')
        except Exception as e:
            print(f"Could not sort summary table by profit. Error: {e}")
        print(f"\n--- Final Results for {scenario_name} (Sorted by Total Profit) ---")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()