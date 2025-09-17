# multi_agent_benchmark.py (v2.0 - Setup Phase)

import os
import pandas as pd
import json
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List

# --- Project Components ---
from components import EcommerceSimulatorV7, SymbolicGuardianV3, CausalEngineV6

# --- AI Agent Imports ---
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool

# --- StateProvider Class ---
# We define this class in the global scope to prevent any reference issues.
class StateProvider:
    """A callable class to hold the current state for an agent's tools."""
    def __init__(self):
        self.state = {}
    def __call__(self) -> Dict[str, Any]:
        return self.state
    def update(self, new_state: Dict[str, Any]):
        self.state = new_state

# --- Helper Functions ---

def get_decision_from_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely extracts the action and predicted value from the agent's JSON output
    and returns them in a consistent dictionary format.
    """
    output = response.get('output', '{}')
    m = re.search(r'\{[\s\S]*\}', output)
    try:
        json_string = m.group(0) if m else "{}"
        json_output = json.loads(json_string)
        
        action = json_output.get('action')
        if not isinstance(action, dict):
            # If action is malformed, create a default one.
            action = {"price_change": 0.0, "ad_spend": 0.0}
        else:
            action.setdefault('price_change', 0.0)
            action.setdefault('ad_spend', 0.0)

        # --- DÜZELTME: Her zaman {'action': ...} formatında bir sözlük döndür ---
        return {
            'action': action,
            'predicted_value': json_output.get('predicted_profit_change', 0.0)
        }
    except (json.JSONDecodeError, AttributeError):
        # Hata durumunda da aynı formatı koru
        return {
            "action": {"price_change": 0.0, "ad_spend": 0.0},
            "predicted_value": 0.0
        }

def create_competitive_agent_executor(openai_api_key: str, state_provider: StateProvider, causal_engine: CausalEngineV6) -> AgentExecutor:
    """Creates a full agent executor with our finalized, robust prompt architecture."""
    guardian = SymbolicGuardianV3()
    
    @tool(description="Checks if a proposed action (price_change, ad_spend) violates any predefined business rules.")
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        return json.dumps(guardian.validate_action(action={"price_change": price_change, "ad_spend": ad_spend}, current_state=state_provider()))

    @tool(description="Estimates the long-term causal impact of an action on a trust-adjusted profit score.")
    def estimate_profit_impact(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        return json.dumps(causal_engine.estimate_causal_effect(action={"price_change": price_change, "ad_spend": ad_spend}, context=state_provider()))

    tools = [check_business_rules, estimate_profit_impact]
    
    HUMAN_PROMPT_TEMPLATE = """
My Agent ID is {agent_id}. Current Week: {week}.
---
My Strategic Doctrine:
{goal}
---
Current Market State (including your rivals' last known positions):
{state_json}
---
Based on all the information above, please proceed with your decision-making process.
"""
# multi_agent_benchmark.py -> create_competitive_agent_executor fonksiyonu içinde

    prompt = ChatPromptTemplate.from_messages([
        ("system", """"You are a world-class AI business strategist competing in a market.
Your reasoning process MUST follow this strict workflow:
1.  **Analysis:** Analyze the market state, your competitors, and your specific objective provided in the user prompt.
2.  **Hypothesis Generation:** Formulate THREE diverse and bold hypotheses for actions. These hypotheses MUST NOT be passive (e.g., avoid suggesting `price_change: 0.0` and `ad_spend: 0.0` unless there is an extremely strong strategic reason).
3.  **Validation:** Use the `check_business_rules` tool on EACH of your three hypotheses to see if they are valid.
4.  **Estimation:** For all VALID hypotheses, use the `estimate_profit_impact` tool to predict their long-term value.
5.  **Final Decision:** Compare the estimated outcomes of your valid hypotheses. Choose the one that best aligns with your strategic doctrine (provided in the user prompt).
6.  **Final Output:** Your final answer MUST be a single JSON code block. Do not add any text or explanation before or after the JSON block. Your entire output should be only the JSON.

A perfect example of a final output is:
```json
{{
  "action": {{
    "price_change": -0.10,
    "ad_spend": 1500.0
  }},
  "predicted_profit_change": 25000.0,
  "rationale": "My analysis of the competitor positions and my aggressive growth doctrine indicates that a tactical 10% price drop, supported by a significant ad campaign, will capture market share and yield the highest long-term profit, even if short-term margins are slightly reduced. The causal model predicts a strong positive impact."
}}
"""),
        ("human", HUMAN_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=openai_api_key)
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


# multi_agent_benchmark.py (v3.0 - Final Version)

def run_multi_agent_competition():
    """
    Sets up and runs the entire multi-agent competitive simulation,
    then analyzes and plots all results.
    """
    # --- Simulation Parameters ---
    NUM_AGENTS = 3
    NUM_WEEKS = 52
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable."); return

    print(f"--- Setting up a {NUM_AGENTS}-agent, {NUM_WEEKS}-week competitive simulation ---")
    
    if os.path.exists("initial_causal_data.pkl"): os.remove("initial_causal_data.pkl")
    
    simulator = EcommerceSimulatorV7(num_agents=NUM_AGENTS, seed=42)
    causal_engine = CausalEngineV6(force_regenerate=True, num_simulations=2000)
    
    agent_names = ["Agent 0: The Aggressive Growth Hacker", "Agent 1: The Premium Brand Custodian", "Agent 2: The Cautious Market Analyst"]
    agent_goals = [
        """Your strategic doctrine is **Aggressive Growth Hacking**. Your single most important metric is **Total Cumulative Profit**. Be ruthless, data-driven, and opportunistic. Brand Trust is a resource to be spent for significant short-term profit.""",
        """Your strategic doctrine is **Premium Brand Custodianship**. Your primary asset is **Brand Trust**. Sustainable, long-term value is more important than short-term profit. Avoid price wars; your primary tool for growth is advertising.""",
        """Your strategic doctrine is **Cautious Market Analysis**. Your main goal is **Stability and Predictability**. Avoid volatility. Your primary directive is to keep the product price within the stable range of **$90 to $120**."""
    ]

    agents = []
    for i in range(NUM_AGENTS):# + DOĞRU KOD -----------------------------------------------------
        # Globalde tanımlı olan StateProvider sınıfını kullanıyoruz.
        # Her ajan için SADECE BİR TANE state_provider nesnesi oluşturuyoruz.
        state_provider = StateProvider()
        
        # Bu TEK nesneyi hem ajan oluşturucuya...
        agent_executor = create_competitive_agent_executor(openai_api_key, state_provider, causal_engine)
        
        # ...hem de ajan listesine ekliyoruz. Artık aynı nesneye referans veriyorlar.
        agents.append((agent_executor, state_provider))

    full_history = []
    with tqdm(total=NUM_WEEKS, desc="Simulating Competitive Weeks") as pbar:
        for week in range(NUM_WEEKS):
            actions_for_turn = []
            current_states = simulator.agent_states
            
            for i in range(NUM_AGENTS):
                agent_executor, state_provider = agents[i]
                my_current_state = current_states[i]
                competitor_info = [{"agent_id": s['agent_id'], "price": s['price'], "brand_trust": s['brand_trust']} for s in current_states if s['agent_id'] != i]
                state_with_competitors = {**my_current_state, "competitor_info": competitor_info}
                state_provider.update(state_with_competitors)

                try:
                    response = agent_executor.invoke({
                        "agent_id": i, "week": week + 1,
                        "goal": agent_goals[i],
                        "state_json": json.dumps(state_with_competitors)
                    })
                    actions_for_turn.append(get_decision_from_response(response))
                except Exception as e:
                    tqdm.write(f"  - Agent {i} failed. Error: {e}")
                    # --- DÜZELTME: Hata durumunda da doğru formatı kullanıyoruz ---
                    actions_for_turn.append({"action": {"price_change": 0.0, "ad_spend": 0.0}, "predicted_value": 0.0})
            
            # Bu satır artık doğru formatta veri aldığı için sorunsuz çalışacak
            actual_actions = [d['action'] for d in actions_for_turn]
            new_states = simulator.take_action(actual_actions)
            
            for state in new_states:
                state_copy = state.copy(); state_copy['week'] = week + 1; full_history.append(state_copy)

            tqdm.write(f"--- Turn {week+1}/{NUM_WEEKS} Results ---")
            for i, state in enumerate(new_states):
                action_taken = actual_actions[i]
                price_change_pct = action_taken.get('price_change', 0.0) * 100
                ad_spend_val = action_taken.get('ad_spend', 0.0)
                tqdm.write(f"  - {agent_names[i]:<35}: Action(Price Chg: {price_change_pct:+.1f}%, Ad: ${ad_spend_val:<5,.0f}) -> Outcome(Profit: ${state['profit']:<7,.0f}, Mkt Share: {state['market_share']:.1%}, Trust: {state['brand_trust']:.3f})")
            
            pbar.update(1)

    # --- 6. Final Analysis & Visualization ---
    print("\n--- Competition Finished! Analyzing Results... ---")
    df = pd.DataFrame(full_history)
    df.to_csv("multi_agent_results.csv", index=False)
    print("Full simulation results saved to 'multi_agent_results.csv'")
    df['agent'] = df['agent_id'].apply(lambda x: agent_names[x])

    if not os.path.exists('results'): os.makedirs('results')
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Summary Table ---
    summary_data = []
    for name in agent_names:
        agent_df = df[df['agent'] == name]
        total_profit = agent_df['profit'].sum()
        final_trust = agent_df['brand_trust'].iloc[-1]
        summary_data.append({"Agent": name, "Total Cumulative Profit": f"${total_profit:,.2f}", "Final Brand Trust": f"{final_trust:.3f}"})
    summary_df = pd.DataFrame(summary_data).sort_values(by="Total Cumulative Profit", ascending=False)
    print("\n--- COMPETITION SUMMARY (Sorted by Profit) ---")
    print(summary_df.to_string(index=False))

    # --- Main 2x2 Benchmark Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f'{NUM_AGENTS}-Agent Competition Over {NUM_WEEKS} Weeks', fontsize=18)
    metrics = ['profit', 'brand_trust', 'price', 'market_share']
    titles = ['Weekly Profit', 'Brand Trust', 'Price', 'Market Share']
    for i, metric in enumerate(metrics):
        ax = axes.flatten()[i]
        for agent_name in df['agent'].unique():
            agent_df = df[df['agent'] == agent_name]
            ax.plot(agent_df['week'], agent_df[metric], label=agent_name, marker='o', alpha=0.8)
        ax.set_title(titles[i]); ax.set_xlabel('Week'); ax.legend()
    graph_filename = f'results/multi_agent_main_dashboard_{NUM_WEEKS}weeks.png'
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(graph_filename); plt.close(fig)
    print(f"\nMain dashboard graph saved to '{graph_filename}'")
   
    # --- Advanced Analysis Plots ---
    # 1. Cumulative Profit
    plt.figure(figsize=(12, 8)); df['cumulative_profit'] = df.groupby('agent')['profit'].cumsum()
    sns.lineplot(data=df, x='week', y='cumulative_profit', hue='agent', marker='o', linewidth=2.5)
    plt.title('Cumulative Profit Over Time (Who is Winning?)', fontsize=16); plt.ylabel('Total Cumulative Profit ($)'); plt.xlabel('Week'); plt.legend(title='Agent'); plt.grid(True)
    plt.savefig('results/analysis_cumulative_profit.png'); plt.close()
    print("Saved cumulative profit graph.")

    # 2. Strategy Map
    plt.figure(figsize=(12, 8)); sns.scatterplot(data=df, x='price', y='weekly_ad_spend', hue='agent', style='agent', s=150, alpha=0.8)
    plt.title('Price vs. Ad Spend Strategy Map', fontsize=16); plt.xlabel('Price ($)'); plt.ylabel('Weekly Ad Spend ($)'); plt.legend(title='Agent'); plt.grid(True)
    plt.savefig('results/analysis_strategy_map.png'); plt.close()
    print("Saved strategy map graph.")
    
    # 3. Market Share Evolution
    market_share_pivot = df.pivot_table(index='week', columns='agent', values='market_share')
    market_share_pivot.plot(kind='area', stacked=True, figsize=(12, 8), alpha=0.7)
    plt.title('Market Share Evolution Over Time', fontsize=16); plt.ylabel('Market Share'); plt.xlabel('Week'); plt.ylim(0, 1); plt.legend(title='Agent'); plt.grid(True)
    plt.savefig('results/analysis_market_share_area.png'); plt.close()
    print("Saved market share evolution graph.")

    # 4. Strategy Distributions
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False); fig.suptitle('Strategy Distributions (Volatility & Tendency)', fontsize=18)
    sns.boxplot(ax=axes[0], data=df, x='agent', y='price'); axes[0].set_title('Price Distribution')
    sns.boxplot(ax=axes[1], data=df, x='agent', y='weekly_ad_spend'); axes[1].set_title('Ad Spend Distribution')
    sns.boxplot(ax=axes[2], data=df, x='agent', y='profit'); axes[2].set_title('Weekly Profit Distribution')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig('results/analysis_distributions.png'); plt.close(fig)
    print("Saved strategy distribution graphs.")


# This is the standard entry point for a Python script.
if __name__ == "__main__":
    run_multi_agent_competition()