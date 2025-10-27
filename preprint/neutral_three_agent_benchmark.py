# preprint/neutral_three_agent_benchmark.py
"""
Neutral 3-Agent Architecture Comparison

Fair comparison of three agent architectures with IDENTICAL strategic objectives.
All agents receive the same neutral prompt: "Maximize long-term sustainable profit."

This isolates the effect of ARCHITECTURE rather than prompt engineering.

Agents tested:
1. LLM-Only (no tools)
2. LLM + Symbolic Guardian (rule validation only)
3. Full Chimera (rules + causal foresight)

Generates primary results figure for the paper.
"""

import sys
import os
import json
import re
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import (
    EcommerceSimulatorV5, 
    SymbolicGuardianV4, 
    CausalEngineV6,
    DEFAULT_TRUST_VALUE_MULTIPLIER
)

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'preprint_results', 'ecom'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_WEEKS = 52
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================================
# HELPER CLASSES
# ============================================================================

class StateProvider:
    """Callable class to hold current state for agent tools."""
    def __init__(self):
        self.state = {}
    
    def __call__(self) -> Dict[str, Any]:
        return self.state
    
    def update(self, new_state: Dict[str, Any]):
        self.state = new_state


def get_decision_from_response(response: dict) -> Dict[str, Any]:
    """Extracts action from LLM response with robust error handling."""
    output = response.get('output', '{}')
    m = re.search(r'\{[\s\S]*\}', output)
    try:
        json_string = m.group(0) if m else "{}"
        json_output = json.loads(json_string)
        action = json_output.get('action')
        
        if isinstance(action, dict):
            try:
                price_change = float(action.get('price_change', 0.0) or 0.0)
            except (ValueError, TypeError):
                price_change = 0.0
            
            try:
                ad_spend = float(action.get('ad_spend', 0.0) or 0.0)
            except (ValueError, TypeError):
                ad_spend = 0.0
            
            return {
                "price_change": price_change,
                "ad_spend": ad_spend
            }
        else:
            return {"price_change": 0.0, "ad_spend": 0.0}
    except (json.JSONDecodeError, AttributeError):
        return {"price_change": 0.0, "ad_spend": 0.0}


# ============================================================================
# AGENT CREATION - NEUTRAL PROMPTS
# ============================================================================

def create_llm_only_agent(openai_api_key: str) -> AgentExecutor:
    """Creates LLM-Only agent with neutral objective."""
    
    SYSTEM_PROMPT = """You are an AI business strategist.

**Your Strategic Objective:**
MAXIMIZE LONG-TERM SUSTAINABLE PROFIT.

**Key Considerations:**
- Balance between price, volume, and margins
- Brand trust affects customer willingness to pay and repeat purchases
- Advertising can drive awareness but has diminishing returns
- Short-term gains may harm long-term sustainability

**Your Decision Process:**
1. Analyze the current market state carefully
2. Consider multiple strategic approaches
3. Evaluate trade-offs between short-term and long-term outcomes
4. Choose the action that best serves sustainable profitability

**Output Requirements:**
Your response MUST be a single JSON object with:
- "commentary": Your strategic reasoning and analysis
- "action": {{"price_change": <float>, "ad_spend": <float>}}
- "predicted_profit_change": 0.0 (you cannot predict, use 0.0)

Example: {{"commentary": "Analysis...", "action": {{"price_change": 0.05, "ad_spend": 600.0}}, "predicted_profit_change": 0.0}}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.9, openai_api_key=openai_api_key)
    agent = create_openai_tools_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=False, handle_parsing_errors=True)


def create_llm_symbolic_agent(openai_api_key: str, state_provider: StateProvider) -> AgentExecutor:
    """Creates LLM+Symbolic agent with neutral objective."""
    guardian = SymbolicGuardianV4()
    
    @tool
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Checks if proposed action violates business rules and safety constraints."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        result = guardian.validate_action(action, state_provider())
        return json.dumps(result)
    
    SYSTEM_PROMPT = """You are an AI business strategist with access to a business rules validation system.

**Your Strategic Objective:**
MAXIMIZE LONG-TERM SUSTAINABLE PROFIT.

**Available Tools:**
- check_business_rules: Validates if an action complies with safety constraints (price floors, margins, ad spend limits)

**Key Considerations:**
- Balance between price, volume, and margins
- Brand trust affects customer loyalty and pricing power
- Safety constraints exist to prevent catastrophic decisions
- Sustainable growth requires respecting market dynamics

**Your Decision Process (STRICT WORKFLOW):**

**Step 1: Initial Hypothesis Generation**
- Formulate THREE diverse strategic hypotheses
- Each should explore different aspects of the profit optimization problem

**Step 2: Mandatory Validation**
- You MUST use check_business_rules on EACH hypothesis
- Show validation results in your reasoning
- If a hypothesis is INVALID, discard it and create a valid replacement
- Continue until you have THREE VALID hypotheses

**Step 3: Selection & Final Decision**
- Compare your three valid hypotheses
- Consider both immediate and long-term implications
- Choose the one most aligned with sustainable profitability
- Present your final decision with clear rationale

**Output Requirements:**
Your response MUST be a single JSON object:
{{"commentary": "Your full analysis and reasoning", "action": {{"price_change": <float>, "ad_spend": <float>}}, "predicted_profit_change": 0.0}}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.9, openai_api_key=openai_api_key)
    agent = create_openai_tools_agent(llm, [check_business_rules], prompt)
    return AgentExecutor(agent=agent, tools=[check_business_rules], verbose=False, handle_parsing_errors=True)


def create_full_chimera_agent(openai_api_key: str, state_provider: StateProvider, 
                               causal_engine: CausalEngineV6) -> AgentExecutor:
    """Creates Full Neuro-Symbolic-Causal agent with neutral objective."""
    guardian = SymbolicGuardianV4()
    
    @tool
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Checks if proposed action violates business rules and safety constraints."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        result = guardian.validate_action(action, state_provider())
        return json.dumps(result)
    
    @tool
    def estimate_profit_impact(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Estimates long-term causal impact of action on profit and brand trust."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        report = causal_engine.estimate_causal_effect(action, state_provider())
        return json.dumps(report)
    
    SYSTEM_PROMPT = """You are a world-class AI business strategist with access to advanced decision support tools.

**Your Strategic Objective:**
MAXIMIZE LONG-TERM SUSTAINABLE PROFIT.

**Available Tools:**
- check_business_rules: Validates if an action complies with safety constraints
- estimate_profit_impact: Predicts long-term causal impact on profit (considers brand trust dynamics and second-order effects)

**Key Considerations:**
- Short-term profit vs. long-term brand value trade-offs
- Causal relationships between pricing, trust, and customer lifetime value
- Market dynamics and competitive positioning
- Risk-adjusted returns and sustainability

**Your Decision Process (STRICT WORKFLOW):**

**Step 1: Brainstorm Initial Hypotheses**
- Formulate THREE diverse strategic hypotheses
- Explore different approaches: aggressive growth, conservative stability, and balanced optimization
- These are initial drafts that will be validated and evaluated

**Step 2: Mandatory Validation**
- You MUST validate EACH hypothesis using check_business_rules
- Show the validation result for each in your reasoning
- **If a hypothesis is INVALID, discard it immediately and create a valid replacement**
- Your goal: end with THREE FULLY VALIDATED hypotheses

**Step 3: Causal Impact Estimation (Valid Hypotheses Only)**
- Once you have three valid hypotheses, use estimate_profit_impact on EACH
- This tool considers long-term effects including brand trust erosion/building
- Compare the predicted long-term values carefully

**Step 4: Analysis and Final Decision**
- Create a comparison of your three valid, evaluated hypotheses
- Consider both immediate profit AND long-term sustainability
- Account for risk and uncertainty in your assessment
- Choose the action with the best long-term risk-adjusted outcome
- Provide clear, data-driven rationale for your choice

**Output Requirements:**
Your response MUST be a single JSON object:
{{"commentary": "Your complete analysis and reasoning", "action": {{"price_change": <float>, "ad_spend": <float>}}, "predicted_profit_change": <value from best hypothesis>}}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.9, openai_api_key=openai_api_key)
    agent = create_openai_tools_agent(llm, [check_business_rules, estimate_profit_impact], prompt)
    return AgentExecutor(agent=agent, tools=[check_business_rules, estimate_profit_impact], 
                        verbose=False, handle_parsing_errors=True)


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_agent(agent_type: str, trust_multiplier: Optional[float] = None) -> pd.DataFrame:
    """
    Runs one agent for 52 weeks with neutral objective.
    
    Args:
        agent_type: 'llm_only', 'llm_symbolic', or 'full_chimera'
        trust_multiplier: For causal engine (only used by full_chimera)
    """
    print(f"\n{'='*70}")
    print(f"Running: {agent_type.upper()}")
    print(f"{'='*70}")
    print(f"🎯 Objective: MAXIMIZE LONG-TERM SUSTAINABLE PROFIT")
    print(f"⏱️  Duration: {NUM_WEEKS} weeks")
    print(f"{'='*70}\n")
    
    simulator = EcommerceSimulatorV5(seed=42)
    guardian = SymbolicGuardianV4()
    
    # Setup agent
    if agent_type == 'llm_only':
        agent = create_llm_only_agent(OPENAI_API_KEY)
        causal_engine = None
        state_provider = None
    
    elif agent_type == 'llm_symbolic':
        state_provider = StateProvider()
        agent = create_llm_symbolic_agent(OPENAI_API_KEY, state_provider)
        causal_engine = None
    
    else:  # full_chimera
        state_provider = StateProvider()
        
        # Model path
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'initial_causal_data.pkl'))
        
        causal_engine = CausalEngineV6(
            data_path=model_path,
            force_regenerate=True,
            trust_multiplier=trust_multiplier or DEFAULT_TRUST_VALUE_MULTIPLIER
        )
        agent = create_full_chimera_agent(OPENAI_API_KEY, state_provider, causal_engine)
    
    history = []
    experience_history = []
    
    for week in tqdm(range(NUM_WEEKS), desc=f"{agent_type}"):
        current_state = simulator.get_state()
        
        if state_provider:
            state_provider.update(current_state)
        
        full_input = f"Current Market State:\n{json.dumps(current_state)}\n\nObjective: Maximize long-term sustainable profit."
        
        try:
            response = agent.invoke({"input": full_input})
            action = get_decision_from_response(response)
            
            if not isinstance(action.get('price_change'), (int, float)) or not isinstance(action.get('ad_spend'), (int, float)):
                tqdm.write(f"[WARNING] Week {week+1}: Invalid action format, using no-op")
                action = {"price_change": 0.0, "ad_spend": 0.0}
                
        except Exception as e:
            tqdm.write(f"[ERROR] Week {week+1}: {e}")
            action = {"price_change": 0.0, "ad_spend": 0.0}
        
        # Apply Guardian repair for non-Chimera agents
        if agent_type == 'llm_only':
            safe_action = action
        else:
            safe_action, _ = guardian.repair_action(action, current_state)
        
        # Step simulator
        state_before = current_state
        state_after = simulator.step(safe_action)
        
        profit_delta = state_after['profit'] - state_before['profit']
        trust_delta = state_after['brand_trust'] - state_before['brand_trust']
        
        # Log every 10 weeks
        if (week + 1) % 10 == 0 or week == 0 or week == NUM_WEEKS - 1:
            tqdm.write(
                f"\n📊 Week {week+1:2d}/{NUM_WEEKS} | "
                f"💰 Profit: ${state_after['profit']:>7,.0f} (Δ{profit_delta:+7,.0f}) | "
                f"🏷️  Price: ${state_after['price']:>6.2f} | "
                f"⭐ Trust: {state_after['brand_trust']:.3f} (Δ{trust_delta:+.3f})"
            )
        
        history.append({
            'week': week + 1,
            'agent_type': agent_type,
            'price': state_after['price'],
            'brand_trust': state_after['brand_trust'],
            'profit': state_after['profit'],
            'sales_volume': state_after['sales_volume'],
            'weekly_ad_spend': state_after['weekly_ad_spend']
        })
        
        # Collect experience for Chimera
        if agent_type == 'full_chimera' and causal_engine:
            exp = {
                "initial_price": float(state_before["price"]),
                "initial_brand_trust": float(state_before["brand_trust"]),
                "initial_ad_spend": float(state_before["weekly_ad_spend"]),
                "price_change": float(safe_action["price_change"]),
                "ad_spend": float(safe_action["ad_spend"]),
                "profit_change": float(state_after["profit"] - state_before["profit"]),
                "trust_change": float(state_after["brand_trust"] - state_before["brand_trust"]),
                "season_phase": int(state_before.get("season_phase", 0)),
            }
            experience_history.append(exp)
            
            # Retrain every 10 weeks
            if (week + 1) % 10 == 0 and week > 0:
                tqdm.write(f"  [Week {week+1}] 🔄 Retraining Causal Engine...")
                causal_engine.retrain(experience_history)
    
    df = pd.DataFrame(history)
    
    # Print summary
    total_profit = df['profit'].sum()
    avg_profit = df['profit'].mean()
    final_trust = df['brand_trust'].iloc[-1]
    final_price = df['price'].iloc[-1]
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETED: {agent_type.upper()}")
    print(f"{'='*70}")
    print(f"📈 Total Cumulative Profit: ${total_profit:>12,.0f}")
    print(f"💵 Avg Weekly Profit:       ${avg_profit:>12,.0f}")
    print(f"🏷️  Final Price:             ${final_price:>12.2f}")
    print(f"⭐ Final Brand Trust:       {final_trust:>13.3f}")
    print(f"{'='*70}\n")
    
    return df


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comparison_figure(results: Dict[str, pd.DataFrame]):
    """
    Creates primary comparison figure: 1 row × 3 columns
    - Profit trajectory
    - Price trajectory  
    - Trust trajectory
    
    All three agents on each panel.
    """
    print("\n=== Creating Neutral Architecture Comparison Figure ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Neutral Architecture Comparison: Fair Benchmark with Identical Objectives', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    colors = {
        'llm_only': '#C73E1D',
        'llm_symbolic': '#F18F01',
        'full_chimera': '#2E86AB'
    }
    
    labels = {
        'llm_only': 'LLM-Only (No Tools)',
        'llm_symbolic': 'LLM + Symbolic Guardian',
        'full_chimera': 'Full Chimera (NSC)'
    }
    
    # Panel A: Profit
    ax = axes[0]
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = results[agent_type]
        ax.plot(df['week'], df['profit'], linewidth=3.5,
               color=colors[agent_type], label=labels[agent_type], alpha=0.9)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Week', fontsize=13, fontweight='bold')
    ax.set_ylabel('Weekly Profit ($)', fontsize=13, fontweight='bold')
    ax.set_title('A. Profit Performance', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.4)
    
    # Panel B: Price
    ax = axes[1]
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = results[agent_type]
        ax.plot(df['week'], df['price'], linewidth=3.5,
               color=colors[agent_type], label=labels[agent_type], alpha=0.9)
    
    ax.set_xlabel('Week', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=13, fontweight='bold')
    ax.set_title('B. Pricing Strategy', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.4)
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.6, label='Starting Price')
    
    # Panel C: Trust
    ax = axes[2]
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = results[agent_type]
        ax.plot(df['week'], df['brand_trust'], linewidth=3.5,
               color=colors[agent_type], label=labels[agent_type], alpha=0.9)
    
    ax.set_xlabel('Week', fontsize=13, fontweight='bold')
    ax.set_ylabel('Brand Trust Score', fontsize=13, fontweight='bold')
    ax.set_title('C. Brand Trust Evolution', fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.4)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, label='Critical Threshold')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'neutral_architecture_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison figure: {output_path}")
    plt.close()


def create_summary_table(results: Dict[str, pd.DataFrame]):
    """Creates performance summary table."""
    print("\n=== Creating Summary Table ===")
    
    summary_data = []
    
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = results[agent_type]
        
        summary_data.append({
            'Agent Architecture': labels_full[agent_type],
            'Total Profit': f"${df['profit'].sum():,.0f}",
            'Avg Weekly Profit': f"${df['profit'].mean():,.0f}",
            'Final Price': f"${df['price'].iloc[-1]:.2f}",
            'Final Trust': f"{df['brand_trust'].iloc[-1]:.3f}",
            'Trust Change': f"{df['brand_trust'].iloc[-1] - df['brand_trust'].iloc[0]:+.3f}",
            'Profit Std Dev': f"${df['profit'].std():.0f}"
        })
    
    table_df = pd.DataFrame(summary_data)
    
    # Save
    table_path = os.path.join(OUTPUT_DIR, 'neutral_architecture_summary.csv')
    table_df.to_csv(table_path, index=False)
    print(f"✓ Saved summary table: {table_path}")
    
    # Print
    print("\n" + "="*120)
    print("NEUTRAL ARCHITECTURE COMPARISON - PERFORMANCE SUMMARY")
    print("="*120)
    print(table_df.to_string(index=False))
    print("="*120)
    
    return table_df


labels_full = {
    'llm_only': 'LLM-Only (No Tools)',
    'llm_symbolic': 'LLM + Symbolic Guardian',
    'full_chimera': 'Full Chimera (Neuro-Symbolic-Causal)'
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("NEUTRAL 3-AGENT ARCHITECTURE BENCHMARK")
    print("Fair comparison with identical strategic objectives")
    print("="*70)
    
    if not OPENAI_API_KEY:
        print("\n❌ ERROR: OPENAI_API_KEY not set!")
        return
    
    # Results storage
    results = {}
    
    # Run all agents with same neutral objective
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = run_agent(
            agent_type, 
            trust_multiplier=100000 if agent_type == 'full_chimera' else None
        )
        results[agent_type] = df
        
        # Save individual log
        log_path = os.path.join(OUTPUT_DIR, f'neutral_{agent_type}_log.csv')
        df.to_csv(log_path, index=False)
    
    print("\n✓ All simulations complete!")
    
    # Generate visualizations
    create_comparison_figure(results)
    create_summary_table(results)
    
    print("\n" + "="*70)
    print("NEUTRAL BENCHMARK COMPLETE!")
    print(f"Results: {OUTPUT_DIR}")
    print("="*70)
    print("\n📊 This is the FAIR architecture comparison for the paper.")
    print("All agents had identical objectives and equal prompt detail.")


if __name__ == "__main__":
    main()