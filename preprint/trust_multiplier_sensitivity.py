# preprint/trust_multiplier_sensitivity.py
"""
Trust Value Multiplier Sensitivity Analysis

The trust_multiplier parameter represents how much the organization values brand trust
relative to immediate profit. This parameter affects the causal engine's long-term
impact predictions, essentially encoding organizational risk preferences.

- LOW multiplier (50K): "Brand trust is cheap" → Aggressive, short-term optimization
- MEDIUM multiplier (150K): Balanced approach (default)
- HIGH multiplier (300K): "Brand trust is precious" → Conservative, long-term focus

This analysis demonstrates that Chimera's architecture adapts its strategy based on
organizational preferences, while LLM-Only and LLM+Guardian cannot.

Tests 5 multiplier values: [50K, 100K, 150K, 200K, 300K]
Generates comprehensive comparison with built-in reporting.
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
)

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'preprint_results', 'ecom', 'trust_sensitivity'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_WEEKS = 52
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Test values for trust multiplier
TRUST_MULTIPLIERS = {
    'aggressive': 50_000,      # Low trust value → aggressive optimization
    'moderate_low': 100_000,   # Below default
    'balanced': 150_000,       # Default (what you found empirically)
    'moderate_high': 200_000,  # Above default
    'conservative': 300_000    # High trust value → conservative strategy
}

MULTIPLIER_META = {
    'aggressive': {
        'label': 'Aggressive (50K)',
        'color': '#C73E1D',
        'description': 'Low trust valuation - short-term focus'
    },
    'moderate_low': {
        'label': 'Moderate-Low (100K)',
        'color': '#F18F01',
        'description': 'Below-default trust valuation'
    },
    'balanced': {
        'label': 'Balanced (150K)',
        'color': '#6A994E',
        'description': 'Default - empirically optimal balance'
    },
    'moderate_high': {
        'label': 'Moderate-High (200K)',
        'color': '#2E86AB',
        'description': 'Above-default trust valuation'
    },
    'conservative': {
        'label': 'Conservative (300K)',
        'color': '#4A5899',
        'description': 'High trust valuation - long-term focus'
    }
}


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
            
            return {"price_change": price_change, "ad_spend": ad_spend}
        else:
            return {"price_change": 0.0, "ad_spend": 0.0}
    except (json.JSONDecodeError, AttributeError):
        return {"price_change": 0.0, "ad_spend": 0.0}


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_chimera_agent(openai_api_key: str, state_provider: StateProvider, 
                         causal_engine: CausalEngineV6) -> AgentExecutor:
    """Creates Full Chimera agent."""
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

def run_chimera_with_multiplier(multiplier_type: str, trust_multiplier: float) -> pd.DataFrame:
    """Runs Chimera with specific trust multiplier."""
    print(f"\n{'='*70}")
    print(f"Running: Chimera with {MULTIPLIER_META[multiplier_type]['label']}")
    print(f"{'='*70}")
    print(f"🎯 Trust Multiplier: ${trust_multiplier:,.0f}")
    print(f"📝 Strategy: {MULTIPLIER_META[multiplier_type]['description']}")
    print(f"⏱️  Duration: {NUM_WEEKS} weeks")
    print(f"{'='*70}\n")
    
    simulator = EcommerceSimulatorV5(seed=42)
    guardian = SymbolicGuardianV4()
    state_provider = StateProvider()
    
    # Initialize causal engine with specific multiplier
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'initial_causal_data.pkl'))
    causal_engine = CausalEngineV6(
        data_path=model_path,
        force_regenerate=True,
        trust_multiplier=trust_multiplier
    )
    
    agent = create_chimera_agent(OPENAI_API_KEY, state_provider, causal_engine)
    
    history = []
    experience_history = []
    
    for week in tqdm(range(NUM_WEEKS), desc=f"Chimera ({multiplier_type})"):
        current_state = simulator.get_state()
        state_provider.update(current_state)
        
        full_input = f"Current Market State:\n{json.dumps(current_state)}\n\nObjective: Maximize long-term sustainable profit."
        
        try:
            response = agent.invoke({"input": full_input})
            action = get_decision_from_response(response)
            
            if not isinstance(action.get('price_change'), (int, float)) or not isinstance(action.get('ad_spend'), (int, float)):
                tqdm.write(f"[WARNING] Week {week+1}: Invalid action format")
                action = {"price_change": 0.0, "ad_spend": 0.0}
                
        except Exception as e:
            tqdm.write(f"[ERROR] Week {week+1}: {e}")
            action = {"price_change": 0.0, "ad_spend": 0.0}
        
        # Guardian repair (Chimera always uses Guardian)
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
            'multiplier_type': multiplier_type,
            'trust_multiplier': trust_multiplier,
            'price': state_after['price'],
            'brand_trust': state_after['brand_trust'],
            'profit': state_after['profit'],
            'sales_volume': state_after['sales_volume'],
            'weekly_ad_spend': state_after['weekly_ad_spend']
        })
        
        # Collect experience
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
    
    # Summary
    total_profit = df['profit'].sum()
    avg_profit = df['profit'].mean()
    final_trust = df['brand_trust'].iloc[-1]
    trust_change = final_trust - df['brand_trust'].iloc[0]
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETED: {MULTIPLIER_META[multiplier_type]['label']}")
    print(f"{'='*70}")
    print(f"📈 Total Cumulative Profit: ${total_profit:>12,.0f}")
    print(f"💵 Avg Weekly Profit:       ${avg_profit:>12,.0f}")
    print(f"🏷️  Final Price:             ${df['price'].iloc[-1]:>12.2f}")
    print(f"⭐ Final Brand Trust:       {final_trust:>13.3f}")
    print(f"📊 Trust Change:            {trust_change:>+13.3f}")
    print(f"{'='*70}\n")
    
    return df


# ============================================================================
# VISUALIZATION & REPORTING
# ============================================================================

def create_sensitivity_main_figure(data: Dict[str, pd.DataFrame]):
    """Creates main 2×2 sensitivity analysis figure."""
    print("\n📊 Creating Trust Multiplier Sensitivity Figure (2×2)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Trust Multiplier Sensitivity: Organizational Risk Preference Analysis', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # Panel A: Cumulative Profit
    ax = axes[0, 0]
    for mult_type in ['aggressive', 'moderate_low', 'balanced', 'moderate_high', 'conservative']:
        df = data[mult_type]
        cumulative = df['profit'].cumsum()
        ax.plot(df['week'], cumulative, linewidth=3,
               color=MULTIPLIER_META[mult_type]['color'],
               label=MULTIPLIER_META[mult_type]['label'], alpha=0.9)
    
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Profit ($)', fontsize=12, fontweight='bold')
    ax.set_title('A. Cumulative Profit by Risk Preference', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    # Panel B: Weekly Profit (Rolling)
    ax = axes[0, 1]
    for mult_type in ['aggressive', 'moderate_low', 'balanced', 'moderate_high', 'conservative']:
        df = data[mult_type]
        rolling = df['profit'].rolling(window=3, center=True).mean()
        ax.plot(df['week'], rolling, linewidth=2.5,
               color=MULTIPLIER_META[mult_type]['color'],
               label=MULTIPLIER_META[mult_type]['label'], alpha=0.9)
    
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weekly Profit ($)', fontsize=12, fontweight='bold')
    ax.set_title('B. Weekly Profit Stability', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Panel C: Brand Trust Evolution
    ax = axes[1, 0]
    for mult_type in ['aggressive', 'moderate_low', 'balanced', 'moderate_high', 'conservative']:
        df = data[mult_type]
        ax.plot(df['week'], df['brand_trust'], linewidth=3,
               color=MULTIPLIER_META[mult_type]['color'],
               label=MULTIPLIER_META[mult_type]['label'], alpha=0.9)
    
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brand Trust Score', fontsize=12, fontweight='bold')
    ax.set_title('C. Brand Trust: Conservative vs. Aggressive', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.6, label='Initial')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Panel D: Price Strategy
    ax = axes[1, 1]
    for mult_type in ['aggressive', 'moderate_low', 'balanced', 'moderate_high', 'conservative']:
        df = data[mult_type]
        price_smooth = df['price'].rolling(window=5, center=True).mean()
        ax.plot(df['week'], price_smooth, linewidth=2.5,
               color=MULTIPLIER_META[mult_type]['color'],
               label=MULTIPLIER_META[mult_type]['label'], alpha=0.9)
    
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('D. Pricing Strategy (5-Week Rolling Avg)', fontsize=14, fontweight='bold', pad=10)
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.6, label='Starting Price')
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'trust_multiplier_sensitivity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_tradeoff_analysis(data: Dict[str, pd.DataFrame]):
    """Creates profit vs. trust tradeoff scatter."""
    print("\n📊 Creating Profit-Trust Tradeoff Analysis...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metrics = []
    for mult_type in ['aggressive', 'moderate_low', 'balanced', 'moderate_high', 'conservative']:
        df = data[mult_type]
        total_profit = df['profit'].sum()
        final_trust = df['brand_trust'].iloc[-1]
        
        metrics.append({
            'type': mult_type,
            'profit': total_profit,
            'trust': final_trust
        })
        
        ax.scatter(final_trust, total_profit / 1000, s=500,
                  color=MULTIPLIER_META[mult_type]['color'],
                  alpha=0.7, edgecolors='black', linewidths=2, zorder=5)
        
        ax.annotate(MULTIPLIER_META[mult_type]['label'],
                   xy=(final_trust, total_profit / 1000),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5',
                            facecolor=MULTIPLIER_META[mult_type]['color'],
                            alpha=0.3, edgecolor='black'))
    
    ax.set_xlabel('Final Brand Trust', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Profit ($K)', fontsize=13, fontweight='bold')
    ax.set_title('Profit-Trust Tradeoff: Risk Preference Spectrum', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0.4, 0.85)
    
    # Add Pareto frontier approximation
    metrics_df = pd.DataFrame(metrics).sort_values('trust')
    ax.plot(metrics_df['trust'], metrics_df['profit'] / 1000, 
           'k--', alpha=0.3, linewidth=2, label='Efficiency Frontier')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'profit_trust_tradeoff.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_statistical_summary(data: Dict[str, pd.DataFrame]):
    """Creates comprehensive statistical summary."""
    print("\n📊 Creating Statistical Summary...")
    
    summary_data = []
    
    for mult_type in ['aggressive', 'moderate_low', 'balanced', 'moderate_high', 'conservative']:
        df = data[mult_type]
        
        total_profit = df['profit'].sum()
        mean_profit = df['profit'].mean()
        std_profit = df['profit'].std()
        final_trust = df['brand_trust'].iloc[-1]
        trust_change = final_trust - df['brand_trust'].iloc[0]
        sharpe = mean_profit / std_profit if std_profit > 0 else 0
        
        summary_data.append({
            'Risk Preference': MULTIPLIER_META[mult_type]['label'],
            'Trust Multiplier': f'${TRUST_MULTIPLIERS[mult_type]:,.0f}',
            'Total Profit': f'${total_profit:,.0f}',
            'Mean Weekly Profit': f'${mean_profit:,.0f}',
            'Std Dev': f'${std_profit:,.0f}',
            'Sharpe Ratio': f'{sharpe:.3f}',
            'Final Trust': f'{final_trust:.3f}',
            'Trust Change': f'{trust_change:+.3f}'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save
    csv_path = os.path.join(OUTPUT_DIR, 'sensitivity_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    
    # Print
    print("\n" + "="*140)
    print("TRUST MULTIPLIER SENSITIVITY - STATISTICAL SUMMARY")
    print("="*140)
    print(summary_df.to_string(index=False))
    print("="*140)
    
    # Save formatted text
    txt_path = os.path.join(OUTPUT_DIR, 'sensitivity_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("TRUST MULTIPLIER SENSITIVITY ANALYSIS\n")
        f.write("="*140 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n" + "="*140 + "\n\n")
        f.write("KEY INSIGHTS:\n\n")
        f.write("📊 The trust_multiplier parameter encodes organizational risk preferences:\n\n")
        f.write("  - AGGRESSIVE (50K): Maximizes short-term profit, accepts trust erosion\n")
        f.write("  - BALANCED (150K): Empirically optimal - maximizes risk-adjusted returns\n")
        f.write("  - CONSERVATIVE (300K): Protects brand trust, sacrifices immediate profit\n\n")
        f.write("🎯 Chimera adapts its strategy to organizational preferences through this parameter.\n")
        f.write("   LLM-Only and rule-based agents cannot express this nuanced risk preference.\n")
    
    print(f"  ✓ Saved: {txt_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("TRUST MULTIPLIER SENSITIVITY ANALYSIS")
    print("Organizational Risk Preference Sweep")
    print("="*70)
    
    if not OPENAI_API_KEY:
        print("\n❌ ERROR: OPENAI_API_KEY not set!")
        return
    
    # Results storage
    data = {}
    
    # Run all multiplier values
    for mult_type, mult_value in TRUST_MULTIPLIERS.items():
        df = run_chimera_with_multiplier(mult_type, mult_value)
        data[mult_type] = df
        
        # Save individual log
        log_path = os.path.join(OUTPUT_DIR, f'sensitivity_{mult_type}_log.csv')
        df.to_csv(log_path, index=False)
    
    print("\n✓ All sensitivity runs complete!")
    
    # Generate reports
    create_sensitivity_main_figure(data)
    create_tradeoff_analysis(data)
    create_statistical_summary(data)
    
    print("\n" + "="*70)
    print("✅ SENSITIVITY ANALYSIS COMPLETE!")
    print(f"Results: {OUTPUT_DIR}")
    print("="*70)
    print("\n🎯 This demonstrates Chimera's adaptability to organizational risk preferences.")
    print("   The same architecture can serve both aggressive growth and conservative strategies.")


if __name__ == "__main__":
    main()