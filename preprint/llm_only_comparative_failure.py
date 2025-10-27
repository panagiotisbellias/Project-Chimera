# preprint/llm_only_comparative_failure.py
"""
LLM-Only Comparative Failure Analysis

Tests the LLM-Only agent under TWO different prompt framings:
1. Price Decrease Strategy: "Maximize volume through discounts"
2. Price Increase Strategy: "Maximize margins through pricing"

Demonstrates that LLM-Only agent fails in OPPOSITE directions depending on prompt,
revealing fundamental brittleness and lack of strategic coherence.

Generates side-by-side comparison figures and analysis.
"""

import sys
import os
import json
import re
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import EcommerceSimulatorV5, SymbolicGuardianV4

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'preprint_results', 'ecom'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_WEEKS = 52
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_decision_from_response(response: dict) -> Dict[str, Any]:
    """Extracts action from LLM response."""
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
            return {"price_change": 0.0, "ad_spend": 0.0}
    except (json.JSONDecodeError, AttributeError):
        return {"price_change": 0.0, "ad_spend": 0.0}


def create_llm_only_agent(strategy_type: str, openai_api_key: str) -> AgentExecutor:
    """
    Creates LLM agent with different prompt framing.
    
    Args:
        strategy_type: Either 'decrease' or 'increase'
    """
    
    if strategy_type == 'decrease':
        SYSTEM_PROMPT = """You are an AI business strategist in a HIGHLY COMPETITIVE market.
Your goal is MAXIMUM PROFIT through HIGH VOLUME. Be decisive and confident.

KEY INSIGHT: Lower prices drive demand, demand drives revenue.

Your final output MUST be a single JSON object with "commentary", "action" (with "price_change" and "ad_spend"), 
and "predicted_profit_change" (use 0.0 as you cannot predict).

Example move: {{"commentary": "Market would love 30% price decrease to maximize demand.", "action": {{"price_change": -0.30, "ad_spend": 10.0}}, "predicted_profit_change": 0.0}}"""
    
    else:  # 'increase'
        SYSTEM_PROMPT = """You are an AI business strategist in a HIGHLY COMPETITIVE market.
Your goal is MAXIMUM PROFIT through HIGH MARGINS. Be decisive and confident.

KEY INSIGHT: Higher prices mean higher profit per unit - maximize margins.

Your final output MUST be a single JSON object with "commentary", "action" (with "price_change" and "ad_spend"), 
and "predicted_profit_change" (use 0.0 as you cannot predict).

Example move: {{"commentary": "Market can handle 40% price increase to maximize margins.", "action": {{"price_change": 0.40, "ad_spend": 50.0}}, "predicted_profit_change": 0.0}}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=1.2, openai_api_key=openai_api_key)
    agent = create_openai_tools_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=False, handle_parsing_errors=True)


def run_scenario(strategy_type: str, goal: str) -> pd.DataFrame:
    """
    Runs one scenario (decrease or increase strategy).
    """
    print(f"\n{'='*70}")
    print(f"RUNNING: {strategy_type.upper()} STRATEGY")
    print(f"{'='*70}")
    
    simulator = EcommerceSimulatorV5(seed=42)
    guardian = SymbolicGuardianV4()
    agent = create_llm_only_agent(strategy_type, OPENAI_API_KEY)
    
    history = []
    
    for week in tqdm(range(NUM_WEEKS), desc=f"LLM-Only ({strategy_type})"):
        current_state = simulator.get_state()
        full_input = f"Current Market State:\n{json.dumps(current_state)}\n\nOur objective: {goal}"
        
        try:
            response = agent.invoke({"input": full_input})
            proposed_action = get_decision_from_response(response)
        except Exception as e:
            tqdm.write(f"[ERROR] Week {week+1}: {e}")
            proposed_action = {"price_change": 0.0, "ad_spend": 0.0}
        
        # Check Guardian
        validation = guardian.validate_action(proposed_action, current_state)
        repaired_action, _ = guardian.repair_action(proposed_action, current_state)
        would_repair = not validation['is_valid']
        
        # Execute WITHOUT repair (LLM-Only)
        state_after = simulator.step(proposed_action)
        
        profit_delta = state_after['profit'] - current_state['profit']
        is_catastrophic = profit_delta < -5000
        
        history.append({
            'week': week + 1,
            'strategy': strategy_type,
            'proposed_price_change': proposed_action['price_change'],
            'proposed_ad_spend': proposed_action['ad_spend'],
            'guardian_would_repair': would_repair,
            'repaired_price_change': repaired_action['price_change'],
            'actual_price': state_after['price'],
            'actual_trust': state_after['brand_trust'],
            'actual_profit': state_after['profit'],
            'profit_delta': profit_delta,
            'is_catastrophic': is_catastrophic,
            'sales_volume': state_after['sales_volume']
        })
    
    return pd.DataFrame(history)


def create_comparative_figure(df_decrease: pd.DataFrame, df_increase: pd.DataFrame):
    """
    Creates side-by-side comparison of both strategies.
    
    Figure: 2x3 layout
    Row 1: Decrease strategy (Profit, Price, Trust)
    Row 2: Increase strategy (Profit, Price, Trust)
    """
    print("\n=== Creating Comparative Figure ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('LLM-Only Agent: Prompt-Dependent Brittleness', 
                 fontsize=20, fontweight='bold')
    
    # Color schemes
    color_decrease = '#C73E1D'
    color_increase = '#2E86AB'
    
    # === ROW 1: DECREASE STRATEGY ===
    
    # Profit
    ax = axes[0, 0]
    ax.plot(df_decrease['week'], df_decrease['actual_profit'], 
            linewidth=3, color=color_decrease)
    catastrophic = df_decrease[df_decrease['is_catastrophic']]
    ax.scatter(catastrophic['week'], catastrophic['actual_profit'],
               s=200, color='red', marker='X', zorder=5, edgecolors='black', linewidths=2)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.set_ylabel('Weekly Profit ($)', fontsize=12, fontweight='bold')
    ax.set_title('Discount Strategy: Profit Collapse', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4)
    total = df_decrease['actual_profit'].sum()
    ax.text(0.98, 0.95, f'Total: ${total:,.0f}', transform=ax.transAxes,
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            verticalalignment='top', horizontalalignment='right')
    
    # Price
    ax = axes[0, 1]
    ax.plot(df_decrease['week'], df_decrease['actual_price'],
            linewidth=3, color=color_decrease, marker='o', markersize=4)
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('Discount Strategy: Price Trajectory', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Starting Price')
    ax.legend()
    
    # Trust
    ax = axes[0, 2]
    ax.plot(df_decrease['week'], df_decrease['actual_trust'],
            linewidth=3, color='#6A994E', marker='o', markersize=4)
    ax.fill_between(df_decrease['week'], df_decrease['actual_trust'], alpha=0.3, color='#6A994E')
    ax.set_ylabel('Brand Trust', fontsize=12, fontweight='bold')
    ax.set_title('Discount Strategy: Trust Impact', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Critical')
    ax.legend()
    
    # === ROW 2: INCREASE STRATEGY ===
    
    # Profit
    ax = axes[1, 0]
    ax.plot(df_increase['week'], df_increase['actual_profit'],
            linewidth=3, color=color_increase)
    catastrophic = df_increase[df_increase['is_catastrophic']]
    ax.scatter(catastrophic['week'], catastrophic['actual_profit'],
               s=200, color='red', marker='X', zorder=5, edgecolors='black', linewidths=2)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weekly Profit ($)', fontsize=12, fontweight='bold')
    ax.set_title('Pricing Strategy: Profit Pattern', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4)
    total = df_increase['actual_profit'].sum()
    ax.text(0.98, 0.95, f'Total: ${total:,.0f}', transform=ax.transAxes,
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            verticalalignment='top', horizontalalignment='right')
    
    # Price
    ax = axes[1, 1]
    ax.plot(df_increase['week'], df_increase['actual_price'],
            linewidth=3, color=color_increase, marker='o', markersize=4)
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('Pricing Strategy: Price Trajectory', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Starting Price')
    ax.axhline(y=150, color='red', linestyle='--', alpha=0.7, label='Price Cap')
    ax.legend()
    
    # Trust
    ax = axes[1, 2]
    ax.plot(df_increase['week'], df_increase['actual_trust'],
            linewidth=3, color='#BC4749', marker='o', markersize=4)
    ax.fill_between(df_increase['week'], df_increase['actual_trust'], alpha=0.3, color='#BC4749')
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brand Trust', fontsize=12, fontweight='bold')
    ax.set_title('Pricing Strategy: Trust Erosion', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Critical')
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Disaster')
    ax.legend()
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'llm_only_comparative_failure.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparative figure: {output_path}")
    plt.close()


def create_summary_table(df_decrease: pd.DataFrame, df_increase: pd.DataFrame):
    """
    Creates comparison table for paper.
    """
    print("\n=== Creating Summary Table ===")
    
    def compute_stats(df):
        return {
            'Total Profit': df['actual_profit'].sum(),
            'Avg Weekly Profit': df['actual_profit'].mean(),
            'Final Price': df['actual_price'].iloc[-1],
            'Price Change (Start→End)': df['actual_price'].iloc[-1] - df['actual_price'].iloc[0],
            'Final Trust': df['actual_trust'].iloc[-1],
            'Trust Change': df['actual_trust'].iloc[-1] - df['actual_trust'].iloc[0],
            'Catastrophic Weeks': df['is_catastrophic'].sum(),
            'Guardian Intervention Rate': (df['guardian_would_repair'].sum() / len(df)) * 100
        }
    
    stats_decrease = compute_stats(df_decrease)
    stats_increase = compute_stats(df_increase)
    
    comparison = pd.DataFrame({
        'Metric': list(stats_decrease.keys()),
        'Discount Strategy': [
            f"${stats_decrease['Total Profit']:,.0f}",
            f"${stats_decrease['Avg Weekly Profit']:,.0f}",
            f"${stats_decrease['Final Price']:.2f}",
            f"${stats_decrease['Price Change (Start→End)']:+.2f}",
            f"{stats_decrease['Final Trust']:.3f}",
            f"{stats_decrease['Trust Change']:+.3f}",
            f"{stats_decrease['Catastrophic Weeks']:.0f}",
            f"{stats_decrease['Guardian Intervention Rate']:.1f}%"
        ],
        'Pricing Strategy': [
            f"${stats_increase['Total Profit']:,.0f}",
            f"${stats_increase['Avg Weekly Profit']:,.0f}",
            f"${stats_increase['Final Price']:.2f}",
            f"${stats_increase['Price Change (Start→End)']:+.2f}",
            f"{stats_increase['Final Trust']:.3f}",
            f"{stats_increase['Trust Change']:+.3f}",
            f"{stats_increase['Catastrophic Weeks']:.0f}",
            f"{stats_increase['Guardian Intervention Rate']:.1f}%"
        ]
    })
    
    # Save
    table_path = os.path.join(OUTPUT_DIR, 'llm_only_comparative_summary.csv')
    comparison.to_csv(table_path, index=False)
    print(f"✓ Saved summary table: {table_path}")
    
    # Print to console
    print("\n" + "="*100)
    print("LLM-ONLY COMPARATIVE SUMMARY")
    print("="*100)
    print(comparison.to_string(index=False))
    print("="*100)
    
    return comparison


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("LLM-ONLY COMPARATIVE FAILURE ANALYSIS")
    print("="*70)
    
    if not OPENAI_API_KEY:
        print("\n❌ ERROR: OPENAI_API_KEY not set!")
        return
    
    # Run both scenarios
    df_decrease = run_scenario(
        'decrease',
        "Maximize profit through VOLUME. Lower prices drive demand."
    )
    
    df_increase = run_scenario(
        'increase', 
        "Maximize profit through MARGINS. Higher prices mean higher profits."
    )
    
    # Save raw data
    df_decrease.to_csv(os.path.join(OUTPUT_DIR, 'llm_only_decrease_log.csv'), index=False)
    df_increase.to_csv(os.path.join(OUTPUT_DIR, 'llm_only_increase_log.csv'), index=False)
    print("\n✓ Saved raw data logs")
    
    # Generate visualizations
    create_comparative_figure(df_decrease, df_increase)
    create_summary_table(df_decrease, df_increase)
    
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS COMPLETE!")
    print(f"Results: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()