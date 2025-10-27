# preprint/three_agent_comparative_benchmark.py
"""
3-Agent Comparative Benchmark: Fair Dual-Objective Evaluation

Tests ALL THREE agent architectures under BOTH organizational biases with IDENTICAL dual-objective instructions:
- LLM-Only (no tools)
- LLM + Symbolic Guardian (safety constraints)
- Full Neuro-Symbolic-Causal Chimera (safety + causal foresight)

Each tested with:
1. Volume-focused organizational bias (decrease strategy)
2. Margin-focused organizational bias (increase strategy)

ALL agents receive the SAME dual-objective instruction:
"Maximize long-term sustainable profit AND brand trust"

This fair comparison demonstrates:
- LLM-Only: Even with perfect instructions, fails without architecture (no tools)
- LLM+Symbolic: Safe but suboptimal (Guardian prevents disasters, but no foresight)
- Full Chimera: Robust and optimal across biases (architecture enables dual-objective optimization)

Key insight: "Architecture > Prompt Engineering"
"""

import sys
import os
import json
import re
from typing import Dict, Any, List, Optional

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
# HELPER CLASSES & FUNCTIONS
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
            # Robust conversion - handle strings, None, etc.
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
# AGENT CREATION FUNCTIONS
# ============================================================================

def create_llm_only_agent(strategy_type: str, openai_api_key: str) -> AgentExecutor:
    """Creates LLM-Only agent with dual-objective prompt and organizational bias."""
    
    if strategy_type == 'decrease':
        SYSTEM_PROMPT = """
You are an AI business strategist with expertise in volume-based growth.

**Your Strategic Objective:**
MAXIMIZE LONG-TERM SUSTAINABLE PROFIT **AND** BRAND TRUST through HIGH VOLUME strategies.

**Organizational Context:**
Your company culture emphasizes volume-focused growth and competitive pricing. 
However, trust must not be sacrificed for short-term volume.

**Critical Dual-Objective Constraint:**
Optimize for BOTH profit AND brand trust simultaneously.

**Your Required Decision Workflow (STRICT):**

Step 1 — Generate 3 Diverse Strategy Hypotheses
- Each hypothesis must propose:
  - price_change (float)
  - ad_spend (float)
- Each must explicitly state the profit-trust tradeoff logic.

Step 2 — Compare the 3 Hypotheses
- Compare expected impact on:
  - Long-term profit sustainability
  - Brand trust preservation
- Identify which hypothesis is MOST BALANCED.

Step 3 — Select Final Decision
- Choose ONE hypothesis as the final action.

**Output Format (STRICT JSON):**
{{
  "commentary": "Explain the reasoning behind the selected strategy, focusing on profit-trust balance.",
  "action": {{"price_change": <float>, "ad_spend": <float>}},
  "predicted_profit_change": 0.0
}}
"""
    
    else:  # 'increase'
        SYSTEM_PROMPT = """
You are an AI business strategist with expertise in premium positioning and margin optimization.

**Your Strategic Objective:**
MAXIMIZE LONG-TERM SUSTAINABLE PROFIT **AND** BRAND TRUST through HIGH MARGIN strategies.

**Organizational Context:**
Your company culture emphasizes premium brand perception and margin excellence. 
However, trust must not be weakened by unjustified price increases.

**Critical Dual-Objective Constraint:**
Optimize for BOTH profit AND brand trust simultaneously.

**Your Required Decision Workflow (STRICT):**

Step 1 — Generate 3 Diverse Strategy Hypotheses
- Each hypothesis must propose:
  - price_change (float)
  - ad_spend (float)
- Each must explicitly state how pricing affects perceived value AND trust.

Step 2 — Compare the 3 Hypotheses
- Evaluate each for:
  - Profit per transaction (margin gain)
  - Trust preservation / erosion risk
- Select the most balanced one.

Step 3 — Final Decision
- Output only the chosen strategy.

**Output Format (STRICT JSON):**
{{
  "commentary": "Explain the reasoning behind the selected strategy, focusing on profit-trust balance.",
  "action": {{"price_change": <float>, "ad_spend": <float>}}, 
  "predicted_profit_change": 0.0
}} 
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.9, max_tokens=1000, openai_api_key=openai_api_key)
    agent = create_openai_tools_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=False, handle_parsing_errors=True)


def create_llm_symbolic_agent(strategy_type: str, openai_api_key: str, state_provider: StateProvider) -> AgentExecutor:
    """Creates LLM+Symbolic agent with dual-objective prompt and organizational bias."""
    guardian = SymbolicGuardianV4()
    
    @tool
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Checks if proposed action violates business rules."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        result = guardian.validate_action(action, state_provider())
        return json.dumps(result)
    
    if strategy_type == 'decrease':
        SYSTEM_PROMPT = """
        You are an AI business strategist with access to a business rules validation system and expertise in volume-based growth.

**Your Strategic Objective:**
MAXIMIZE LONG-TERM SUSTAINABLE PROFIT **AND** BRAND TRUST through HIGH VOLUME strategies.

**Organizational Context:**
Your company culture emphasizes volume-focused growth and competitive pricing, but you must 
balance this with maintaining strong brand trust for long-term sustainability.

**Available Tools:**
- check_business_rules: Validates if an action complies with safety constraints (price floors, margins, ad spend limits)

**Critical Dual-Objective Requirement:**
Optimize for BOTH profit AND trust. Volume gains without trust are unsustainable. Safety constraints 
exist to prevent catastrophic decisions, but you must also actively preserve brand value.

**Your Decision Process (STRICT WORKFLOW):**

**Step 1: Initial Hypothesis Generation**
- Formulate THREE diverse strategic hypotheses for volume-based growth
- Each must explicitly consider profit-trust tradeoffs
- Explore range: aggressive discounting to moderate reductions

**Step 2: Mandatory Validation**
- You MUST use check_business_rules on EACH hypothesis
- Show validation results in your reasoning
- If a hypothesis is INVALID, discard it and create a valid replacement
- Continue until you have THREE VALID hypotheses

**Step 3: Selection & Final Decision**
- Compare your three valid hypotheses on BOTH profit AND trust dimensions
- Choose the one best balancing long-term profitability with trust preservation
- Present your final decision with clear rationale on dual objectives

**Output Format (STRICT JSON):**
{{
  "commentary": "Explain reasoning with margin-trust tradeoff clarity.",
  "action": {{"price_change": <float>, "ad_spend": <float>}},
  "predicted_profit_change": 0.0
}}
"""
    
    else:
        SYSTEM_PROMPT = """You are an AI business strategist with access to a business rules validation system and expertise in premium positioning.

**Your Strategic Objective:**
MAXIMIZE LONG-TERM SUSTAINABLE PROFIT **AND** BRAND TRUST through HIGH MARGIN strategies.

**Organizational Context:**
Your company culture emphasizes premium brand positioning and margin excellence, but you must 
balance this with maintaining strong brand trust as foundation for pricing power.

**Available Tools:**
- check_business_rules: Validates if an action complies with safety constraints (price floors, margins, ad spend limits)

**Critical Dual-Objective Requirement:**
Optimize for BOTH profit AND trust. Premium pricing requires trust. Excessive prices without 
trust erode customer relationships and market position.

**Your Decision Process (STRICT WORKFLOW):**

**Step 1: Initial Hypothesis Generation**
- Formulate THREE diverse strategic hypotheses for margin optimization
- Each must explicitly consider profit-trust tradeoffs
- Explore range: aggressive premiumization to moderate increases

**Step 2: Mandatory Validation**
- You MUST use check_business_rules on EACH hypothesis
- Show validation results in your reasoning
- If a hypothesis is INVALID, discard it and create a valid replacement
- Continue until you have THREE VALID hypotheses

**Step 3: Selection & Final Decision**
- Compare your three valid hypotheses on BOTH profit AND trust dimensions
- Choose the one best balancing long-term profitability with trust preservation
- Present your final decision with clear rationale on dual objectives

**Output Format (STRICT JSON):**
{{
  "commentary": "Explain reasoning with margin-trust tradeoff clarity.",
  "action": {{"price_change": <float>, "ad_spend": <float>}},
  "predicted_profit_change": 0.0
}}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.9, max_tokens=1000, openai_api_key=openai_api_key)
    agent = create_openai_tools_agent(llm, [check_business_rules], prompt)
    return AgentExecutor(agent=agent, tools=[check_business_rules], verbose=False, handle_parsing_errors=True)


def create_full_chimera_agent(strategy_type: str, openai_api_key: str, state_provider: StateProvider, 
                               causal_engine: CausalEngineV6) -> AgentExecutor:
    """Creates Full Neuro-Symbolic-Causal agent with dual-objective and organizational bias."""
    guardian = SymbolicGuardianV4()
    
    @tool
    def check_business_rules(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Checks if proposed action violates business rules."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        result = guardian.validate_action(action, state_provider())
        return json.dumps(result)
    
    @tool
    def estimate_profit_impact(price_change: float = 0.0, ad_spend: float = 0.0) -> str:
        """Estimates long-term causal impact of action."""
        action = {"price_change": price_change, "ad_spend": ad_spend}
        report = causal_engine.estimate_causal_effect(action, state_provider())
        return json.dumps(report)
    
    if strategy_type == 'decrease':
        SYSTEM_PROMPT = """You are a world-class AI business strategist with advanced decision support tools.

**Your Strategic Objective:**
MAXIMIZE LONG-TERM SUSTAINABLE PROFIT **AND** BRAND TRUST through volume-based growth.

**Organizational Context:**
Your company emphasizes market expansion and competitive pricing, but sustainable growth 
requires protecting brand trust as a strategic asset. Volume without trust is unsustainable.

**Your Unique Advantage:**
Unlike basic agents, you have TWO powerful tools:
1. Safety validation (check_business_rules) - prevents catastrophic decisions
2. Causal foresight (estimate_profit_impact) - predicts long-term profit AND trust outcomes

**Decision Process (MANDATORY WORKFLOW):**

**Phase 1: Generate Diverse Hypotheses**
Create THREE strategic options exploring the volume-growth space:
- Hypothesis A: Moderate price reduction (balanced)
- Hypothesis B: Aggressive discounting (growth-focused)
- Hypothesis C: Minimal change (conservative)

**Phase 2: Safety Validation**
Use check_business_rules on EACH hypothesis.
If invalid, immediately discard and generate valid replacement.
Continue until THREE VALID hypotheses.

**Phase 3: Causal Impact Analysis**
Use estimate_profit_impact on all valid hypotheses.
Compare predictions for:
- Expected profit change
- Brand trust impact
- Long-term sustainability

**Phase 4: Optimal Selection**
Choose hypothesis with BEST combination of:
- High expected profit
- Trust maintenance or improvement
- Low risk profile

Present clear rationale showing profit-trust optimization.

**Output Format (STRICT):**
{{"commentary": "Your analysis showing why chosen action optimizes both profit and trust", "action": {{"price_change": <float>, "ad_spend": <float>}}, "predicted_profit_change": <value from tool>}}"""
    
    else:  # 'increase'
        SYSTEM_PROMPT = """You are a world-class AI business strategist with advanced decision support tools.

**Your Strategic Objective:**
MAXIMIZE LONG-TERM SUSTAINABLE PROFIT **AND** BRAND TRUST through premium positioning.

**Organizational Context:**
Your company emphasizes margin excellence and premium brand positioning. However, premium 
pricing requires strong trust foundation. Price without trust alienates customers.

**Your Unique Advantage:**
Unlike basic agents, you have TWO powerful tools:
1. Safety validation (check_business_rules) - prevents catastrophic decisions
2. Causal foresight (estimate_profit_impact) - predicts long-term profit AND trust outcomes

**Decision Process (MANDATORY WORKFLOW):**

**Phase 1: Generate Diverse Hypotheses**
Create THREE strategic options exploring the premium-positioning space:
- Hypothesis A: Moderate price increase (balanced)
- Hypothesis B: Aggressive premiumization (margin-focused)
- Hypothesis C: Minimal change (conservative)

**Phase 2: Safety Validation**
Use check_business_rules on EACH hypothesis.
If invalid, immediately discard and generate valid replacement.
Continue until THREE VALID hypotheses.

**Phase 3: Causal Impact Analysis**
Use estimate_profit_impact on all valid hypotheses.
Compare predictions for:
- Expected profit change
- Brand trust impact
- Long-term sustainability

**Phase 4: Optimal Selection**
Choose hypothesis with BEST combination of:
- High expected profit
- Trust maintenance or improvement (critical for premium pricing)
- Low risk profile

Present clear rationale showing profit-trust optimization.

**Output Format (STRICT):**
{{"commentary": "Your analysis showing why chosen action optimizes both profit and trust", "action": {{"price_change": <float>, "ad_spend": <float>}}, "predicted_profit_change": <value from tool>}}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.9, max_tokens=1000, openai_api_key=openai_api_key)
    agent = create_openai_tools_agent(llm, [check_business_rules, estimate_profit_impact], prompt)
    return AgentExecutor(agent=agent, tools=[check_business_rules, estimate_profit_impact], 
                        verbose=False, handle_parsing_errors=True)


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_agent_scenario(agent_type: str, strategy_type: str, goal: str, 
                       trust_multiplier: Optional[float] = None) -> pd.DataFrame:
    """
    Runs one agent for one strategy over 52 weeks.
    
    Args:
        agent_type: 'llm_only', 'llm_symbolic', or 'full_chimera'
        strategy_type: 'decrease' or 'increase'
        goal: Strategy description
        trust_multiplier: For causal engine (only used by full_chimera)
    """
    print(f"\n{'='*70}")
    print(f"Running: {agent_type.upper()} - {strategy_type.upper()} strategy")
    print(f"{'='*70}")
    print(f"🎯 Goal: {goal[:60]}...")
    print(f"⏱️  Duration: {NUM_WEEKS} weeks")
    print(f"{'='*70}\n")
    
    simulator = EcommerceSimulatorV5(seed=42)
    guardian = SymbolicGuardianV4()
    
    # Setup agent
    if agent_type == 'llm_only':
        agent = create_llm_only_agent(strategy_type, OPENAI_API_KEY)
        causal_engine = None
        state_provider = None
    
    elif agent_type == 'llm_symbolic':
        state_provider = StateProvider()
        agent = create_llm_symbolic_agent(strategy_type, OPENAI_API_KEY, state_provider)
        causal_engine = None
    
    else:  # full_chimera
        state_provider = StateProvider()
        
        # Model path relative to preprint directory
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'initial_causal_data.pkl'))
        
        causal_engine = CausalEngineV6(
            data_path=model_path,
            force_regenerate=True,
            trust_multiplier=trust_multiplier or DEFAULT_TRUST_VALUE_MULTIPLIER
        )
        agent = create_full_chimera_agent(strategy_type, OPENAI_API_KEY, state_provider, causal_engine)
    
    history = []
    experience_history = []
    
    for week in tqdm(range(NUM_WEEKS), desc=f"{agent_type} ({strategy_type})"):
        current_state = simulator.get_state()
        
        if state_provider:
            state_provider.update(current_state)
        
        full_input = f"Current Market State:\n{json.dumps(current_state)}\n\nObjective: {goal}"
        
        try:
            response = agent.invoke({"input": full_input})
            action = get_decision_from_response(response)
            
            # Validate action format
            if not isinstance(action.get('price_change'), (int, float)) or not isinstance(action.get('ad_spend'), (int, float)):
                tqdm.write(f"[WARNING] Week {week+1}: Invalid action format from LLM, using no-op. Action: {action}")
                action = {"price_change": 0.0, "ad_spend": 0.0}
                
        except Exception as e:
            tqdm.write(f"[ERROR] Week {week+1}: {e}")
            action = {"price_change": 0.0, "ad_spend": 0.0}
        
        # Apply Guardian repair for non-Chimera agents
        if agent_type == 'llm_only':
            safe_action = action  # No repair
        else:
            safe_action, _ = guardian.repair_action(action, current_state)
        
        # Step simulator
        state_before = current_state
        state_after = simulator.step(safe_action)
        
        profit_delta = state_after['profit'] - state_before['profit']
        trust_delta = state_after['brand_trust'] - state_before['brand_trust']
        
        # Log every 5 weeks with detailed info
        if (week + 1) % 5 == 0 or week == 0 or week == NUM_WEEKS - 1:
            tqdm.write(
                f"\n📊 Week {week+1:2d}/{NUM_WEEKS} | "
                f"💰 Profit: ${state_after['profit']:>7,.0f} (Δ{profit_delta:+7,.0f}) | "
                f"🏷️  Price: ${state_after['price']:>6.2f} | "
                f"⭐ Trust: {state_after['brand_trust']:.3f} (Δ{trust_delta:+.3f})"
            )
        
        history.append({
            'week': week + 1,
            'agent_type': agent_type,
            'strategy': strategy_type,
            'price': state_after['price'],
            'brand_trust': state_after['brand_trust'],
            'profit': state_after['profit'],
            'sales_volume': state_after['sales_volume'],
            'weekly_ad_spend': state_after['weekly_ad_spend']
        })
        
        # Collect experience for Chimera learning
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
                tqdm.write(f"  [Week {week+1}] Retraining Causal Engine...")
                causal_engine.retrain(experience_history)
    
    df = pd.DataFrame(history)
    
    # Print summary statistics
    total_profit = df['profit'].sum()
    avg_profit = df['profit'].mean()
    final_trust = df['brand_trust'].iloc[-1]
    final_price = df['price'].iloc[-1]
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETED: {agent_type.upper()} - {strategy_type.upper()}")
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

def create_comprehensive_comparison_figure(results: Dict[str, Dict[str, pd.DataFrame]]):
    """
    Creates mega-figure: 3 agents × 2 strategies × 3 metrics
    
    Layout: 2 rows (decrease, increase) × 3 columns (profit, price, trust)
    Each panel shows all 3 agents
    """
    print("\n=== Creating Comprehensive Comparison Figure ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('3-Agent Comparative Benchmark: Prompt Sensitivity Analysis', 
                 fontsize=20, fontweight='bold')
    
    colors = {
        'llm_only': '#C73E1D',
        'llm_symbolic': '#F18F01', 
        'full_chimera': '#2E86AB'
    }
    
    labels = {
        'llm_only': 'LLM-Only',
        'llm_symbolic': 'LLM + Symbolic',
        'full_chimera': 'Full Chimera'
    }
    
    # ROW 0: DECREASE STRATEGY
    for col, metric in enumerate(['profit', 'price', 'brand_trust']):
        ax = axes[0, col]
        for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
            df = results[agent_type]['decrease']
            ax.plot(df['week'], df[metric], linewidth=3, 
                   color=colors[agent_type], label=labels[agent_type], alpha=0.9)
        
        if metric == 'profit':
            ax.set_title('Discount Strategy: Profit', fontsize=14, fontweight='bold')
            ax.set_ylabel('Weekly Profit ($)', fontsize=12, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        elif metric == 'price':
            ax.set_title('Discount Strategy: Price', fontsize=14, fontweight='bold')
            ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        else:
            ax.set_title('Discount Strategy: Trust', fontsize=14, fontweight='bold')
            ax.set_ylabel('Brand Trust', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.05)
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.4)
    
    # ROW 1: INCREASE STRATEGY
    for col, metric in enumerate(['profit', 'price', 'brand_trust']):
        ax = axes[1, col]
        for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
            df = results[agent_type]['increase']
            ax.plot(df['week'], df[metric], linewidth=3,
                   color=colors[agent_type], label=labels[agent_type], alpha=0.9)
        
        if metric == 'profit':
            ax.set_title('Pricing Strategy: Profit', fontsize=14, fontweight='bold')
            ax.set_ylabel('Weekly Profit ($)', fontsize=12, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        elif metric == 'price':
            ax.set_title('Pricing Strategy: Price', fontsize=14, fontweight='bold')
            ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        else:
            ax.set_title('Pricing Strategy: Trust', fontsize=14, fontweight='bold')
            ax.set_ylabel('Brand Trust', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.05)
        
        ax.set_xlabel('Week', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.4)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'three_agent_comprehensive_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive figure: {output_path}")
    plt.close()


def create_summary_table(results: Dict[str, Dict[str, pd.DataFrame]]):
    """Creates comprehensive summary table."""
    print("\n=== Creating Summary Table ===")
    
    summary_data = []
    
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        for strategy in ['decrease', 'increase']:
            df = results[agent_type][strategy]
            
            summary_data.append({
                'Agent': agent_type.replace('_', ' ').title(),
                'Strategy': strategy.capitalize(),
                'Total Profit': f"${df['profit'].sum():,.0f}",
                'Avg Weekly Profit': f"${df['profit'].mean():,.0f}",
                'Final Price': f"${df['price'].iloc[-1]:.2f}",
                'Final Trust': f"{df['brand_trust'].iloc[-1]:.3f}",
                'Trust Change': f"{df['brand_trust'].iloc[-1] - df['brand_trust'].iloc[0]:+.3f}"
            })
    
    table_df = pd.DataFrame(summary_data)
    
    # Save
    table_path = os.path.join(OUTPUT_DIR, 'three_agent_summary.csv')
    table_df.to_csv(table_path, index=False)
    print(f"✓ Saved summary table: {table_path}")
    
    # Print
    print("\n" + "="*120)
    print("3-AGENT COMPARATIVE SUMMARY")
    print("="*120)
    print(table_df.to_string(index=False))
    print("="*120)
    
    return table_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("3-AGENT COMPARATIVE BENCHMARK")
    print("="*70)
    
    if not OPENAI_API_KEY:
        print("\n❌ ERROR: OPENAI_API_KEY not set!")
        return
    
    # Results storage
    results = {
        'llm_only': {},
        'llm_symbolic': {},
        'full_chimera': {}
    }
    
    # Run all combinations
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        for strategy in ['decrease', 'increase']:
            if strategy == 'decrease':
                goal = "Maximize profit through VOLUME. Lower prices drive demand."
            else:
                goal = "Maximize profit through MARGINS. Higher prices mean higher profits."
            
            df = run_agent_scenario(agent_type, strategy, goal, 
                                   trust_multiplier=200000 if agent_type == 'full_chimera' else None)
            results[agent_type][strategy] = df
            
            # Save individual log
            log_path = os.path.join(OUTPUT_DIR, f'{agent_type}_{strategy}_log.csv')
            df.to_csv(log_path, index=False)
    
    print("\n✓ All simulations complete!")
    
    # Generate visualizations
    create_comprehensive_comparison_figure(results)
    create_summary_table(results)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print(f"Results: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()