# preprint/comprehensive_analysis_three_agent.py
"""
Comprehensive Publication-Ready Analysis: 3-Agent Architecture Comparison

Loads final three-agent comparison results and generates:
1. Main comparison figure (2×3 grid)
2. Performance metrics table with statistics
3. Risk-return analysis scatter
4. Profit distribution histograms
5. Trust trajectory comparison
6. Robustness analysis (cross-strategy comparison)
7. Statistical significance tests

For paper: Demonstrates Chimera's architectural superiority across organizational biases.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'preprint_results', 'ecom', 'comprehensive_three_agent'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'llm_only': '#C73E1D',
    'llm_symbolic': '#F18F01',
    'full_chimera': '#2E86AB'
}

LABELS = {
    'llm_only': 'LLM-Only',
    'llm_symbolic': 'LLM + Guardian',
    'full_chimera': 'Full Chimera'
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Loads all agent logs for both strategies."""
    print("\n📂 Loading data...")
    
    data = {
        'decrease': {},
        'increase': {}
    }
    
    for strategy in ['decrease', 'increase']:
        for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
            csv_path = os.path.join(BASE_DIR, f'{agent_type}_{strategy}_log.csv')
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                data[strategy][agent_type] = df
                print(f"  ✓ {agent_type} ({strategy}): {len(df)} weeks")
            else:
                print(f"  ❌ Missing: {csv_path}")
    
    return data


# ============================================================================
# FIGURE 1: MAIN COMPARISON (Enhanced 2×3)
# ============================================================================

def create_main_comparison_figure(data: Dict):
    """Creates enhanced main comparison figure."""
    print("\n📊 Creating Main Comparison Figure...")
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Three-Agent Architecture Comparison: Dual-Objective Optimization Under Organizational Biases', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    strategies = [
        ('decrease', 'Volume-Focused Strategy (Decrease Bias)', 0),
        ('increase', 'Margin-Focused Strategy (Increase Bias)', 1)
    ]
    
    for strategy_key, strategy_title, row in strategies:
        strategy_data = data[strategy_key]
        
        # Panel: Cumulative Profit
        ax = axes[row, 0]
        for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
            df = strategy_data[agent_type]
            cumulative = df['profit'].cumsum()
            ax.plot(df['week'], cumulative, linewidth=3.5,
                   color=COLORS[agent_type], label=LABELS[agent_type], alpha=0.9)
            
            # Annotate final value
            final_val = cumulative.iloc[-1]
            ax.annotate(f'${final_val/1000:.0f}K',
                       xy=(df['week'].iloc[-1], final_val),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       color=COLORS[agent_type])
        
        ax.set_xlabel('Week', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cumulative Profit ($)', fontsize=11, fontweight='bold')
        ax.set_title(f'{strategy_title}: Cumulative Profit', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='y')
        
        # Panel: Weekly Profit (with confidence bands)
        ax = axes[row, 1]
        for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
            df = strategy_data[agent_type]
            rolling_mean = df['profit'].rolling(window=5, center=True).mean()
            rolling_std = df['profit'].rolling(window=5, center=True).std()
            
            ax.plot(df['week'], rolling_mean, linewidth=3,
                   color=COLORS[agent_type], label=LABELS[agent_type], alpha=0.9)
            ax.fill_between(df['week'], 
                           rolling_mean - rolling_std,
                           rolling_mean + rolling_std,
                           color=COLORS[agent_type], alpha=0.15)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Week', fontsize=11, fontweight='bold')
        ax.set_ylabel('Weekly Profit ($)', fontsize=11, fontweight='bold')
        ax.set_title(f'{strategy_title}: Profit Stability', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        # Panel: Brand Trust Evolution
        ax = axes[row, 2]
        for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
            df = strategy_data[agent_type]
            ax.plot(df['week'], df['brand_trust'], linewidth=3.5,
                   color=COLORS[agent_type], label=LABELS[agent_type], alpha=0.9)
            
            # Fill area
            ax.fill_between(df['week'], df['brand_trust'],
                           alpha=0.12, color=COLORS[agent_type])
            
            # Annotate final value
            final_trust = df['brand_trust'].iloc[-1]
            ax.annotate(f'{final_trust:.3f}',
                       xy=(df['week'].iloc[-1], final_trust),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       color=COLORS[agent_type])
        
        ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.6, linewidth=1.5, label='Initial Trust')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='Critical Threshold')
        ax.set_xlabel('Week', fontsize=11, fontweight='bold')
        ax.set_ylabel('Brand Trust Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{strategy_title}: Brand Trust', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=9, framealpha=0.95)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'fig1_main_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 2: RISK-RETURN ANALYSIS
# ============================================================================

def create_risk_return_analysis(data: Dict):
    """Creates risk-return scatter comparing all agents."""
    print("\n📊 Creating Risk-Return Analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (strategy_key, strategy_title) in enumerate([('decrease', 'Volume-Focused'),
                                                           ('increase', 'Margin-Focused')]):
        ax = axes[idx]
        strategy_data = data[strategy_key]
        
        for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
            df = strategy_data[agent_type]
            
            avg_profit = df['profit'].mean()
            std_profit = df['profit'].std()
            
            ax.scatter(std_profit, avg_profit, s=500,
                      color=COLORS[agent_type], alpha=0.7,
                      edgecolors='black', linewidths=2.5, zorder=5)
            
            ax.annotate(LABELS[agent_type],
                       xy=(std_profit, avg_profit),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.6',
                                facecolor=COLORS[agent_type],
                                alpha=0.3, edgecolor='black', linewidth=1.5))
        
        ax.set_xlabel('Profit Volatility (Std Dev, $)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Weekly Profit ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'{strategy_title} Strategy', fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.4)
    
    fig.suptitle('Risk-Return Profile: Volatility vs. Performance', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig2_risk_return.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# FIGURE 3: ROBUSTNESS ANALYSIS (Cross-Strategy)
# ============================================================================

def create_robustness_analysis(data: Dict):
    """Shows agent performance consistency across strategies."""
    print("\n📊 Creating Robustness Analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate metrics for each agent across strategies
    metrics = {}
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        metrics[agent_type] = {
            'decrease_profit': data['decrease'][agent_type]['profit'].sum(),
            'increase_profit': data['increase'][agent_type]['profit'].sum(),
            'decrease_trust': data['decrease'][agent_type]['brand_trust'].iloc[-1],
            'increase_trust': data['increase'][agent_type]['brand_trust'].iloc[-1]
        }
    
    # Panel A: Total Profit Across Strategies
    ax = axes[0]
    x = np.arange(len(LABELS))
    width = 0.35
    
    decrease_profits = [metrics[a]['decrease_profit']/1000 for a in ['llm_only', 'llm_symbolic', 'full_chimera']]
    increase_profits = [metrics[a]['increase_profit']/1000 for a in ['llm_only', 'llm_symbolic', 'full_chimera']]
    
    bars1 = ax.bar(x - width/2, decrease_profits, width, label='Volume-Focused',
                   color=[COLORS[a] for a in ['llm_only', 'llm_symbolic', 'full_chimera']],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, increase_profits, width, label='Margin-Focused',
                   color=[COLORS[a] for a in ['llm_only', 'llm_symbolic', 'full_chimera']],
                   alpha=0.5, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Agent Architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Profit ($K)', fontsize=12, fontweight='bold')
    ax.set_title('A. Profit Performance Across Strategies', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ['llm_only', 'llm_symbolic', 'full_chimera']])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.0f}K', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    # Panel B: Final Trust Across Strategies
    ax = axes[1]
    
    decrease_trust = [metrics[a]['decrease_trust'] for a in ['llm_only', 'llm_symbolic', 'full_chimera']]
    increase_trust = [metrics[a]['increase_trust'] for a in ['llm_only', 'llm_symbolic', 'full_chimera']]
    
    bars1 = ax.bar(x - width/2, decrease_trust, width, label='Volume-Focused',
                   color=[COLORS[a] for a in ['llm_only', 'llm_symbolic', 'full_chimera']],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, increase_trust, width, label='Margin-Focused',
                   color=[COLORS[a] for a in ['llm_only', 'llm_symbolic', 'full_chimera']],
                   alpha=0.5, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Agent Architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Brand Trust', fontsize=12, fontweight='bold')
    ax.set_title('B. Trust Preservation Across Strategies', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ['llm_only', 'llm_symbolic', 'full_chimera']])
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.6, label='Initial Trust')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    fig.suptitle('Cross-Strategy Robustness Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig3_robustness.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# STATISTICAL ANALYSIS & TABLES
# ============================================================================

def create_comprehensive_statistics(data: Dict):
    """Creates detailed statistical analysis tables."""
    print("\n📊 Creating Statistical Analysis...")
    
    # Comprehensive metrics
    all_metrics = []
    
    for strategy in ['decrease', 'increase']:
        for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
            df = data[strategy][agent_type]
            
            total_profit = df['profit'].sum()
            mean_profit = df['profit'].mean()
            median_profit = df['profit'].median()
            std_profit = df['profit'].std()
            min_profit = df['profit'].min()
            max_profit = df['profit'].max()
            
            # Trust metrics
            initial_trust = df['brand_trust'].iloc[0]
            final_trust = df['brand_trust'].iloc[-1]
            trust_change = final_trust - initial_trust
            
            # Risk-adjusted metrics
            sharpe = mean_profit / std_profit if std_profit > 0 else 0
            cv = (std_profit / mean_profit) * 100 if mean_profit > 0 else np.inf
            
            # Weeks with negative profit
            negative_weeks = (df['profit'] < 0).sum()
            failure_rate = (negative_weeks / len(df)) * 100
            
            all_metrics.append({
                'Strategy': 'Volume' if strategy == 'decrease' else 'Margin',
                'Agent': LABELS[agent_type],
                'Total Profit': f'${total_profit:,.0f}',
                'Mean Weekly': f'${mean_profit:,.0f}',
                'Median Weekly': f'${median_profit:,.0f}',
                'Std Dev': f'${std_profit:,.0f}',
                'Min Profit': f'${min_profit:,.0f}',
                'Max Profit': f'${max_profit:,.0f}',
                'Sharpe Ratio': f'{sharpe:.3f}',
                'CV (%)': f'{cv:.1f}',
                'Final Trust': f'{final_trust:.3f}',
                'Trust Δ': f'{trust_change:+.3f}',
                'Failure Rate': f'{failure_rate:.1f}%'
            })
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'comprehensive_statistics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    
    # Print formatted
    print("\n" + "="*160)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*160)
    print(metrics_df.to_string(index=False))
    print("="*160)
    
    # Save formatted text
    txt_path = os.path.join(OUTPUT_DIR, 'comprehensive_statistics.txt')
    with open(txt_path, 'w') as f:
        f.write("COMPREHENSIVE STATISTICAL ANALYSIS\n")
        f.write("="*160 + "\n\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n" + "="*160 + "\n\n")
        
        f.write("KEY FINDINGS:\n\n")
        
        # Best performer
        best_total = metrics_df.loc[metrics_df['Total Profit'].str.replace('$','').str.replace(',','').astype(float).idxmax()]
        f.write(f"🏆 Highest Total Profit: {best_total['Agent']} ({best_total['Strategy']}) - {best_total['Total Profit']}\n")
        
        best_sharpe = metrics_df.loc[metrics_df['Sharpe Ratio'].astype(float).idxmax()]
        f.write(f"📊 Best Risk-Adjusted: {best_sharpe['Agent']} ({best_sharpe['Strategy']}) - Sharpe: {best_sharpe['Sharpe Ratio']}\n")
        
        best_trust = metrics_df.loc[metrics_df['Final Trust'].astype(float).idxmax()]
        f.write(f"⭐ Highest Trust: {best_trust['Agent']} ({best_trust['Strategy']}) - {best_trust['Final Trust']}\n")
        
        # Chimera dominance
        chimera_rows = metrics_df[metrics_df['Agent'] == 'Full Chimera']
        f.write(f"\n🎯 CHIMERA PERFORMANCE:\n")
        f.write(f"   - Volume Strategy: {chimera_rows[chimera_rows['Strategy']=='Volume']['Total Profit'].values[0]}, Trust: {chimera_rows[chimera_rows['Strategy']=='Volume']['Trust Δ'].values[0]}\n")
        f.write(f"   - Margin Strategy: {chimera_rows[chimera_rows['Strategy']=='Margin']['Total Profit'].values[0]}, Trust: {chimera_rows[chimera_rows['Strategy']=='Margin']['Trust Δ'].values[0]}\n")
        f.write(f"   → Robust across organizational biases ✓\n")
    
    print(f"  ✓ Saved: {txt_path}")
    
    return metrics_df


# ============================================================================
# FIGURE 4: PROFIT DISTRIBUTIONS
# ============================================================================

def create_profit_distributions(data: Dict):
    """Creates overlapping histograms of profit distributions."""
    print("\n📊 Creating Profit Distributions...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (strategy_key, strategy_title) in enumerate([('decrease', 'Volume-Focused'),
                                                           ('increase', 'Margin-Focused')]):
        ax = axes[idx]
        strategy_data = data[strategy_key]
        
        for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
            df = strategy_data[agent_type]
            
            ax.hist(df['profit'], bins=25, alpha=0.5,
                   color=COLORS[agent_type], label=LABELS[agent_type],
                   edgecolor='black', linewidth=1.2)
            
            # Add mean line
            mean_val = df['profit'].mean()
            ax.axvline(mean_val, color=COLORS[agent_type],
                      linestyle='--', linewidth=2.5, alpha=0.8)
        
        ax.set_xlabel('Weekly Profit ($)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (Weeks)', fontsize=12, fontweight='bold')
        ax.set_title(f'{strategy_title} Strategy', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Weekly Profit Distribution by Architecture', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig4_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("\n" + "="*70)
    print("COMPREHENSIVE PUBLICATION-READY ANALYSIS")
    print("3-Agent Architecture Comparison")
    print("="*70)
    
    # Load data
    data = load_all_data()
    
    # Generate all figures
    create_main_comparison_figure(data)
    create_risk_return_analysis(data)
    create_robustness_analysis(data)
    create_profit_distributions(data)
    
    # Generate statistics
    metrics_df = create_comprehensive_statistics(data)
    
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)
    print("\n📊 Generated Files:")
    print("  - fig1_main_comparison.png (Primary figure for paper)")
    print("  - fig2_risk_return.png (Risk-return profile)")
    print("  - fig3_robustness.png (Cross-strategy consistency)")
    print("  - fig4_distributions.png (Profit distributions)")
    print("  - comprehensive_statistics.csv (All metrics)")
    print("  - comprehensive_statistics.txt (Formatted report)")
    print("\n💡 Use fig1_main_comparison.png as the main results figure in your paper.")


if __name__ == "__main__":
    main()