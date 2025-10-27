# preprint/architecture_report.py
"""
Architecture Comparison Report Generator

Loads CSV results from neutral architecture comparison and generates
publication-quality figures and analysis for the paper.

Input: CSV logs from architecture_comparison/
Output: Multiple professional figures + statistical analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

INPUT_DIR = "../results/preprint_results/ecom/architecture_comparison"
OUTPUT_DIR = "../results/preprint_results/ecom/architecture_comparison"

# Agent metadata
AGENT_META = {
    'llm_only': {
        'label': 'LLM-Only',
        'color': '#C73E1D',
        'linestyle': '-',
        'description': 'No safety constraints, no foresight'
    },
    'llm_symbolic': {
        'label': 'LLM + Guardian',
        'color': '#F18F01',
        'linestyle': '-',
        'description': 'Symbolic safety constraints'
    },
    'full_chimera': {
        'label': 'Full Chimera',
        'color': '#2E86AB',
        'linestyle': '-',
        'description': 'Neuro-Symbolic-Causal hybrid'
    }
}


def load_data():
    """Loads all agent CSV logs."""
    print("📂 Loading data...")
    
    data = {}
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        csv_path = os.path.join(INPUT_DIR, f'neutral_{agent_type}_log.csv')
        if os.path.exists(csv_path):
            data[agent_type] = pd.read_csv(csv_path)
            print(f"  ✓ Loaded {agent_type}: {len(data[agent_type])} weeks")
        else:
            print(f"  ❌ Missing: {csv_path}")
    
    return data


def create_main_comparison_figure(data):
    """
    Creates main 2×2 comparison figure for paper.
    
    Panels:
    - Top-left: Cumulative Profit
    - Top-right: Weekly Profit (with variance bands)
    - Bottom-left: Brand Trust Evolution
    - Bottom-right: Price Strategy (smoothed)
    """
    print("\n📊 Creating Main Comparison Figure (2×2)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Architecture Comparison: Comprehensive Performance Analysis', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # ===== PANEL A: CUMULATIVE PROFIT =====
    ax = axes[0, 0]
    
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = data[agent_type]
        cumulative = df['profit'].cumsum()
        
        ax.plot(df['week'], cumulative, 
                linewidth=3.5,
                color=AGENT_META[agent_type]['color'],
                label=AGENT_META[agent_type]['label'],
                alpha=0.9)
    
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Profit ($)', fontsize=12, fontweight='bold')
    ax.set_title('A. Cumulative Profit Over Time', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    # Add final values as annotations
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = data[agent_type]
        final_value = df['profit'].sum()
        final_week = df['week'].iloc[-1]
        cumulative_final = df['profit'].cumsum().iloc[-1]
        
        ax.annotate(f'${final_value/1e6:.2f}M',
                   xy=(final_week, cumulative_final),
                   xytext=(5, 0), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   color=AGENT_META[agent_type]['color'])
    
    # ===== PANEL B: WEEKLY PROFIT WITH VARIANCE =====
    ax = axes[0, 1]
    
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = data[agent_type]
        
        # Rolling mean for smoothing
        rolling_mean = df['profit'].rolling(window=3, center=True).mean()
        rolling_std = df['profit'].rolling(window=3, center=True).std()
        
        ax.plot(df['week'], rolling_mean,
                linewidth=3,
                color=AGENT_META[agent_type]['color'],
                label=AGENT_META[agent_type]['label'],
                alpha=0.9)
        
        # Variance band
        ax.fill_between(df['week'],
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        color=AGENT_META[agent_type]['color'],
                        alpha=0.15)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weekly Profit ($)', fontsize=12, fontweight='bold')
    ax.set_title('B. Weekly Profit (3-Week Rolling Average)', fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # ===== PANEL C: BRAND TRUST EVOLUTION =====
    ax = axes[1, 0]
    
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = data[agent_type]
        
        ax.plot(df['week'], df['brand_trust'],
                linewidth=3.5,
                color=AGENT_META[agent_type]['color'],
                label=AGENT_META[agent_type]['label'],
                alpha=0.9)
        
        # Fill area
        ax.fill_between(df['week'], df['brand_trust'], 
                        alpha=0.15,
                        color=AGENT_META[agent_type]['color'])
    
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brand Trust Score', fontsize=12, fontweight='bold')
    ax.set_title('C. Brand Trust Evolution', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='Critical Threshold')
    ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='Initial Trust')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # ===== PANEL D: PRICE STRATEGY (SMOOTHED) =====
    ax = axes[1, 1]
    
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = data[agent_type]
        
        # 5-week rolling average for smoothness
        price_smooth = df['price'].rolling(window=5, center=True).mean()
        
        ax.plot(df['week'], price_smooth,
                linewidth=3,
                color=AGENT_META[agent_type]['color'],
                label=AGENT_META[agent_type]['label'],
                alpha=0.9)
    
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('D. Pricing Strategy (5-Week Rolling Avg)', fontsize=14, fontweight='bold', pad=10)
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.6, linewidth=1.5, label='Starting Price')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'architecture_comparison_main.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_risk_return_analysis(data):
    """
    Creates risk-return scatter plot.
    
    X-axis: Average weekly profit
    Y-axis: Profit volatility (std dev)
    
    Better = High return, Low volatility (top-left)
    """
    print("\n📊 Creating Risk-Return Analysis...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metrics = []
    
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = data[agent_type]
        
        avg_profit = df['profit'].mean()
        std_profit = df['profit'].std()
        
        metrics.append({
            'agent': agent_type,
            'avg_profit': avg_profit,
            'std_profit': std_profit
        })
        
        # Scatter point
        ax.scatter(avg_profit, std_profit,
                  s=400,
                  color=AGENT_META[agent_type]['color'],
                  alpha=0.7,
                  edgecolors='black',
                  linewidths=2,
                  zorder=5)
        
        # Label
        ax.annotate(AGENT_META[agent_type]['label'],
                   xy=(avg_profit, std_profit),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor=AGENT_META[agent_type]['color'],
                            alpha=0.3, edgecolor='black'))
    
    ax.set_xlabel('Average Weekly Profit ($)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Profit Volatility (Std Dev, $)', fontsize=13, fontweight='bold')
    ax.set_title('Risk-Return Profile: Architecture Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.4)
    
    # Add quadrant labels
    ax.text(0.05, 0.95, 'Low Return\nHigh Risk', transform=ax.transAxes,
           fontsize=10, alpha=0.5, va='top')
    ax.text(0.95, 0.95, 'High Return\nHigh Risk', transform=ax.transAxes,
           fontsize=10, alpha=0.5, va='top', ha='right')
    ax.text(0.05, 0.05, 'Low Return\nLow Risk', transform=ax.transAxes,
           fontsize=10, alpha=0.5, va='bottom')
    ax.text(0.95, 0.05, '🎯 High Return\nLow Risk', transform=ax.transAxes,
           fontsize=11, fontweight='bold', alpha=0.7, va='bottom', ha='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'risk_return_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()
    
    return metrics


def create_profit_distributions(data):
    """
    Creates overlapping histograms of weekly profit distributions.
    Shows stability and outliers.
    """
    print("\n📊 Creating Profit Distribution Analysis...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = data[agent_type]
        
        ax.hist(df['profit'], bins=20, 
               alpha=0.5,
               color=AGENT_META[agent_type]['color'],
               label=AGENT_META[agent_type]['label'],
               edgecolor='black',
               linewidth=1.5)
    
    ax.set_xlabel('Weekly Profit ($)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Weeks)', fontsize=13, fontweight='bold')
    ax.set_title('Weekly Profit Distribution by Architecture', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean lines
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = data[agent_type]
        mean_profit = df['profit'].mean()
        
        ax.axvline(mean_profit, 
                  color=AGENT_META[agent_type]['color'],
                  linestyle='--',
                  linewidth=2.5,
                  alpha=0.8)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'profit_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_statistical_summary(data):
    """
    Creates comprehensive statistical summary table.
    """
    print("\n📊 Creating Statistical Summary...")
    
    summary_data = []
    
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = data[agent_type]
        
        # Calculate metrics
        total_profit = df['profit'].sum()
        mean_profit = df['profit'].mean()
        median_profit = df['profit'].median()
        std_profit = df['profit'].std()
        min_profit = df['profit'].min()
        max_profit = df['profit'].max()
        
        # Trust metrics
        trust_start = df['brand_trust'].iloc[0]
        trust_end = df['brand_trust'].iloc[-1]
        trust_change = trust_end - trust_start
        trust_mean = df['brand_trust'].mean()
        
        # Risk-adjusted return (Sharpe-like)
        sharpe = mean_profit / std_profit if std_profit > 0 else 0
        
        # Coefficient of variation (lower is better)
        cv = (std_profit / mean_profit) * 100 if mean_profit > 0 else 0
        
        summary_data.append({
            'Architecture': AGENT_META[agent_type]['label'],
            'Total Profit': f'${total_profit:,.0f}',
            'Mean Weekly Profit': f'${mean_profit:,.0f}',
            'Median Weekly Profit': f'${median_profit:,.0f}',
            'Std Dev': f'${std_profit:,.0f}',
            'Min Profit': f'${min_profit:,.0f}',
            'Max Profit': f'${max_profit:,.0f}',
            'Sharpe Ratio': f'{sharpe:.3f}',
            'Coeff. of Variation': f'{cv:.1f}%',
            'Final Trust': f'{trust_end:.3f}',
            'Trust Change': f'{trust_change:+.3f}',
            'Mean Trust': f'{trust_mean:.3f}'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'statistical_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    
    # Print to console
    print("\n" + "="*150)
    print("ARCHITECTURE COMPARISON - STATISTICAL SUMMARY")
    print("="*150)
    print(summary_df.to_string(index=False))
    print("="*150)
    
    # Save as formatted text
    txt_path = os.path.join(OUTPUT_DIR, 'statistical_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("ARCHITECTURE COMPARISON - STATISTICAL SUMMARY\n")
        f.write("="*150 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n" + "="*150 + "\n\n")
        
        f.write("KEY INSIGHTS:\n\n")
        
        # Winner analysis
        best_total = summary_df.loc[summary_df['Total Profit'].str.replace('$','').str.replace(',','').astype(float).idxmax()]
        f.write(f"🏆 Highest Total Profit: {best_total['Architecture']} ({best_total['Total Profit']})\n")
        
        best_sharpe = summary_df.loc[summary_df['Sharpe Ratio'].astype(float).idxmax()]
        f.write(f"📊 Best Risk-Adjusted Return: {best_sharpe['Architecture']} (Sharpe: {best_sharpe['Sharpe Ratio']})\n")
        
        best_trust = summary_df.loc[summary_df['Final Trust'].astype(float).idxmax()]
        f.write(f"⭐ Highest Brand Trust: {best_trust['Architecture']} ({best_trust['Final Trust']})\n")
    
    print(f"  ✓ Saved: {txt_path}")
    
    return summary_df


def create_performance_radar(data):
    """
    Creates radar chart comparing architectures across multiple dimensions.
    """
    print("\n📊 Creating Performance Radar Chart...")
    
    from math import pi
    
    # Define metrics (normalized 0-1, higher is better)
    metrics = ['Total Profit', 'Stability', 'Trust Building', 'Avg Profit', 'Consistency']
    
    agent_scores = {}
    
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        df = data[agent_type]
        
        # Normalize metrics to 0-1 scale (higher is better)
        total_profit_norm = df['profit'].sum() / 2000000  # Assume max ~2M
        stability_norm = 1 - (df['profit'].std() / 15000)  # Lower std is better
        trust_norm = df['brand_trust'].iloc[-1]  # Final trust
        avg_profit_norm = df['profit'].mean() / 50000  # Assume max ~50k
        consistency_norm = 1 - (df['profit'].std() / df['profit'].mean()) if df['profit'].mean() > 0 else 0
        
        agent_scores[agent_type] = [
            min(total_profit_norm, 1.0),
            min(stability_norm, 1.0),
            trust_norm,
            min(avg_profit_norm, 1.0),
            min(consistency_norm, 1.0)
        ]
    
    # Setup radar
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for agent_type in ['llm_only', 'llm_symbolic', 'full_chimera']:
        values = agent_scores[agent_type]
        values += values[:1]
        
        ax.plot(angles, values,
               linewidth=2.5,
               color=AGENT_META[agent_type]['color'],
               label=AGENT_META[agent_type]['label'])
        
        ax.fill(angles, values,
               color=AGENT_META[agent_type]['color'],
               alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, framealpha=0.95)
    plt.title('Multi-Dimensional Performance Comparison', 
             fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'performance_radar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("ARCHITECTURE COMPARISON REPORT GENERATOR")
    print("="*70)
    
    # Load data
    data = load_data()
    
    if len(data) < 3:
        print("\n❌ ERROR: Missing data files!")
        print(f"Expected 3 CSV files in: {INPUT_DIR}")
        return
    
    # Generate all figures
    create_main_comparison_figure(data)
    risk_metrics = create_risk_return_analysis(data)
    create_profit_distributions(data)
    create_statistical_summary(data)
    create_performance_radar(data)
    
    print("\n" + "="*70)
    print("✅ REPORT GENERATION COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*70)
    print("\n📊 Generated Files:")
    print("  - architecture_comparison_main.png (Primary figure for paper)")
    print("  - risk_return_analysis.png")
    print("  - profit_distributions.png")
    print("  - performance_radar.png")
    print("  - statistical_summary.csv")
    print("  - statistical_summary.txt")
    print("\n💡 Use 'architecture_comparison_main.png' as the main figure in your paper.")


if __name__ == "__main__":
    main()