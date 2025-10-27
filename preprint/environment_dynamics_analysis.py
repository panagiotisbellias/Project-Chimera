# preprint/environment_dynamics_analysis.py
"""
Environment Dynamics Analysis for Preprint

This script analyzes and visualizes the core dynamics of the EcommerceSimulatorV5
to demonstrate that it exhibits realistic business behavior:
1. Price elasticity of demand
2. Brand trust effects on demand
3. Advertising impact (logarithmic)
4. Seasonal patterns

Generates publication-quality figures for the paper.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import EcommerceSimulatorV5

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory (relative to preprint folder)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'preprint_results', 'ecom'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_price_elasticity(simulator: EcommerceSimulatorV5, base_state: Dict[str, Any]) -> pd.DataFrame:
    """
    Tests how demand responds to price changes while holding other factors constant.
    
    Returns DataFrame with columns: price, demand, revenue, profit
    """
    print("\n=== Analyzing Price Elasticity ===")
    
    results = []
    price_range = np.linspace(50, 200, 31)  # $50 to $200
    
    for target_price in price_range:
        # Reset to base state
        simulator.reset(seed=42)
        
        # Manually set the state to our controlled conditions
        simulator.state.update({
            'price': target_price,
            'brand_trust': base_state['brand_trust'],
            'weekly_ad_spend': base_state['weekly_ad_spend'],
            'season_phase': base_state['season_phase']
        })
        
        # Take a step with zero action to observe demand
        state_after = simulator.step({'price_change': 0.0, 'ad_spend': base_state['weekly_ad_spend']})
        
        results.append({
            'price': target_price,
            'demand': state_after['sales_volume'],
            'revenue': state_after['sales_volume'] * target_price,
            'profit': state_after['profit']
        })
    
    df = pd.DataFrame(results)
    print(f"Price elasticity analysis complete: {len(df)} data points")
    return df


def analyze_trust_effect(simulator: EcommerceSimulatorV5, base_state: Dict[str, Any]) -> pd.DataFrame:
    """
    Tests how brand trust affects demand while holding other factors constant.
    
    Returns DataFrame with columns: brand_trust, demand, willingness_to_pay
    """
    print("\n=== Analyzing Brand Trust Effect ===")
    
    results = []
    trust_range = np.linspace(0.2, 1.0, 25)  # Min to max trust
    
    for trust_level in trust_range:
        simulator.reset(seed=42)
        
        # Set controlled state
        simulator.state.update({
            'price': base_state['price'],
            'brand_trust': trust_level,
            'weekly_ad_spend': base_state['weekly_ad_spend'],
            'season_phase': base_state['season_phase']
        })
        
        state_after = simulator.step({'price_change': 0.0, 'ad_spend': base_state['weekly_ad_spend']})
        
        results.append({
            'brand_trust': trust_level,
            'demand': state_after['sales_volume'],
            'profit': state_after['profit']
        })
    
    df = pd.DataFrame(results)
    print(f"Trust effect analysis complete: {len(df)} data points")
    return df


def analyze_ad_spend_effect(simulator: EcommerceSimulatorV5, base_state: Dict[str, Any]) -> pd.DataFrame:
    """
    Tests how advertising spend affects demand (should show logarithmic returns).
    
    Returns DataFrame with columns: ad_spend, demand, roi, profit
    """
    print("\n=== Analyzing Advertising Effect ===")
    
    results = []
    ad_range = np.linspace(0, 5000, 51)  # $0 to $5000
    
    for ad_spend in ad_range:
        simulator.reset(seed=42)
        
        simulator.state.update({
            'price': base_state['price'],
            'brand_trust': base_state['brand_trust'],
            'weekly_ad_spend': 0,  # Start from 0
            'season_phase': base_state['season_phase']
        })
        
        state_after = simulator.step({'price_change': 0.0, 'ad_spend': ad_spend})
        
        # Calculate ROI (Return on Investment)
        incremental_revenue = state_after['sales_volume'] * base_state['price']
        roi = (incremental_revenue - ad_spend) / max(ad_spend, 1)  # Avoid division by zero
        
        results.append({
            'ad_spend': ad_spend,
            'demand': state_after['sales_volume'],
            'profit': state_after['profit'],
            'roi': roi
        })
    
    df = pd.DataFrame(results)
    print(f"Ad spend analysis complete: {len(df)} data points")
    return df


def analyze_seasonality(simulator: EcommerceSimulatorV5, base_state: Dict[str, Any]) -> pd.DataFrame:
    """
    Tests seasonal demand patterns over a full year (52 weeks).
    
    Returns DataFrame with columns: week, season_phase, demand, seasonal_multiplier
    """
    print("\n=== Analyzing Seasonality ===")
    
    results = []
    
    for week in range(52):
        simulator.reset(seed=42)
        
        simulator.state.update({
            'price': base_state['price'],
            'brand_trust': base_state['brand_trust'],
            'weekly_ad_spend': base_state['weekly_ad_spend'],
            'season_phase': week,
            'week': week
        })
        
        state_after = simulator.step({'price_change': 0.0, 'ad_spend': base_state['weekly_ad_spend']})
        
        # Calculate implicit seasonal multiplier from observed demand
        # This is approximate since other factors also affect demand
        results.append({
            'week': week,
            'season_phase': week,
            'demand': state_after['sales_volume'],
            'profit': state_after['profit']
        })
    
    df = pd.DataFrame(results)
    print(f"Seasonality analysis complete: {len(df)} data points")
    return df


def create_dynamics_figure(price_df, trust_df, ad_df, season_df):
    """
    Creates a comprehensive 2x2 figure showing all environment dynamics.
    """
    print("\n=== Creating Comprehensive Dynamics Figure ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('E-Commerce Simulator: Core Business Dynamics', fontsize=18, fontweight='bold')
    
    # --- Panel A: Price Elasticity ---
    ax = axes[0, 0]
    ax2 = ax.twinx()
    
    ax.plot(price_df['price'], price_df['demand'], 'o-', color='#2E86AB', linewidth=2.5, markersize=6, label='Demand')
    ax2.plot(price_df['price'], price_df['profit'], 's-', color='#A23B72', linewidth=2.5, markersize=6, label='Profit')
    
    ax.set_xlabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Demand (units)', fontsize=12, fontweight='bold', color='#2E86AB')
    ax2.set_ylabel('Profit ($)', fontsize=12, fontweight='bold', color='#A23B72')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    ax.set_title('A. Price Elasticity of Demand', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for optimal price region
    optimal_idx = price_df['profit'].idxmax()
    optimal_price = price_df.iloc[optimal_idx]['price']
    optimal_profit = price_df.iloc[optimal_idx]['profit']
    ax2.annotate(f'Optimal: ${optimal_price:.0f}', 
                 xy=(optimal_price, optimal_profit),
                 xytext=(optimal_price + 20, optimal_profit * 0.9),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=10, fontweight='bold', color='red')
    
    # --- Panel B: Trust Effect ---
    ax = axes[0, 1]
    ax2 = ax.twinx()
    
    ax.plot(trust_df['brand_trust'], trust_df['demand'], 'o-', color='#F18F01', linewidth=2.5, markersize=6)
    ax2.plot(trust_df['brand_trust'], trust_df['profit'], 's-', color='#C73E1D', linewidth=2.5, markersize=6)
    
    ax.set_xlabel('Brand Trust Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Demand (units)', fontsize=12, fontweight='bold', color='#F18F01')
    ax2.set_ylabel('Profit ($)', fontsize=12, fontweight='bold', color='#C73E1D')
    ax.tick_params(axis='y', labelcolor='#F18F01')
    ax2.tick_params(axis='y', labelcolor='#C73E1D')
    ax.set_title('B. Brand Trust Impact', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    
    # --- Panel C: Advertising Returns ---
    ax = axes[1, 0]
    ax2 = ax.twinx()
    
    ax.plot(ad_df['ad_spend'], ad_df['demand'], 'o-', color='#6A994E', linewidth=2.5, markersize=6)
    ax2.plot(ad_df['ad_spend'], ad_df['roi'], 's-', color='#BC4749', linewidth=2.5, markersize=6)
    
    ax.set_xlabel('Weekly Ad Spend ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Demand (units)', fontsize=12, fontweight='bold', color='#6A994E')
    ax2.set_ylabel('ROI (Return per $1 spent)', fontsize=12, fontweight='bold', color='#BC4749')
    ax.tick_params(axis='y', labelcolor='#6A994E')
    ax2.tick_params(axis='y', labelcolor='#BC4749')
    ax.set_title('C. Advertising: Logarithmic Returns', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add annotation for diminishing returns
    ax.annotate('Diminishing\nReturns', 
                xy=(3500, ad_df[ad_df['ad_spend'] >= 3500]['demand'].iloc[0]),
                xytext=(2500, ad_df['demand'].max() * 0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red')
    
    # --- Panel D: Seasonality ---
    ax = axes[1, 1]
    
    ax.plot(season_df['week'], season_df['demand'], 'o-', color='#4A5899', linewidth=2.5, markersize=6)
    ax.fill_between(season_df['week'], season_df['demand'], alpha=0.3, color='#4A5899')
    
    ax.set_xlabel('Week of Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Demand (units)', fontsize=12, fontweight='bold')
    ax.set_title('D. Seasonal Demand Pattern', fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    
    # Add quarter markers
    for quarter in [0, 13, 26, 39]:
        ax.axvline(x=quarter, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(6, season_df['demand'].max() * 0.95, 'Q1', fontsize=10, ha='center')
    ax.text(19, season_df['demand'].max() * 0.95, 'Q2', fontsize=10, ha='center')
    ax.text(32, season_df['demand'].max() * 0.95, 'Q3', fontsize=10, ha='center')
    ax.text(45, season_df['demand'].max() * 0.95, 'Q4', fontsize=10, ha='center')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'environment_dynamics_comprehensive.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive dynamics figure: {output_path}")
    
    plt.close()


def generate_summary_statistics(price_df, trust_df, ad_df, season_df):
    """
    Generates summary statistics for the paper (LaTeX table).
    """
    print("\n=== Summary Statistics ===")
    
    # Price elasticity (approximate)
    price_low = price_df[price_df['price'] <= 80]['demand'].mean()
    price_high = price_df[price_df['price'] >= 120]['demand'].mean()
    elasticity_approx = (price_high - price_low) / price_low / ((120 - 80) / 80)
    
    # Trust effect (linear approximation)
    trust_low = trust_df[trust_df['brand_trust'] <= 0.4]['demand'].mean()
    trust_high = trust_df[trust_df['brand_trust'] >= 0.8]['demand'].mean()
    trust_impact = (trust_high - trust_low) / trust_low * 100
    
    # Ad spend ROI at different levels
    roi_low = ad_df[ad_df['ad_spend'] <= 500]['roi'].mean()
    roi_mid = ad_df[(ad_df['ad_spend'] > 500) & (ad_df['ad_spend'] <= 2000)]['roi'].mean()
    roi_high = ad_df[ad_df['ad_spend'] > 2000]['roi'].mean()
    
    # Seasonal variance
    seasonal_min = season_df['demand'].min()
    seasonal_max = season_df['demand'].max()
    seasonal_range = (seasonal_max - seasonal_min) / season_df['demand'].mean() * 100
    
    print(f"\nPrice Elasticity (approximate): {elasticity_approx:.2f}")
    print(f"Trust Impact (low vs. high): +{trust_impact:.1f}% demand increase")
    print(f"Ad Spend ROI:")
    print(f"  - Low spend (<$500): {roi_low:.2f}x")
    print(f"  - Medium spend ($500-$2000): {roi_mid:.2f}x")
    print(f"  - High spend (>$2000): {roi_high:.2f}x")
    print(f"Seasonal Demand Range: ±{seasonal_range:.1f}% from mean")
    
    # Save to text file for easy reference
    summary_path = os.path.join(OUTPUT_DIR, 'environment_dynamics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("ENVIRONMENT DYNAMICS - SUMMARY STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Price Elasticity (approximate): {elasticity_approx:.2f}\n")
        f.write(f"Trust Impact (low vs. high): +{trust_impact:.1f}% demand increase\n\n")
        f.write("Ad Spend ROI:\n")
        f.write(f"  - Low spend (<$500): {roi_low:.2f}x\n")
        f.write(f"  - Medium spend ($500-$2000): {roi_mid:.2f}x\n")
        f.write(f"  - High spend (>$2000): {roi_high:.2f}x\n\n")
        f.write(f"Seasonal Demand Range: ±{seasonal_range:.1f}% from mean\n")
    
    print(f"✓ Saved summary statistics: {summary_path}")


def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("ENVIRONMENT DYNAMICS ANALYSIS - PREPRINT")
    print("=" * 70)
    
    # Initialize simulator
    simulator = EcommerceSimulatorV5(seed=42)
    
    # Define baseline state (realistic mid-game conditions)
    base_state = {
        'price': 100.0,
        'brand_trust': 0.7,
        'weekly_ad_spend': 500.0,
        'season_phase': 10  # Mid Q1
    }
    
    print(f"\nBaseline Conditions:")
    print(f"  Price: ${base_state['price']:.2f}")
    print(f"  Brand Trust: {base_state['brand_trust']:.2f}")
    print(f"  Weekly Ad Spend: ${base_state['weekly_ad_spend']:.2f}")
    print(f"  Season Phase: Week {base_state['season_phase']}")
    
    # Run analyses
    price_df = analyze_price_elasticity(simulator, base_state)
    trust_df = analyze_trust_effect(simulator, base_state)
    ad_df = analyze_ad_spend_effect(simulator, base_state)
    season_df = analyze_seasonality(simulator, base_state)
    
    # Save raw data
    price_df.to_csv(os.path.join(OUTPUT_DIR, 'dynamics_price_elasticity.csv'), index=False)
    trust_df.to_csv(os.path.join(OUTPUT_DIR, 'dynamics_trust_effect.csv'), index=False)
    ad_df.to_csv(os.path.join(OUTPUT_DIR, 'dynamics_ad_spend.csv'), index=False)
    season_df.to_csv(os.path.join(OUTPUT_DIR, 'dynamics_seasonality.csv'), index=False)
    print("\n✓ Saved all raw data CSVs")
    
    # Create visualizations
    create_dynamics_figure(price_df, trust_df, ad_df, season_df)
    
    # Generate summary statistics
    generate_summary_statistics(price_df, trust_df, ad_df, season_df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()