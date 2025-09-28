# =============================================================================
# generate_final_report.py (Final Corrected Version)
#
# Description:
#   This script loads the final backtest results and generates a clean,
#   professional, and analytically correct 4-panel performance report.
#   Fixes the Max Drawdown calculation and formatting error.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def generate_final_report():
    """
    Loads backtest CSVs and generates the final, corrected, and enhanced
    performance report dashboard.
    """
    print("--- Loading data from backtest CSV files ---")
    
    # --- 1. Load Data ---
    try:
        history_df = pd.read_csv("results/quant/final_backtest_history.csv")
        actions_df = pd.read_csv("results/quant/final_backtest_actions.csv")
    except FileNotFoundError:
        print("❌ ERROR: CSV files not found in 'results/quant/' directory. Please run the backtest first.")
        return

    print("--- Preparing data for visualization ---")
    
    # --- 2. Prepare Data Robustly ---
    def extract_market_data(market_data_str):
        try:
            return json.loads(market_data_str.replace("'", "\""))
        except (json.JSONDecodeError, TypeError):
            return {}
            
    market_data_cols = pd.DataFrame(history_df['market_data'].apply(extract_market_data).tolist())
    
    expected_market_cols = ['Close', 'Rsi_14']
    for col in expected_market_cols:
        if col not in market_data_cols.columns:
            market_data_cols[col] = np.nan

    history_df = pd.concat([history_df.drop('market_data', axis=1), market_data_cols], axis=1)

    if history_df['Close'].isnull().all():
        print("❌ ERROR: Could not extract 'Close' price from history CSV.")
        return

    # Normalize values for comparison
    initial_portfolio_value = history_df['portfolio_value'].iloc[0]
    initial_btc_price = history_df['Close'].iloc[0]
    history_df['portfolio_normalized'] = 100 * (history_df['portfolio_value'] / initial_portfolio_value)
    history_df['btc_normalized'] = 100 * (history_df['Close'] / initial_btc_price)
    
    # The formatting string `:.2%` will handle the multiplication by 100.
    cumulative_roll_max = history_df['portfolio_value'].cummax()
    history_df['drawdown_ratio'] = (history_df['portfolio_value'] / cumulative_roll_max) - 1.0
    max_drawdown = history_df['drawdown_ratio'].min() # This will be a negative ratio
    # --- END OF FIX ---
    
    # Calculate other final metrics
    final_value = history_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_portfolio_value) - 1
    trades = actions_df[actions_df['type'] != 'HOLD']
    
    print("--- Generating the final, clean report ---")

    # --- 3. Create the 2x2 Dashboard ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle('Project Chimera: Final Quant Agent Performance', fontsize=28, weight='bold')

    # Plot 1: Normalized Performance
    axes[0, 0].plot(history_df.index, history_df['portfolio_normalized'], label='Chimera Agent', color='royalblue', lw=2.5)
    axes[0, 0].plot(history_df.index, history_df['btc_normalized'], label='BTC Buy & Hold', color='black', linestyle='--', lw=1.5)
    axes[0, 0].set_title('Chimera Agent vs. BTC Hold', fontsize=20, weight='bold')
    axes[0, 0].set_ylabel('Performance (Normalized to 100)')
    axes[0, 0].legend(loc='upper left', fontsize=12)
    axes[0, 0].grid(True, which='major', linestyle='--', linewidth=0.5)

    # Plot 2: Portfolio Drawdown History
    # We plot the ratio and format the y-axis ticks as percentages
    axes[0, 1].plot(history_df.index, history_df['drawdown_ratio'] * 100, color='firebrick', lw=2)
    axes[0, 1].fill_between(history_df.index, history_df['drawdown_ratio'] * 100, 0, color='firebrick', alpha=0.1)
    axes[0, 1].set_title('Portfolio Drawdown History', fontsize=20, weight='bold')
    axes[0, 1].set_ylabel('Drawdown from Peak (%)')
    axes[0, 1].axhline(0, color='black', linestyle='-', lw=0.5)
    # Correctly format the Y-axis to show percentages
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))


    # Plot 3: Strategy Map on Price Chart
    ax3 = axes[1, 0]
    ax3.plot(history_df.index, history_df['Close'], color='darkgray', lw=2, label='BTC Price')
    
    trades_with_price = trades.merge(history_df[['Close']], left_on='day', right_index=True)
    
    action_types = {'BUY': '^', 'SELL': 'v', 'SHORT': 'o'}
    colors = {'BUY': 'green', 'SELL': 'red', 'SHORT': 'purple'}
    
    for atype, marker in action_types.items():
        signals = trades_with_price[trades_with_price['type'] == atype]
        if not signals.empty:
            ax3.scatter(signals['day'], signals['Close'], color=colors[atype], marker=marker, s=200, 
                        label=atype, edgecolors='black', zorder=10)
    
    ax3.set_title('Strategy Map on Price Chart', fontsize=20, weight='bold')
    ax3.set_ylabel('BTC Price ($)')
    ax3.set_xlabel('Trading Day')
    ax3.legend()

    # Plot 4: Key Performance Indicators (with correct formatting)
    ax4 = axes[1, 1]
    metrics_text = (
        f"----- Final Results -----\n\n"
        f"Final Portfolio Value: ${final_value:,.2f}\n"
        f"Total Return: {total_return:.2%}\n\n"
        f"----- Key Metrics -----\n\n"
        f"Max Drawdown: {max_drawdown:.2%}\n" # The f-string now correctly formats the ratio
        f"Total Trades: {len(trades)}\n"
    )
    ax4.text(0.5, 0.5, metrics_text, fontsize=20, va='center', ha='center', linespacing=1.8,
             bbox=dict(boxstyle="round,pad=1", fc='whitesmoke', ec='black', lw=1, alpha=0.9))
    ax4.set_title('Key Performance Indicators', fontsize=20, weight='bold')
    ax4.axis('off')

    # --- Save the Final Report ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = "results/Chimera_Performance_Report_Final.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ Enhanced and CORRECTED report saved to '{output_path}'")
    plt.show()

if __name__ == '__main__':
    if os.path.exists("results/final_backtest_history.csv") and os.path.exists("results/final_backtest_actions.csv"):
        generate_final_report()
    else:
        print("Could not find result CSVs. Please run the main backtest script first.")