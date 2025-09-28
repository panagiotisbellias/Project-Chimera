# =============================================================================
# quant_test_causal_engine.py
#
# Description:
#   Phase 1 of our rigorous testing plan for the Causal Engine.
#   This script performs a deep analysis of the input data and features to
#   validate their quality, distributions, and, most importantly, their
#   actual historical relationship with future profit.
#
#   Tests included:
#     1.1: Feature Correlation Heatmap.
#     1.2: Feature Distribution Plots.
#     1.3: Feature vs. Future Profit Relationship Analysis.
# =============================================================================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Add project root to path to import our data preparation script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.quant_prepare_training_data import create_features, FEATURE_COLS_DEFAULT, TRAINING_HORIZON

# --- Configuration ---
# Directory to save all test results
OUTPUT_DIR = "results/quant/causal_engine_tests_phase1"

# --- Helper Function for Plotting ---
def plot_feature_outcome_relationship(df, feature_name, outcome_name, bins=10):
    """
    Bins a continuous feature and plots the mean outcome for each bin.
    This helps visualize the relationship between a feature and a future outcome.
    """
    # Create bins of equal size based on the feature's quantiles
    try:
        df[f'{feature_name}_bin'] = pd.qcut(df[feature_name], bins, labels=False, duplicates='drop')
        binned_data = df.groupby(f'{feature_name}_bin')[outcome_name].mean()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=binned_data.index, y=binned_data.values, palette="viridis")
        plt.title(f'Average Future Profit vs. {feature_name} Bins', fontsize=16)
        plt.xlabel(f'{feature_name} (Binned from Low to High)')
        plt.ylabel(f'Average {outcome_name}')
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the plot
        output_path = os.path.join(OUTPUT_DIR, f'phase1_test3_relationship_{feature_name}_vs_profit.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved relationship plot for '{feature_name}'")
    except Exception as e:
        print(f"Could not plot relationship for {feature_name}: {e}")

# --- Main Test Execution ---
if __name__ == "__main__":
    print("--- Starting Phase 1: Deep Data & Feature Analysis ---")
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"All results will be saved in: '{OUTPUT_DIR}'")

    # --- Data Loading ---
    # Use the exact same function as our main pipeline to ensure consistency
    df_features = create_features()
    
    if df_features is not None:
        # --- Test 1.1: Feature Correlation Heatmap ---
        print("\nRunning Test 1.1: Feature Correlation Heatmap...")
        plt.figure(figsize=(12, 10))
        # Ensure we only correlate the features used by the model
        correlation_matrix = df_features[FEATURE_COLS_DEFAULT].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.savefig(os.path.join(OUTPUT_DIR, 'phase1_test1_feature_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved correlation heatmap.")

        # --- Test 1.2: Feature Distribution Plots ---
        print("\nRunning Test 1.2: Feature Distribution Plots...")
        num_features = len(FEATURE_COLS_DEFAULT)
        # Create a grid of subplots
        fig, axes = plt.subplots(nrows=(num_features + 2) // 3, ncols=3, figsize=(18, num_features * 1.5))
        axes = axes.flatten() # Flatten the grid to easily iterate over it
        
        for i, col in enumerate(FEATURE_COLS_DEFAULT):
            sns.histplot(df_features[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=12)
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'phase1_test2_feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved feature distribution plots.")

        # --- Test 1.3: Feature vs. Future Profit Relationship ---
        print("\nRunning Test 1.3: Feature vs. Future Profit Analysis...")
        # Calculate the actual future profit (our ground truth)
        outcome_col = f'profit_change_h{TRAINING_HORIZON}'
        df_features[outcome_col] = df_features['Close'].shift(-TRAINING_HORIZON) / df_features['Close'] - 1
        
        # Plot the relationship for our key strategic features
        features_to_analyze = [
            "Rsi_14",
            "Roc_5",
            "Macdh_12_26_9",
            "Bb_position",
            "Price_vs_sma10",
            "Sma10_vs_sma50"
        ]
        
        for feature in features_to_analyze:
            if feature in df_features.columns:
                plot_feature_outcome_relationship(df_features.copy(), feature, outcome_col)
            else:
                print(f"Warning: Feature '{feature}' not found for relationship analysis.")

        print("\n--- Phase 1 Testing Complete ---")