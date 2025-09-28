# =============================================================================
# quant_test_causal_engine_phase2.py
#
# Description:
#   Phase 2 of our rigorous testing plan: "Reading the Model's Brain".
#   This script analyzes the trained Causal Forest model to understand its
#   internal logic and decision-making patterns.
#
#   Tests included:
#     2.1: Partial Dependence Plots (PDP) for all key features.
#     2.2: SHAP (SHapley Additive exPlanations) analysis for specific scenarios.
# =============================================================================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import shap
import pickle

# Add project root to path to import our components and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.quant_prepare_training_data import create_features, FEATURE_COLS_DEFAULT
from src.components import CausalEngineV7_Quant

# --- Configuration ---
OUTPUT_DIR = "results/quant/causal_engine_tests_phase2"
# The standard action we want to explain (e.g., what influences a BUY decision)
ACTION_TO_EXPLAIN = {'type': 'BUY', 'amount': 0.5}

# --- Main Test Execution ---
if __name__ == "__main__":
    print("--- Starting Phase 2: Reading the Model's Brain ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"All results will be saved in: '{OUTPUT_DIR}'")

    # --- Load Data and Train the Causal Engine ---
    # We need both the data and the trained model for this phase.
    print("Loading data using the standard pipeline...")
    df_features = create_features()
    
    print("Initializing and training the Causal Engine...")
    # The engine trains itself upon initialization.
    causal_engine = CausalEngineV7_Quant(data_path="causal_training_data_balanced.csv")
    
    # --- Test 2.1: Partial Dependence Plots (PDP) ---
    print("\nRunning Test 2.1: Partial Dependence Plots...")
    
    features_to_plot_pdp = [
        "Rsi_14", "Roc_5", "Macdh_12_26_9", "Bb_position",
        "Price_vs_sma10", "Sma10_vs_sma50"
    ]
    
    # Get a baseline context (average values for all features)
    base_context = df_features[causal_engine.feature_cols].mean().to_dict()
    
    for feature in features_to_plot_pdp:
        if feature not in df_features.columns:
            print(f"Warning: Feature '{feature}' not found for PDP.")
            continue
            
        # Create a grid of values for the feature we are analyzing
        feature_grid = np.linspace(df_features[feature].min(), df_features[feature].max(), 30)
        effects = []
        
        for value in feature_grid:
            temp_context = base_context.copy()
            temp_context[feature] = value
            effect = causal_engine.estimate_causal_effect(ACTION_TO_EXPLAIN, temp_context)
            effects.append(effect)
            
        plt.figure(figsize=(10, 6))
        plt.plot(feature_grid, effects, marker='o', linestyle='-')
        plt.title(f'Partial Dependence of a BUY Action on {feature}', fontsize=16)
        plt.xlabel(f'{feature} Value')
        plt.ylabel(f'Estimated Causal Effect of a BUY')
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.grid(True, linestyle='--', alpha=0.6)
        output_path = os.path.join(OUTPUT_DIR, f'phase2_test1_pdp_{feature}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved PDP plot for '{feature}'")

    # --- Test 2.2: SHAP Analysis for Specific Scenarios ---
    print("\nRunning Test 2.2: SHAP Analysis for Specific Decision Scenarios...")
    
    # SHAP needs a wrapper function that takes a NumPy array and returns predictions
    def prediction_wrapper(data_as_numpy):
        df_for_prediction = pd.DataFrame(data_as_numpy, columns=causal_engine.feature_cols)
        predictions = []
        for i, row in df_for_prediction.iterrows():
            context = row.to_dict()
            effect = causal_engine.estimate_causal_effect(ACTION_TO_EXPLAIN, context)
            predictions.append(effect)
        return np.array(predictions)

    # We use a sample of the data as a background for the SHAP explainer
    background_data = df_features[causal_engine.feature_cols].sample(50)
    explainer = shap.KernelExplainer(prediction_wrapper, background_data)

    # Let's pick some interesting scenarios from the data to explain
    # Scenario 1: A high-momentum, bullish case from our Phase 1 analysis
    bullish_scenario = df_features[
        (df_features['Rsi_14'] > 65) &
        (df_features['Roc_5'] > df_features['Roc_5'].quantile(0.9))
    ].sample(1)

    # Scenario 2: An ambiguous, conflicting case
    ambiguous_scenario = df_features[
        (df_features['Rsi_14'] > 65) & # High RSI (bearish for mean-reversion)
        (df_features['Macdh_12_26_9'] > df_features['Macdh_12_26_9'].quantile(0.8)) # Strong momentum (bullish)
    ].sample(1)

    scenarios_to_explain = {
        "bullish_scenario": bullish_scenario[causal_engine.feature_cols],
        "ambiguous_scenario": ambiguous_scenario[causal_engine.feature_cols]
    }
    
    for name, scenario_df in scenarios_to_explain.items():
        if not scenario_df.empty:
            shap_values = explainer.shap_values(scenario_df.iloc[0])
            
            # Create and save a force plot (interactive HTML)
            force_plot = shap.force_plot(explainer.expected_value, shap_values, scenario_df.iloc[0])
            output_path_html = os.path.join(OUTPUT_DIR, f'phase2_test2_shap_force_{name}.html')
            shap.save_html(output_path_html, force_plot)
            print(f"✅ Saved SHAP force plot (HTML) for '{name}'")
            
            # Create and save a waterfall plot (static PNG)
            plt.figure()
            shap.plots.waterfall(shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=scenario_df.iloc[0]), max_display=10, show=False)
            output_path_png = os.path.join(OUTPUT_DIR, f'phase2_test2_shap_waterfall_{name}.png')
            plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved SHAP waterfall plot (PNG) for '{name}'")

    print("\n--- Phase 2 Testing Complete ---")