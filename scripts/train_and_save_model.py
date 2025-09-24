# train_and_save_model.py

import pandas as pd
import numpy as np
import pickle
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components import EcommerceSimulatorV5, SymbolicGuardianV4

def generate_synthetic_data(num_simulations=2000, num_steps=50):
    """
    Generates synthetic data for a realistic and 'robust' model training.
    This data simulates the complex relationships in an e-commerce environment.
    """
    rows = []
    print(f"Generating synthetic data via {num_simulations} simulations...")

    simulator = EcommerceSimulatorV5(seed=42)
    guardian = SymbolicGuardianV4()
    rng = np.random.default_rng(123)

    abs_max_discount = guardian.cfg['max_dn']
    abs_max_increase = guardian.cfg['max_up']
    abs_ad_cap = guardian.cfg['ad_cap']

    for i in tqdm(range(num_simulations), desc="Running Simulations"):
        simulator.reset(seed=i)
        for _ in range(num_steps):
            state_before = simulator.get_state()

            price_change = float(rng.uniform(-abs_max_discount, abs_max_increase))
            ad_spend = float(rng.uniform(0, abs_ad_cap))
            
            action_for_sim = {
                "price_change": price_change,
                "ad_spend": ad_spend
            }
            
            state_after = simulator.step(action_for_sim)
            profit_change = float(state_after["profit"] - state_before["profit"])
            trust_change = float(state_after["brand_trust"] - state_before["brand_trust"])

            rows.append({
                "initial_price": float(state_before["price"]),
                "initial_brand_trust": float(state_before["brand_trust"]),
                "initial_ad_spend": float(state_before["weekly_ad_spend"]),
                "season_phase": int(state_before["season_phase"]),
                "price_change": float(action_for_sim["price_change"]),
                "ad_spend": float(action_for_sim["ad_spend"]),
                "profit_change": profit_change,
                "trust_change": trust_change,
            })
    
    print("Data generation complete.")
    return pd.DataFrame(rows)

# 1. Generate Data
synthetic_data = generate_synthetic_data()

# 2. Define and Train the Model
print("Defining the Causal Forest model...")
causal_model = CausalForestDML(
    model_y=GradientBoostingRegressor(random_state=123),
    model_t=MultiOutputRegressor(GradientBoostingRegressor(random_state=123)),
    discrete_treatment=False,
    random_state=123
)

outcome = synthetic_data['profit_change'] 
treatments = synthetic_data[['price_change', 'ad_spend']]
features = synthetic_data[['initial_price', 'initial_brand_trust', 'initial_ad_spend', 'season_phase']]

print("Model training is starting... (This might take a few minutes)")
causal_model.fit(Y=outcome, T=treatments, X=features, W=features)
print("Model training completed successfully.")

# 3. Save the Trained Model
model_filename = 'trained_causal_model.pkl'
print(f"Saving the trained model as '{model_filename}'...")
with open(model_filename, 'wb') as f:
    pickle.dump(causal_model, f)

print(f"\nSave successful! '{model_filename}' has been created in your project directory.")
print("You can now proceed to update components.py and run the main Streamlit app.")