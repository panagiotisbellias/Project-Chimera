# components.py

import json
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor

# Central constant for cost per item
COST_PER_ITEM = 50.0

class EcommerceSimulatorV4:
    """A simulator with sophisticated economic principles, including direct ad impact."""
    def __init__(self, initial_state=None):
        if initial_state: self.state = initial_state
        else: self.state = { "week": 1, "price": 100.0, "ad_budget": 1000.0, "brand_trust": 0.7, "sales_volume": 0, "profit": 0.0 }
        self._update_sales_and_profit()

    def get_state_string(self): return json.dumps(self.state, indent=2)
    
    def _update_sales_and_profit(self):
        price_elasticity = 1.5; base_demand = 200000
        ad_multiplier = 1 + np.log1p(self.state["ad_budget"] / 2000) * 0.5
        self.state["sales_volume"] = int((base_demand / (self.state["price"] ** price_elasticity)) * self.state["brand_trust"] * ad_multiplier * np.random.normal(1.0, 0.05))
        self.state["profit"] = self.state["sales_volume"] * (self.state["price"] - COST_PER_ITEM) - self.state["ad_budget"]

    def take_action(self, action: dict):
        price_change = action.get("price_change", 0.0) or 0.0
        ad_spend = action.get("ad_spend", 0.0) or 0.0
        if price_change > 0.10: self.state["brand_trust"] *= 0.98
        elif price_change < -0.05: self.state["brand_trust"] *= 1.03
        self.state["price"] *= (1 + price_change)
        if ad_spend > 0:
            trust_increase_factor = 1 + np.log1p(ad_spend / 1000) * 0.01 
            self.state["brand_trust"] *= trust_increase_factor
        self.state["ad_budget"] += ad_spend
        self.state["price"] = np.clip(self.state["price"], COST_PER_ITEM, 150.0)
        self.state["brand_trust"] = np.clip(self.state["brand_trust"], 0.2, 1.0)
        self.state["week"] += 1
        self._update_sales_and_profit()
        return self.state.copy()

class SymbolicGuardianV2:
    """Validates actions against business rules."""
    def __init__(self): 
        self.rules = {"max_discount_per_week": 0.4, "min_profit_margin_percentage": 0.15, "max_ad_spend_increase_per_week": 500.0, "max_price": 150.0 }
    
    def validate_action(self, action: dict, current_state: dict):
        price_change = action.get("price_change", 0.0) or 0.0
        ad_spend = action.get("ad_spend", 0.0) or 0.0
        if price_change < 0 and abs(price_change) > self.rules["max_discount_per_week"]: return {"is_valid": False, "message": "Rule Violation: Weekly discount cannot exceed 40%."}
        if ad_spend > self.rules["max_ad_spend_increase_per_week"]: return {"is_valid": False, "message": "Rule Violation: Weekly ad spend increase cannot exceed 500."}
        future_price = current_state["price"] * (1 + price_change)
        if future_price > self.rules["max_price"]: return {"is_valid": False, "message": f"Rule Violation: Price cannot exceed the ${self.rules['max_price']} limit."}
        if future_price <= COST_PER_ITEM or (future_price - COST_PER_ITEM) / future_price < self.rules["min_profit_margin_percentage"]: 
            return {"is_valid": False, "message": f"Rule Violation: Profit margin cannot fall below 15% or price cannot be below cost (${COST_PER_ITEM})."}
        return {"is_valid": True, "message": "Action is valid and compliant with all rules."}


# components.py içindeki CausalEngineV4 class'ını güncelleyin

class CausalEngineV4:
    """Causal inference engine that can be retrained by combining old and new data."""
    def __init__(self):
        print("-> Causal Engine V4: Generating initial simulation data for training...")
        self.initial_train_history = []
        self._initial_training()

    def _initial_training(self):
        sim_v4 = EcommerceSimulatorV4()
        initial_history = []
        # !!! GÜNCELLEME: Başlangıçtaki "dünya bilgisi" veri setini 5 kat artırdık !!!
        # Bu, ajanın tek bir kötü tecrübeye aşırı uyum sağlamasını engelleyecek.
        for _ in range(10000): # Eskiden 2000'di
            random_action = {"price_change": np.random.uniform(-0.2, 0.2), "ad_spend": np.random.uniform(0, 500)}
            sim_v4.take_action(random_action)
            initial_history.append(sim_v4.state.copy())
        
        # Store this initial "world knowledge"
        self.initial_train_history = initial_history
        self._fit_model(self.initial_train_history)
        print("   - Causal Engine V4 initial training complete.")

    def _fit_model(self, training_data: list):
        """Fits the CausalForestDML model on the provided training data."""
        data = pd.DataFrame(training_data)
        self.model = CausalForestDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor(), discrete_treatment=False, random_state=123)
        Y = data['profit']; T = data['price']; XW = data[['ad_budget', 'brand_trust']]
        self.model.fit(Y, T, X=XW, W=XW)

    def retrain(self, new_experience_history: list):
        """
        Retrains the model by COMBINING the initial broad knowledge
        with the new, specific experiences. This prevents catastrophic forgetting.
        """
        # !!! CRITICAL FIX HERE: We combine, not replace !!!
        combined_history = self.initial_train_history + new_experience_history
        
        print(f"-> Retraining Causal Engine with {len(combined_history)} total data points ({len(self.initial_train_history)} initial + {len(new_experience_history)} new)...")
        self._fit_model(combined_history)
        print("   - Causal Engine retraining complete.")

    def estimate_causal_effect(self, action: dict, context: dict):
        current_price = context['price']
        price_change = action.get("price_change", 0.0) or 0.0
        X_context = pd.DataFrame([{"ad_budget": context['ad_budget'], "brand_trust": context['brand_trust']}])
        future_price = current_price * (1 + price_change)
        effect = self.model.effect(X_context, T0=current_price, T1=future_price)
        report = {"estimated_profit_change": f"{effect[0]:.2f}"}
        return report