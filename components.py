# components.py

import os
import json
import pickle
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


COST_PER_ITEM = 50.0

# ----------------------------
# Data types
# ----------------------------

@dataclass
class Action:
    price_change: float = 0.0
    ad_spend: float = 0.0


# ----------------------------
# EcommerceSimulatorV5
# ----------------------------

class EcommerceSimulatorV5:
    """
    A simulator with more realistic and stable economic principles.
        - Weekly advertising budget spending (not cumulative).
        - Numpy Generator-based seed management for repeatability.
        - Brand trust erosion and price/name effects.
        - Simple seasonality (optional).
    """

    def __init__(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        base_demand: float = 800.0,
        price_elasticity: float = 1.2,
        ad_log_scale: float = 0.3,
        trust_ad_gain: float = 0.01,
        trust_decay: float = 0.002,
        price_penalty_threshold: float = 150.0,
        price_upper: float = 150.0,
        price_lower: float = COST_PER_ITEM,
        ad_min: float = 0.0,
        ad_max: float = 5_000.0,
        seasonality_amp: float = 0.2,  # 0.0 is unavailable
        noise_sigma: float = 0.05,
    ):
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.params = dict(
            base_demand=base_demand,
            price_elasticity=price_elasticity,
            ad_log_scale=ad_log_scale,
            trust_ad_gain=trust_ad_gain,
            trust_decay=trust_decay,
            price_penalty_threshold=price_penalty_threshold,
            price_upper=price_upper,
            price_lower=price_lower,
            ad_min=ad_min,
            ad_max=ad_max,
            seasonality_amp=seasonality_amp,
            noise_sigma=noise_sigma,
        )

        if initial_state:
            self.state = initial_state.copy()
        else:
            self.state = {
                "week": 1,
                "price": 100.0,
                "weekly_ad_spend": 500.0,
                "brand_trust": 0.7,
                "sales_volume": 0,
                "profit": 0.0,
                "season_phase": 0,  # 0..51
            }
        # İlk hesap
        self._update_sales_and_profit()

    def reset(self, seed: Optional[int] = None, initial_state: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if initial_state:
            self.state = initial_state.copy()
        else:
            self.state = {
                "week": 1,
                "price": 100.0,
                "weekly_ad_spend": 500.0,
                "brand_trust": 0.7,
                "sales_volume": 0,
                "profit": 0.0,
                "season_phase": 0,
            }
        self._update_sales_and_profit()
        return self.state.copy()

    def get_state_string(self) -> str:
        return json.dumps(self.state, indent=2)

    def _seasonality_factor(self) -> float:
        amp = self.params["seasonality_amp"]
        if amp <= 0.0:
            return 1.0
        # Weekly sinus effect for seasonality
        phase = self.state.get("season_phase", 0) % 52
        s = 1.0 + amp * np.sin(2 * np.pi * phase / 52.0)
        return max(0.75, s)  # limiting effect

    def _update_sales_and_profit(self):
        p = self.params
        # Ad spend effect
        weekly_ad = np.clip(self.state["weekly_ad_spend"], p["ad_min"], p["ad_max"])
        ad_multiplier = 1.0 + np.log1p(weekly_ad / 1000.0) * p["ad_log_scale"]

        # Price factor
        price = float(np.clip(self.state["price"], p["price_lower"], p["price_upper"]))
        price_factor = (max(1e-6, price) / 100.0) ** (-p["price_elasticity"])

        # Season factor
        season_factor = self._seasonality_factor()

        # Noise (mild, multiplicative)
        noise = self.rng.normal(1.0, p["noise_sigma"])

        raw_demand = p["base_demand"] * self.state["brand_trust"] * ad_multiplier * price_factor * season_factor
        demand = max(0.0, raw_demand * max(0.8, min(1.2, noise)))  # sert sınırlar
        self.state["sales_volume"] = int(demand)

        # Revenue and costs
        revenue = self.state["sales_volume"] * price
        cogs = self.state["sales_volume"] * COST_PER_ITEM
        op_cost = weekly_ad
        profit = revenue - (cogs + op_cost)
        self.state["profit"] = float(profit)

    def take_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        p = self.params
        price_change = float(action.get("price_change", 0.0) or 0.0)
        ad_spend = float(action.get("ad_spend", 0.0) or 0.0)
        ad_spend = float(np.clip(ad_spend, p["ad_min"], p["ad_max"]))

        # Price change effect
        if price_change > 0.10:
            self.state["brand_trust"] *= (1.0 - 0.02)  
        elif price_change < -0.05:
            self.state["brand_trust"] *= (1.0 + 0.03)  

        # Price update
        new_price = self.state["price"] * (1.0 + price_change)
        new_price = float(np.clip(new_price, p["price_lower"], p["price_upper"]))
        self.state["price"] = new_price

        # Ad spend updates
        self.state["weekly_ad_spend"] = ad_spend

        # The effect of advertising on brand trust (log scale)
        if ad_spend > 0:
            self.state["brand_trust"] *= (1.0 + np.log1p(ad_spend / 1000.0) * p["trust_ad_gain"])

        # General wear
        self.state["brand_trust"] *= (1.0 - p["trust_decay"])

        # Safety limits
        self.state["brand_trust"] = float(np.clip(self.state["brand_trust"], 0.2, 1.0))

        # Advance week/season phase
        self.state["week"] += 1
        self.state["season_phase"] = (self.state.get("season_phase", 0) + 1) % 52

        # Update accounts
        self._update_sales_and_profit()
        return self.state.copy()


# ----------------------------
# SymbolicGuardianV3
# ----------------------------

class SymbolicGuardianV3:
    """
    Symmetric rules and automatic repair logic.
        - Price increases/discounts are limited by percentage.
        - Minimum profit margin and maximum price are maintained.
        - Weekly ad spend: absolute cap + previous week's increase cap.
    """

    def __init__(
        self,
        max_discount_per_week: float = 0.40,       # %No discounts over 40
        max_price_increase_per_week: float = 0.50, # No more than 50% increase
        min_profit_margin_percentage: float = 0.15,# minimum 15% margin
        max_price: float = 150.0,
        unit_cost: float = COST_PER_ITEM,
        ad_absolute_cap: float = 5000.0,           # weekly absolute ceiling
        ad_increase_cap: float = 1000.0,           # maximum increase compared to the previous week
    ):
        self.cfg = dict(
            max_dn=max_discount_per_week,
            max_up=max_price_increase_per_week,
            min_margin=min_profit_margin_percentage,
            max_price=max_price,
            unit_cost=unit_cost,
            ad_cap=ad_absolute_cap,
            ad_increase_cap=ad_increase_cap,
        )

    def validate_action(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        c = self.cfg
        price_change = float(action.get("price_change", 0.0) or 0.0)
        ad_spend = float(action.get("ad_spend", 0.0) or 0.0)
        prev_ad = float(current_state.get("weekly_ad_spend", 0.0) or 0.0)

        # Negative ad spend is prohibited
        if ad_spend < 0:
            return {"is_valid": False, "message": "Rule Violation: Ad spend cannot be negative."}

        # Absolute cap
        if ad_spend > c["ad_cap"]:
            return {
                "is_valid": False,
                "message": f"Rule Violation: Weekly ad spend cannot exceed ${c['ad_cap']}.",
            }

        # Relative increase cap
        if ad_spend - prev_ad > c["ad_increase_cap"]:
            return {
                "is_valid": False,
                "message": f"Rule Violation: Weekly ad spend increase cannot exceed ${c['ad_increase_cap']} vs last week.",
            }

        # Symmetric price percentage
        if price_change < 0 and abs(price_change) > c["max_dn"]:
            return {"is_valid": False, "message": f"Rule Violation: Weekly discount cannot exceed {c['max_dn']*100:.0f}%."}
        if price_change > 0 and price_change > c["max_up"]:
            return {"is_valid": False, "message": f"Rule Violation: Weekly price increase cannot exceed {c['max_up']*100:.0f}%."}

        # Future price and margin
        future_price = float(current_state["price"]) * (1.0 + price_change)
        if future_price > c["max_price"]:
            return {"is_valid": False, "message": f"Rule Violation: Price cannot exceed ${c['max_price']}."}

        margin = (future_price - c["unit_cost"]) / max(1e-6, future_price)
        if future_price <= c["unit_cost"] or margin < c["min_margin"]:
            return {
                "is_valid": False,
                "message": f"Rule Violation: Profit margin < {c['min_margin']*100:.0f}% or price below cost (${c['unit_cost']}).",
            }

        return {"is_valid": True, "message": "Action is valid and compliant with all rules."}

    def repair_action(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        In case of violation of the rules, the action is made valid by correcting it as much as possible.
        """
        c = self.cfg
        a = dict(
            price_change=float(action.get("price_change", 0.0) or 0.0),
            ad_spend=float(action.get("ad_spend", 0.0) or 0.0),
        )
        prev_ad = float(current_state.get("weekly_ad_spend", 0.0) or 0.0)

        # Ad spend: negative → 0; absolute cap; relative cap
        a["ad_spend"] = max(0.0, a["ad_spend"])
        a["ad_spend"] = min(c["ad_cap"], a["ad_spend"])
        if a["ad_spend"] - prev_ad > c["ad_increase_cap"]:
            a["ad_spend"] = prev_ad + c["ad_increase_cap"]

        # 2) Price change percentage
        a["price_change"] = max(-c["max_dn"], min(c["max_up"], a["price_change"]))

        # 3) Future price ceiling and minimum margin requirement
        base_price = float(current_state["price"])
        future_price = base_price * (1.0 + a["price_change"])

        # Max price 
        if future_price > c["max_price"]:
            a["price_change"] = (c["max_price"] / base_price) - 1.0
            future_price = c["max_price"]

        # Min marj 
        margin = (future_price - c["unit_cost"]) / max(1e-6, future_price)
        if future_price <= c["unit_cost"] or margin < c["min_margin"]:
            target_price = c["unit_cost"] / (1.0 - c["min_margin"])
            a["price_change"] = (target_price / base_price) - 1.0

        report = self.validate_action(a, current_state)
        return a, report


# ----------------------------
# CausalEngineV5
# ----------------------------

DEFAULT_TRUST_VALUE_MULTIPLIER = 100_000 

class CausalEngineV5:
    """
    The final version of the causal engine has been restructured.
        - Treatment = [price_change, ad_spend].
        - Outcome = Trust-Adjusted Profit Change (Dynamically adjustable long-term value).
        - Context = initial_price, initial_brand_trust, initial_ad_spend.
    """

    def __init__(
        self,
        data_path: str = "initial_causal_data.pkl",
        force_regenerate: bool = False,
        generator_seed: int = 123,
        # UPDATE: Parameter to import the multiplier and set a default value
        trust_multiplier: float = DEFAULT_TRUST_VALUE_MULTIPLIER,
    ):
        self.data_path = data_path
        self.model = None
        # UPDATE: We are storing the multiplier as a class property
        self.trust_multiplier = trust_multiplier
        self.initial_train_history: Optional[pd.DataFrame] = None
        self.rng = np.random.default_rng(generator_seed)

        if os.path.exists(self.data_path) and not force_regenerate:
            print("-> Causal Engine V5: Loading existing startup data...")
            self._load_data()
        else:
            print("-> Causal Engine V5: Generating new initial data (may take a few minutes)...")
            guardian = SymbolicGuardianV3()
            sim = EcommerceSimulatorV5(seed=42)
            self.initial_train_history = self._generate_initial_data(sim, guardian)
            self._save_data()

        print("   - Data is ready. Causal Forest model is being trained...")
        self._fit_model(self.initial_train_history)
        print("   - Causal Engine V5 initialization and training completed.")

    def _generate_initial_data(
        self,
        simulator: EcommerceSimulatorV5,
        guardian: SymbolicGuardianV3,
        num_simulations: int = 120,
        num_steps: int = 60,
        price_change_range: Tuple[float, float] = (-0.30, 0.40),
        ad_spend_range: Tuple[float, float] = (0.0, 2000.0),
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for i in range(num_simulations):
            simulator.reset(seed=i)
            for _ in range(num_steps):
                state_before = simulator.state.copy()
                raw_action = {
                    "price_change": float(self.rng.uniform(*price_change_range)),
                    "ad_spend": float(self.rng.uniform(*ad_spend_range)),
                }
                safe_action, _ = guardian.repair_action(raw_action, state_before)
                simulator.take_action(safe_action)
                state_after = simulator.state.copy()

                profit_change = float(state_after["profit"] - state_before["profit"])
                trust_change = float(state_after["brand_trust"] - state_before["brand_trust"])

                rows.append({
                    #UPDATE: Added ad spend to provide better context to the model
                    "initial_price": float(state_before["price"]),
                    "initial_brand_trust": float(state_before["brand_trust"]),
                    "initial_ad_spend": float(state_before["weekly_ad_spend"]),
                    "price_change": float(safe_action["price_change"]),
                    "ad_spend": float(safe_action["ad_spend"]),
                    # UPDATE: Target variable is calculated using dynamic multiplier
                    "trust_adjusted_profit_change": profit_change + (trust_change * self.trust_multiplier),
                    "profit_change": profit_change,
                    "trust_change": trust_change,
                    "sales_change": float(state_after["sales_volume"] - state_before["sales_volume"]),
                })
        return pd.DataFrame(rows)

    def _save_data(self):
        with open(self.data_path, "wb") as f:
            pickle.dump(self.initial_train_history, f)
        print(f"   - Initial data was saved to file '{self.data_path}'.")

    def _load_data(self):
        with open(self.data_path, "rb") as f:
            self.initial_train_history = pickle.load(f)
        print(f"   - Initial data was loaded from file '{self.data_path}'.")

    def _fit_model(self, training_data: pd.DataFrame):
        # UPDATE: We are using the new target variable (Y) and full context (XW)
        Y = training_data["trust_adjusted_profit_change"].values
        T = training_data[["price_change", "ad_spend"]].values
        XW = training_data[["initial_price", "initial_brand_trust", "initial_ad_spend"]].values

        self.model = CausalForestDML(
            model_y=GradientBoostingRegressor(random_state=123),
            model_t=MultiOutputRegressor(GradientBoostingRegressor(random_state=123)),
            discrete_treatment=False,
            random_state=123,
        )
        self.model.fit(Y, T, X=XW, W=XW)

    def retrain(self, new_experience_history: List[Dict[str, Any]]):
        new_data = pd.DataFrame(new_experience_history)

        # UPDATE: Calculate target variable with dynamic multiplier for new live data
        if not new_data.empty and "profit_change" in new_data.columns and "trust_change" in new_data.columns:
            new_data["trust_adjusted_profit_change"] = new_data["profit_change"] + (new_data["trust_change"] * self.trust_multiplier)

        combined_history = pd.concat([self.initial_train_history, new_data], ignore_index=True)
        print(f"-> Causal Engine is being retrained. Total data points: {len(combined_history)}")
        self._fit_model(combined_history)
        self.initial_train_history = combined_history
        print("   - Retraining completed.")

    def estimate_causal_effect(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
            In one step, the expected long-term value impact of the proposed action 
            versus 'doing nothing' (T0=[0,0]) in the given context.
        """
        if self.model is None:
            raise RuntimeError("Causal model is not trained.")

        # UPDATE: We use all three context features for prediction
        X_context = pd.DataFrame([{
            "initial_price": float(context["price"]),
            "initial_brand_trust": float(context["brand_trust"]),
            "initial_ad_spend": float(context["weekly_ad_spend"]), 
        }])

        T0 = np.array([[0.0, 0.0]])
        T1 = np.array([[float(action.get("price_change", 0.0) or 0.0),
                        float(action.get("ad_spend", 0.0) or 0.0)]])

        effect = self.model.effect(X_context, T0=T0, T1=T1)
        
        # UPDATE: We changed the output key to reflect the actual value it returns
        return {"estimated_long_term_value": float(effect[0])}

    def simulate_plan(
        self,
        simulator: EcommerceSimulatorV5,
        guardian: SymbolicGuardianV3,
        start_state: Optional[Dict[str, Any]],
        plan: List[Dict[str, Any]],
        n_rollouts: int = 64,
        seed: Optional[int] = 999,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        profits = []
        sales_list = []
        trust_list = []

        for r in range(n_rollouts):
            if start_state is not None:
                s0 = start_state.copy()
                sim = EcommerceSimulatorV5(initial_state=s0, seed=int(rng.integers(0, 1_000_000)))
            else:
                sim = EcommerceSimulatorV5(seed=int(rng.integers(0, 1_000_000)))

            profit_acc = 0.0
            sales_acc = 0.0
            trust_end = sim.state["brand_trust"]

            for step_action in plan:
                safe_action, _ = guardian.repair_action(step_action, sim.state)
                before = sim.state.copy()
                after = sim.take_action(safe_action)
                profit_acc += float(after["profit"] - before["profit"])
                sales_acc += float(after["sales_volume"])
                trust_end = float(after["brand_trust"])

            profits.append(profit_acc)
            sales_list.append(sales_acc)
            trust_list.append(trust_end)

        def _summary(arr: List[float]) -> Dict[str, float]:
            a = np.asarray(arr, dtype=float)
            return {
                "mean": float(np.mean(a)),
                "p05": float(np.percentile(a, 5)),
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
            }

        return {
            "profit": _summary(profits),
            "sales": _summary(sales_list),
            "final_trust": _summary(trust_list),
            "horizon": len(plan),
            "rollouts": n_rollouts,
        }