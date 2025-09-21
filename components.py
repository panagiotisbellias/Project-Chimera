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
import shap
from tqdm import tqdm

# --- New Import ---
from environments import BaseEnvironment, BaseMultiAgentEnvironment


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

class EcommerceSimulatorV5(BaseEnvironment): # MODIFIED: Inherits from BaseEnvironment
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
        seasonality_amp: float = 0.2,
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
                "season_phase": 0,
            }
        # Initial calculation
        self._update_sales_and_profit()

    def reset(self, seed: Optional[int] = None, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resets the environment to its initial state."""
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

    def get_state(self) -> Dict[str, Any]:
        """Returns a copy of the current environment state."""
        return self.state.copy()

    def get_state_string(self) -> str:
        return json.dumps(self.state, indent=2)

    def _seasonality_factor(self) -> float:
        amp = self.params["seasonality_amp"]
        if amp <= 0.0:
            return 1.0
        phase = self.state.get("season_phase", 0) % 52
        s = 1.0 + amp * np.sin(2 * np.pi * phase / 52.0)
        return max(0.75, s)

    def _update_sales_and_profit(self):
        p = self.params
        weekly_ad = np.clip(self.state["weekly_ad_spend"], p["ad_min"], p["ad_max"])
        ad_multiplier = 1.0 + np.log1p(weekly_ad / 1000.0) * p["ad_log_scale"]

        price = float(np.clip(self.state["price"], p["price_lower"], p["price_upper"]))
        price_factor = (max(1e-6, price) / 100.0) ** (-p["price_elasticity"])
        
        season_factor = self._seasonality_factor()
        noise = self.rng.normal(1.0, p["noise_sigma"])

        raw_demand = p["base_demand"] * self.state["brand_trust"] * ad_multiplier * price_factor * season_factor
        demand = max(0.0, raw_demand * max(0.8, min(1.2, noise)))
        self.state["sales_volume"] = int(demand)

        revenue = self.state["sales_volume"] * price
        cogs = self.state["sales_volume"] * COST_PER_ITEM
        op_cost = weekly_ad
        profit = revenue - (cogs + op_cost)
        self.state["profit"] = float(profit)

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]: # MODIFIED: Renamed from take_action
        """Applies an action and advances the simulation by one week."""
        p = self.params
        price_change = float(action.get("price_change", 0.0) or 0.0)
        ad_spend = float(action.get("ad_spend", 0.0) or 0.0)
        ad_spend = float(np.clip(ad_spend, p["ad_min"], p["ad_max"]))

        if price_change > 0.10:
            self.state["brand_trust"] *= (1.0 - 0.02)  
        elif price_change < -0.05:
            self.state["brand_trust"] *= (1.0 + 0.03)  

        new_price = self.state["price"] * (1.0 + price_change)
        new_price = float(np.clip(new_price, p["price_lower"], p["price_upper"]))
        self.state["price"] = new_price
        self.state["weekly_ad_spend"] = ad_spend

        if ad_spend > 0:
            self.state["brand_trust"] *= (1.0 + np.log1p(ad_spend / 1000.0) * p["trust_ad_gain"])

        self.state["brand_trust"] *= (1.0 - p["trust_decay"])
        self.state["brand_trust"] = float(np.clip(self.state["brand_trust"], 0.2, 1.0))

        self.state["week"] += 1
        self.state["season_phase"] = (self.state.get("season_phase", 0) + 1) % 52

        self._update_sales_and_profit()
        return self.state.copy()


# ----------------------------
# SymbolicGuardianV4
# (No changes needed here)
# ----------------------------
class SymbolicGuardianV4:
    # ... (code for SymbolicGuardianV4 remains unchanged)
    def __init__(
        self,
        max_discount_per_week: float = 0.40,      
        max_price_increase_per_week: float = 0.50, 
        min_profit_margin_percentage: float = 0.15,
        max_price: float = 150.0,
        unit_cost: float = COST_PER_ITEM,
        ad_absolute_cap: float = 5000.0,         
        ad_increase_cap: float = 1000.0,          
        safety_buffer_ratio: float = 0.01,        
        safety_buffer_abs: float = 0.0            
    ):
        self.cfg = dict(
            max_dn=max_discount_per_week,
            max_up=max_price_increase_per_week,
            min_margin=min_profit_margin_percentage,
            max_price=max_price,
            unit_cost=unit_cost,
            ad_cap=ad_absolute_cap,
            ad_increase_cap=ad_increase_cap,
            safety_buffer_ratio=safety_buffer_ratio,
            safety_buffer_abs=safety_buffer_abs
        )
    # ... rest of the class is unchanged
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
        min_safe_price = c["unit_cost"] / (1.0 - c["min_margin"])
        # NEW in V4: Apply safety buffer to validation threshold
        min_safe_price_with_buffer = min_safe_price * (1.0 + c["safety_buffer_ratio"]) + c["safety_buffer_abs"]

        if future_price <= c["unit_cost"] or future_price < min_safe_price_with_buffer or margin < c["min_margin"]:
            return {
                "is_valid": False,
                "message": (
                    f"Rule Violation: Profit margin < {c['min_margin']*100:.0f}% "
                    f"or price below safe threshold (${min_safe_price_with_buffer:.2f} with buffer)."
                ),
            }

        return {"is_valid": True, "message": "Action is valid and compliant with all rules."}

    def repair_action(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

        # Price change percentage
        a["price_change"] = max(-c["max_dn"], min(c["max_up"], a["price_change"]))

        # Future price ceiling and minimum margin requirement
        base_price = float(current_state["price"])
        future_price = base_price * (1.0 + a["price_change"])

        # Max price 
        if future_price > c["max_price"]:
            a["price_change"] = (c["max_price"] / base_price) - 1.0
            future_price = c["max_price"]

        # Min margin with safety buffer
        margin = (future_price - c["unit_cost"]) / max(1e-6, future_price)
        min_safe_price = c["unit_cost"] / (1.0 - c["min_margin"])
        min_safe_price_with_buffer = min_safe_price * (1.0 + c["safety_buffer_ratio"]) + c["safety_buffer_abs"]

        if future_price <= c["unit_cost"] or future_price < min_safe_price_with_buffer or margin < c["min_margin"]:
            target_price = min_safe_price_with_buffer
            a["price_change"] = (target_price / base_price) - 1.0

        report = self.validate_action(a, current_state)
        return a, report


DEFAULT_TRUST_VALUE_MULTIPLIER = 100_000 

# ----------------------------
# CausalEngineV6
# ----------------------------
class CausalEngineV6:

    def __init__(
        self,
        data_path: str = "initial_causal_data.pkl",
        force_regenerate: bool = False,
        generator_seed: int = 123,
        trust_multiplier: float = DEFAULT_TRUST_VALUE_MULTIPLIER,
        num_simulations: int = 500,
    ):
        self.data_path = data_path
        self.model = None
        self.trust_multiplier = trust_multiplier
        self.initial_train_history: Optional[pd.DataFrame] = None
        self.rng = np.random.default_rng(generator_seed)

        if os.path.exists(self.data_path) and not force_regenerate:
            print("-> Causal Engine V6: Loading existing startup data...")
            self._load_data()
        else:
            print("-> Causal Engine V6: Generating new initial data (may take a few minutes)...")
            guardian = SymbolicGuardianV4()
            sim = EcommerceSimulatorV5(seed=42) 
            self.initial_train_history = self._generate_initial_data(sim, guardian, num_simulations)
            self._save_data()

        print("   - Data is ready. Causal Forest model is being trained...")
        self._fit_model(self.initial_train_history)
        print("   - Causal Engine V6 initialization and training completed.")

    def _generate_initial_data(
        self,
        simulator: EcommerceSimulatorV5,
        guardian: SymbolicGuardianV4,
        num_simulations: int = 500,
        num_steps: int = 50,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        print("-> Causal Engine V6: Generating new, truly randomized initial data...")
        
        abs_max_discount = guardian.cfg['max_dn']
        abs_max_increase = guardian.cfg['max_up']
        abs_ad_cap = guardian.cfg['ad_cap']

        for i in tqdm(range(num_simulations), desc="Generating Initial Training Data"):
            simulator.reset(seed=i)
            for _ in range(num_steps):
                state_before = simulator.state.copy()

                price_change = float(self.rng.uniform(-abs_max_discount, abs_max_increase))
                ad_spend = float(self.rng.uniform(0, abs_ad_cap))

                action_for_sim = {
                    "price_change": price_change,
                    "ad_spend": ad_spend
                }
                
                # NOTE: This internal call uses the original name, which is fine as it's a temporary instance.
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
        Y = training_data["trust_adjusted_profit_change"].values
        T = training_data[["price_change", "ad_spend"]].values
        XW = training_data[["initial_price", "initial_brand_trust", "initial_ad_spend", "season_phase"]].values

        self.model = CausalForestDML(
            model_y=GradientBoostingRegressor(random_state=123),
            model_t=MultiOutputRegressor(GradientBoostingRegressor(random_state=123)),
            discrete_treatment=False,
            random_state=123,
        )
        self.model.fit(Y, T, X=XW, W=XW)

    def retrain(self, new_experience_history: List[Dict[str, Any]]):
        new_data = pd.DataFrame(new_experience_history)

        if not new_data.empty and "profit_change" in new_data.columns and "trust_change" in new_data.columns:
            new_data["trust_adjusted_profit_change"] = new_data["profit_change"] + (new_data["trust_change"] * self.trust_multiplier)

        combined_history = pd.concat([self.initial_train_history, new_data], ignore_index=True)
        print(f"-> Causal Engine is being retrained. Total data points: {len(combined_history)}")
        self._fit_model(combined_history)
        self.initial_train_history = combined_history
        print("   - Retraining completed.")

    def estimate_causal_effect(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Causal model is not trained.")

        X_context = pd.DataFrame([{
            "initial_price": float(context["price"]),
            "initial_brand_trust": float(context["brand_trust"]),
            "initial_ad_spend": float(context["weekly_ad_spend"]), 
            "season_phase": int(context.get("season_phase", 0)),
        }])

        T0 = np.array([[0.0, 0.0]])
        T1 = np.array([[float(action.get("price_change", 0.0) or 0.0),
                        float(action.get("ad_spend", 0.0) or 0.0)]])

        effect = self.model.effect(X_context, T0=T0, T1=T1)
        
        return {"estimated_long_term_value": float(effect[0])}

    def simulate_plan(
        self,
        simulator: EcommerceSimulatorV5,
        guardian: SymbolicGuardianV4,
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
                # NOTE: This internal call uses the original name, which is fine as it's a temporary instance.
                after = sim.step(safe_action)
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
    

    def explain_decision(self, context: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Causal model is not trained.")

        feature_names = ["initial_price", "initial_brand_trust", "initial_ad_spend", "season_phase", "price_change", "ad_spend"]

        instance_data = {
            "initial_price": context.get("price", 0.0),
            "initial_brand_trust": context.get("brand_trust", 0.0),
            "initial_ad_spend": context.get("weekly_ad_spend", 0.0),
            "season_phase": int(context.get("season_phase", 0)),
            "price_change": action.get("price_change", 0.0),
            "ad_spend": action.get("ad_spend", 0.0)
        }
        instance_df = pd.DataFrame([instance_data], columns=feature_names)
        instance_numpy = instance_df.values

        def prediction_wrapper(data_numpy):
            X_context = data_numpy[:, :4]
            T1_action = data_numpy[:, 4:]
            T0_noop = np.zeros_like(T1_action)
            return self.model.effect(X_context, T0=T0_noop, T1=T1_action)

        background_data_sample = self.initial_train_history.sample(min(100, len(self.initial_train_history)))
        
        background_data_numpy = background_data_sample[feature_names].values

        explainer = shap.Explainer(prediction_wrapper, background_data_numpy)
        
        shap_values_obj = explainer(instance_numpy)

        explanation = {
            "features": feature_names,
            "shap_values": shap_values_obj.values[0].tolist(),
            "base_value": shap_values_obj.base_values[0]
        }
        
        return explanation
    
# ----------------------------
# EcommerceSimulatorV7
# ----------------------------

class EcommerceSimulatorV7(BaseMultiAgentEnvironment): # MODIFIED: Inherits from BaseMultiAgentEnvironment
    """
    A multi-agent competitive market simulator.
    """
    def __init__(
        self,
        num_agents: int = 3,
        seed: Optional[int] = None,
        base_demand: float = 1000.0,
        price_elasticity: float = 1.2,
        ad_attraction_factor: float = 0.5,
        ad_log_scale: float = 0.3,
        trust_ad_gain: float = 0.015,
        trust_decay: float = 0.002,
        price_upper: float = 150.0,
        price_lower: float = COST_PER_ITEM,
        ad_min: float = 0.0,
        ad_max: float = 5_000.0,
        seasonality_amp: float = 0.2,
        noise_sigma: float = 0.05,
    ):
        self.num_agents = num_agents
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.params = {
            'base_demand': base_demand, 'price_elasticity': price_elasticity,
            'ad_attraction_factor': ad_attraction_factor,
            'ad_log_scale': ad_log_scale, 'trust_ad_gain': trust_ad_gain,
            'trust_decay': trust_decay, 'price_upper': price_upper,
            'price_lower': price_lower, 'ad_min': ad_min, 'ad_max': ad_max,
            'seasonality_amp': seasonality_amp, 'noise_sigma': noise_sigma,
        }
        self.week = 1
        self.season_phase = 0
        self.agent_states = self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Resets the simulation for all agents to their initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.week = 1
        self.season_phase = 0
        self.agent_states = []
        
        default_initial_state = {
            "price": 100.0,
            "weekly_ad_spend": 500.0,
            "brand_trust": 0.7,
            "sales_volume": 0,
            "profit": 0.0,
            "market_share": 1.0 / self.num_agents,
        }

        for i in range(self.num_agents):
            state = default_initial_state.copy()
            state['agent_id'] = i
            self.agent_states.append(state)
            
        return self.agent_states.copy()

    def get_state(self) -> Dict[str, Any]: # MODIFIED: Renamed from get_full_state
        """Returns the complete state of the simulation."""
        return {
            "week": self.week,
            "season_phase": self.season_phase,
            "agents": self.agent_states.copy()
        }

    def _apply_actions(self, actions: List[Dict[str, Any]]):
        """Applies individual agent actions to update their internal states."""
        p = self.params
        for i, action in enumerate(actions):
            state = self.agent_states[i]
            
            price_change = float(action.get("price_change", 0.0) or 0.0)
            new_price = state["price"] * (1.0 + price_change)
            state["price"] = float(np.clip(new_price, p['price_lower'], p['price_upper']))

            state["weekly_ad_spend"] = float(np.clip(action.get("ad_spend", 0.0), p['ad_min'], p['ad_max']))

            if price_change > 0.10:
                state["brand_trust"] *= (1.0 - 0.02)
            elif price_change < -0.05:
                state["brand_trust"] *= (1.0 + 0.03)
            
            if state["weekly_ad_spend"] > 0:
                state["brand_trust"] *= (1.0 + np.log1p(state["weekly_ad_spend"] / 1000.0) * p['trust_ad_gain'])
            
            state["brand_trust"] *= (1.0 - p['trust_decay'])
            state["brand_trust"] = float(np.clip(state["brand_trust"], 0.2, 1.0))

    def _update_market_and_sales(self):
        """Calculates total demand and distributes it based on market share."""
        p = self.params
        
        total_market_ad_spend = sum(s['weekly_ad_spend'] for s in self.agent_states)
        avg_market_price = sum(s['price'] for s in self.agent_states) / self.num_agents
        
        season_factor = 1.0 + p['seasonality_amp'] * np.sin(2 * np.pi * self.season_phase / 52.0)
        ad_multiplier = 1.0 + np.log1p(total_market_ad_spend / 1000.0) * p['ad_log_scale']
        price_factor = (avg_market_price / 100.0) ** (-p['price_elasticity'])
        noise = self.rng.normal(1.0, p['noise_sigma'])
        
        total_demand = int(p['base_demand'] * ad_multiplier * price_factor * season_factor * noise)
        
        attractiveness_scores = []
        for state in self.agent_states:
            price_attraction = state['price'] ** -p['price_elasticity']
            ad_attraction = 1 + np.log1p(state['weekly_ad_spend'] / 1000.0) * p['ad_attraction_factor']
            score = state['brand_trust'] * price_attraction * ad_attraction
            attractiveness_scores.append(score)
            
        total_attractiveness = sum(attractiveness_scores)
        if total_attractiveness == 0: total_attractiveness = 1.0

        for i, state in enumerate(self.agent_states):
            state['market_share'] = attractiveness_scores[i] / total_attractiveness
            state['sales_volume'] = int(total_demand * state['market_share'])
            revenue = state['sales_volume'] * state['price']
            cogs = state['sales_volume'] * COST_PER_ITEM
            op_cost = state['weekly_ad_spend']
            state['profit'] = revenue - (cogs + op_cost)

    def step(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]: # MODIFIED: Renamed from take_action
        """Takes a list of actions and resolves the turn for the entire market."""
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, but received {len(actions)}.")
            
        self._apply_actions(actions)
        self._update_market_and_sales()
        
        self.week += 1
        self.season_phase = (self.season_phase + 1) % 52
        
        return self.agent_states.copy()