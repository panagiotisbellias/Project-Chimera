# formal_verifier.py (Final Corrected Version)

from z3 import Solver, Real, If, And, Or, Not, unsat, sat
from components import SymbolicGuardianV3

def run_verification():
    """
    Main function to run all formal verification proofs for SymbolicGuardianV3.
    """
    print("--- Initializing Formal Verification for SymbolicGuardianV3 ---")
    guardian = SymbolicGuardianV3()
    cfg = guardian.cfg
    print(f"Loaded Guardian Config: {cfg}")

    # --- Symbolic Variables ---
    current_price = Real('current_price')
    current_ad_spend = Real('current_ad_spend')
    price_change_proposed = Real('price_change_proposed')
    ad_spend_proposed = Real('ad_spend_proposed')

    # --- Translate 'repair_action' logic to Z3 expressions ---
    ad_spend_after_non_negative = If(ad_spend_proposed < 0, 0.0, ad_spend_proposed)
    ad_spend_after_absolute_cap = If(ad_spend_after_non_negative > cfg['ad_cap'], cfg['ad_cap'], ad_spend_after_non_negative)
    ad_spend_repaired = If(ad_spend_after_absolute_cap - current_ad_spend > cfg['ad_increase_cap'],
                           current_ad_spend + cfg['ad_increase_cap'],
                           ad_spend_after_absolute_cap)

    price_change_clipped = If(price_change_proposed > cfg['max_up'], cfg['max_up'],
                              If(price_change_proposed < -cfg['max_dn'], -cfg['max_dn'], price_change_proposed))
    price_after_clip = current_price * (1 + price_change_clipped)
    price_after_max_cap = If(price_after_clip > cfg['max_price'], cfg['max_price'], price_after_clip)
    target_price = cfg['unit_cost'] / (1.0 - cfg['min_margin'])
    final_price = If(price_after_max_cap < target_price, target_price, price_after_max_cap)
    
    # --- Helper function for running proofs (FINAL CORRECTED VERSION) ---
    def prove_invariant(name: str, invariant_formula, preconditions=None):
        solver = Solver()
        
        if preconditions is not None:
            solver.add(preconditions)
        
        # CORRECTED LOGIC: Use Z3's Not() function instead of Python's 'not' operator
        rule_violation_formula = Not(invariant_formula)
        solver.add(rule_violation_formula)
        
        print(f"\nProving Invariant: '{name}'...")
        result = solver.check()
        
        if result == unsat:
            print(f"✅ SUCCESS (unsat): The invariant is PROVEN to be always true.")
            return True
        elif result == sat:
            print(f"❌ FAILURE (sat): A counter-example was found!")
            print("  - Counter-example:", solver.model())
            return False
        else:
            print(f"❔ UNKNOWN: The solver's result was: {result}")
            return False

    # --- Define and run all proofs ---
    all_proofs_passed = True
    
    preconditions = And(current_price > cfg['unit_cost'], current_ad_spend >= 0)

    # Proof 1: Ad Spend is never negative
    inv1 = ad_spend_repaired >= 0
    if not prove_invariant("Ad Spend Non-Negative", inv1, preconditions): all_proofs_passed = False

    # Proof 2: Ad Spend never exceeds absolute cap
    inv2 = ad_spend_repaired <= cfg['ad_cap']
    if not prove_invariant("Ad Spend Absolute Cap", inv2, preconditions): all_proofs_passed = False
    
    # Proof 3: Profit Margin is always maintained
    final_profit_margin = (final_price - cfg['unit_cost']) / final_price
    inv3 = final_profit_margin >= cfg['min_margin']
    if not prove_invariant("Minimum Profit Margin", inv3, preconditions): all_proofs_passed = False

    print("\n--- Verification Summary ---")
    if all_proofs_passed:
        print("🎉 All critical safety invariants for SymbolicGuardianV3 have been mathematically proven! 🎉")
    else:
        print("🔥 One or more invariants failed. Please review the counter-examples above. 🔥")

if __name__ == "__main__":
    run_verification()