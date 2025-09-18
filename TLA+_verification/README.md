# SymbolicGuardian V4 — Formal Verification (TLA+)

## Purpose
SymbolicGuardianV4 introduces a configurable safety buffer above the minimum safe price threshold to eliminate rounding/precision edge‑case violations observed in formal analysis.  
This verification ensures that the repair logic **always** enforces:
- Buffered minimum profit margin
- Maximum price cap
- Absolute and relative advertising spend caps

The model is written in TLA+ and checked with the TLC model checker to provide mathematical confidence in these safety guarantees.

---

## Scope & Assumptions
- Price and advertising spend changes are drawn from discrete sets.
- Negative values are represented as `(0 - X)` to avoid parser ambiguity.
- Integer division (`\div`) is used for percentage calculations.
- Buffer parameters in this verification run:
  - `SAFETY_BUFFER_RATIO = 1` (i.e., +1%)
  - `SAFETY_BUFFER_ABS = 0`
- The model abstracts away real‑world noise, external market factors, and floating‑point behavior.

---

## How to Run the Model

1. Open **TLA+ Toolbox**.
2. Load `ChimeraGuardianProof.tla` from this direectory.
3. Create or open the model configuration (`MC.cfg`).
4. In **What is the behavior spec?**:
   - Init: `Init`
   - Next: `Next`
5. In **What to check?**:
   - Check **Deadlock**
   - Add the following invariants:
     - `Invariant_BufferedMargin`
     - `Invariant_PriceCap`
     - `Invariant_AdSpendAbsolute`
     - `Invariant_AdSpendRelative`
6. Run the model.

---

## Results Summary (Run: 2025‑09‑17)

![TLA+ Run Result](TLA+_verification/img/tla+_run_result.png)

- **Distinct states explored:** 7,639,419  
- **Diameter reached:** 5  
- **Invariant violations:** 0  
- **Observed fingerprint collision probability:** ~1.85 × 10⁻⁹  
- **Calculated upper bound:** 6.95 × 10⁻⁵

**Interpretation:**  
Under the modeled nondeterminism over price and advertising proposals, TLC exhaustively explored millions of states without finding any violation of the defined safety properties. The observed collision probability is far below the calculated bound, supporting the reliability of the result.

---

## Files in This Directory

- `ChimeraGuardianProof.tla` — TLA+ specification of Guardian V4 logic.
- `MC.cfg` — Model configuration (Init/Next, invariants).
- `runs/` — Saved TLC run outputs (e.g., `MC.out`, `MC_TE.out`).
- `img/` — Visualizations of state‑space progress and invariant status.

---

## Reproducibility

Anyone with TLA+ Toolbox can reproduce these results by following the steps above and using the provided `.tla` and `.cfg` files.  
For command‑line TLC runs, use the same Init/Next and invariants as in the configuration file.

---

## Release Note Excerpt

> Guardian V4’s repair logic has been formally verified in TLA+ to maintain all safety properties across millions of possible states. This includes a +1% safety buffer above the minimum safe price to eliminate rounding‑related edge cases, with zero invariant violations observed.
