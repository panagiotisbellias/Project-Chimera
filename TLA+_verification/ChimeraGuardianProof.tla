---------------- MODULE ChimeraGuardianProof ----------------

EXTENDS Naturals, TLC

(*
  Model Summary:
  - price: Weekly product price (integer).
  - week: Week counter (integer).
  - ad_spend: Current week's advertising spend (integer).
  - prev_ad: Previous week's advertising spend (integer).

  V4 alignment (with safety buffer):
  - Proposed price changes are drawn from a finite set (incl. a negative option).
  - Price repair:
      * Clip proposed percent change to [-MAX_DISCOUNT_PERC, +MAX_INCREASE_PERC].
      * Apply MAX_PRICE cap.
      * Enforce minimum profit margin with an additional safety buffer
        to avoid rounding/precision edge-case violations.
  - Ad spend repair:
      * Non-negative.
      * Absolute cap (AD_CAP).
      * Relative week-over-week cap (AD_INCREASE_CAP).
  - Invariant:
      * Price always at or above the buffered minimum safe price (and above unit cost).
*)

VARIABLES price, week, ad_spend, prev_ad

\* Configuration (operators, no model-time assignment required)
UNIT_COST == 5000
MIN_PROFIT_MARGIN_PERC == 15
MAX_DISCOUNT_PERC == 40
MAX_INCREASE_PERC == 50
MAX_PRICE == 15000
AD_CAP == 5000
AD_INCREASE_CAP == 1000

\* V4 safety buffer parameters (match Python defaults)
SAFETY_BUFFER_RATIO == 1    \* Represents +1% as integer percentage; ratio = SAFETY_BUFFER_RATIO / 100 = 0.01
SAFETY_BUFFER_ABS == 0      \* Optional absolute buffer in currency units

\* Helper: exact minimum safe price (integer division)
MIN_SAFE_PRICE ==
  (UNIT_COST * 100) \div (100 - MIN_PROFIT_MARGIN_PERC)

\* Helper: buffered minimum safe price (V4 logic)
\* We model ratio as integer percent to avoid non-integer arithmetic:
\*   buffered = MIN_SAFE_PRICE * (1 + ratio) + abs_buffer
\*            = MIN_SAFE_PRICE + (MIN_SAFE_PRICE * SAFETY_BUFFER_RATIO) / 100 + SAFETY_BUFFER_ABS
\* Use integer division \div for the ratio term.
MIN_SAFE_PRICE_WITH_BUFFER ==
  MIN_SAFE_PRICE + ((MIN_SAFE_PRICE * SAFETY_BUFFER_RATIO) \div 100) + SAFETY_BUFFER_ABS

\* Price repair function (clip percentage, apply cap, enforce buffered min margin)
RepairedPrice(p_current, p_change_perc) ==
  LET clipped ==
        IF p_change_perc > MAX_INCREASE_PERC
        THEN MAX_INCREASE_PERC
        ELSE IF p_change_perc < (0 - MAX_DISCOUNT_PERC)
             THEN (0 - MAX_DISCOUNT_PERC)
             ELSE p_change_perc
      adjusted ==
        p_current + ((p_current * clipped) \div 100)
      capped ==
        IF adjusted > MAX_PRICE THEN MAX_PRICE ELSE adjusted
  IN
    IF capped < MIN_SAFE_PRICE_WITH_BUFFER
    THEN MIN_SAFE_PRICE_WITH_BUFFER
    ELSE capped

\* Advertising spend repair function (non-negative, absolute cap, relative cap)
RepairedAdSpend(ad, prev) ==
  LET non_negative ==
        IF ad < 0 THEN 0 ELSE ad
      abs_capped ==
        IF non_negative > AD_CAP THEN AD_CAP ELSE non_negative
      rel_capped ==
        IF (abs_capped - prev) > AD_INCREASE_CAP
        THEN (prev + AD_INCREASE_CAP)
        ELSE abs_capped
  IN rel_capped

\* Profit margin safety check (integer-safe comparison)
\* Ensures price is strictly above UNIT_COST and satisfies buffered threshold.
ProfitMarginIsSafeBuffered(p) ==
  /\ p > UNIT_COST
  /\ p >= MIN_SAFE_PRICE_WITH_BUFFER

\* Initial state
Init ==
  /\ price = 10000
  /\ week = 1
  /\ ad_spend = 0
  /\ prev_ad = 0

\* Action choices (avoid unary minus by using (0 - 50))
PriceChangeChoices == {(0 - 50), 0, 20, 60}
AdSpendChoices == {0, 500, 1000, 2000, 4000, 5000}

\* Weekly step
Next ==
  ( /\ week < 52
    /\ \E change \in PriceChangeChoices:
         \E ad_proposed \in AdSpendChoices:
           /\ price' = RepairedPrice(price, change)
           /\ ad_spend' = RepairedAdSpend(ad_proposed, ad_spend)
           /\ prev_ad' = ad_spend
           /\ week' = week + 1
  )
  \/ ( /\ week >= 52
       /\ UNCHANGED <<price, week, ad_spend, prev_ad>> )

\* System behavior (stuttering allowed)
vars == <<price, week, ad_spend, prev_ad>>
Spec == Init /\ [][Next]_vars

\* Invariants (V4 safety goals)
Invariant_BufferedMargin == ProfitMarginIsSafeBuffered(price)
Invariant_PriceCap == price <= MAX_PRICE
Invariant_AdSpendAbsolute == ad_spend <= AD_CAP
Invariant_AdSpendRelative == (ad_spend - prev_ad) <= AD_INCREASE_CAP

=============================================================================
