# src/config.py

# =============================================================================
# Project-Wide Configuration
#
# Description:
#   This file serves as the single source of truth for shared configuration
#   variables across the entire project, such as feature lists.
# =============================================================================

# The definitive list of features for the "fast-paced" Causal Engine.
# Both data preparation and the engine itself will import from this list.
FEATURE_COLS_DEFAULT = [
    "Rsi_14",
    "Roc_15",         
    "Macdh_12_26_9",
    "Price_vs_sma20",  
    "Sma20_vs_sma100",  
    "Bb_position",
    "Atr_14"
]