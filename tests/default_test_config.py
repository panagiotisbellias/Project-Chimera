# Config for basic benchmark script.
basic = {
    "NUM_WEEKS_TO_SIMULATE": 52,
    "AGENT_TYPES_TO_TEST": [
        "Full Neuro-Symbolic-Causal",
        "LLM + Symbolic",
        "LLM-Only",
    ],
}

# Config for multi agent benchmark script.
multi_agent = {"NUM_AGENTS": 3, "NUM_WEEKS": 52}
