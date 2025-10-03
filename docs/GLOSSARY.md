# Project-Chimera Glossary

This glossary defines key terms used throughout Project-Chimera to help contributors and users understand the concepts quickly.

---

## Core Concepts & Project Terminology

| Term                                         | Description                                                                                                                                   |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| ACTION_TO_EXPLAIN                            | A standard action dictionary (e.g., `{'type': 'BUY', 'amount': 0.5}`) used to investigate what influences a specific model decision.          |
| AGENT_DOCTRINES                              | Strategic goals and guidelines assigned to each agent type.                                                                                   |
| AGENT_MODEL                                  | LLM model used by the Chimera-Quant agent (e.g., `"gpt-4o"`).                                                                                 |
| AI Gladiator                                 | A strategic AI agent with a type and doctrine, used in the simulation.                                                                        |
| Absolute advertising spend cap               | Maximum fixed amount allowed for advertising spend.                                                                                           |
| Absolute and Relative Advertising Spend Caps | Safety constraints limiting ad spending in simulations.                                                                                       |
| Adaptive Strategy Lab                        | The interactive environment where users test AI agents on dynamic e-commerce simulations.                                                     |
| Automated Benchmarking Suite                 | Scripts (`benchmark.py`) for performance evaluation of different agent architectures across scenarios.                                        |
| Balanced Strategy                            | A benchmark scenario aiming for a trade-off between profit and brand trust, avoiding extremes.                                                |
| BaseMultiAgentEnvironment                    | Abstract base class extending `BaseEnvironment` for multi-agent environments, where multiple agents can act simultaneously.                   |
| Battle Duration (Weeks)                      | Number of turns/weeks for which the Colosseum simulation runs.                                                                                |
| Bearish Scenario (Expect SHORT)              | A unit test case where market indicators suggest overbought conditions, expected to trigger a SHORT decision.                                 |
| Benchmark Scenarios                          | Predefined test scenarios (e.g., Brand Trust, Profit Maximization, Balanced Strategy) to evaluate agent performance.                          |
| Benchmark scripts (`benchmark.py`)           | Scripts for evaluating and testing performance of Chimera agents under various scenarios.                                                     |
| Binned Feature                               | Feature values grouped into quantile bins to visualize the relationship with outcomes.                                                        |
| Brand Trust                                  | A metric acting as armor that mitigates damage to the War Chest.                                                                              |
| Brand Trust Focus                            | A benchmark scenario prioritizing high brand trust over immediate profit.                                                                     |
| Bullish Scenario (Expect BUY)                | A unit test case where market indicators suggest oversold conditions, expected to trigger a BUY decision.                                     |
| Buffered Minimum Profit Margin               | Safety parameter enforced by SymbolicGuardianV4 to prevent losses due to rounding or edge cases.                                              |
| CHART_COLORS                                 | Predefined color palette used for charts and visualizations.                                                                                  |
| CLOSED_BETA_KEY                              | The secret key needed to access the closed beta version of the simulation.                                                                    |
| Causal                                       | Data-driven causal inference engine (“The Oracle”) predicting financial impact of decisions (uses EconML).                                    |
| Causal Engine                                | The core project module (here `CausalEngineV7_Quant`) that predicts or simulates causal effects on profit using features and historical data. |
| CausalEngineV5                               | A causal inference module (defined in `components.py`) that models cause-effect relationships for decision-making.                            |
| CausalEngineV6                               | Project-specific causal engine module that retrains on performance data for dynamic learning.                                                 |
| CausalEngineV6_UseReady                      | Component that estimates causal effects of actions on long-term value and generates explainable AI (XAI) outputs.                             |
| CausalEngineV7_Quant                         | Component that estimates the causal profit impact of validated trading actions using historical data and feature context.                     |
| CausalForestDML                              | Causal machine learning model used to estimate treatment effects of actions (price_change, ad_spend) on profit_change.                        |
| Chimera Agent                                | The custom trading agent whose performance is being evaluated in the backtests.                                                               |
| Chimera Colosseum                            | A multi-agent competitive simulation environment where AI gladiators battle in live simulations.                                              |
| Chimera-Quant                                | The autonomous trading agent being backtested; follows a strict decision-making workflow with causal estimation and risk validation.          |
| ChimeraGuardianProof.tla                     | TLA+ specification file containing SymbolicGuardianV4 logic.                                                                                  |
| Chimera_Performance_Report_Final.png         | The final output report image containing the 4-panel dashboard for the Chimera Agent performance.                                             |
| Closed Beta                                  | Early access program for selected users to test Project Chimera features.                                                                     |
| Colosseum                                    | The gamified simulation arena where AI gladiators compete.                                                                                    |
| DEFAULT_TRUST_VALUE_MULTIPLIER               | A default multiplier used to scale trust-related metrics in simulations when a dynamic multiplier is not specified.                           |
| Domain-Agnostic Framework                    | Project vision to generalize Chimera logic for domains beyond e-commerce, like finance or healthcare.                                         |
| ELIMINATION_THRESHOLD                        | War Chest value below which a gladiator is considered eliminated.                                                                             |
| EcommerceSimulatorV5                         | A simulation engine representing an e-commerce market environment, providing market state and accepting agent actions.                        |
| EcommerceSimulatorV7                         | Core simulation engine that models the competitive e-commerce environment.                                                                    |
| FEATURE_COLS_DEFAULT                         | List of standardized feature column names used for model training and analysis, imported from `quant_prepare_training_data.py`.               |
| Fingerprint collision probability            | Probability that TLC states could falsely appear identical due to hashing; used to validate model reliability.                                |
| Full Neuro-Symbolic-Causal                   | The complete Chimera agent integrating neural, symbolic, and causal components.                                                               |
| Gladiator                                    | An AI agent participating in the Colosseum battles.                                                                                           |
| GradientBoostingRegressor                    | Base model used within the causal model for predicting outcomes.                                                                              |
| HUMAN_PROMPT_TEMPLATE                        | A prompt template passed to each agent, including the agent’s ID, week, strategic goal, and current market state.                             |
| INITIAL_CAPITAL                              | Starting capital allocated to the Chimera-Quant agent for the simulation.                                                                     |
| Init                                         | Initial state definition in TLA+ model.                                                                                                       |
| Invariant_AdSpendAbsolute                    | TLA+ invariant ensuring absolute advertising spend cap is respected.                                                                          |
| Invariant_AdSpendRelative                    | TLA+ invariant ensuring relative advertising spend cap is respected.                                                                          |
| Invariant_BufferedMargin                     | TLA+ invariant ensuring that the buffered minimum profit margin is never violated.                                                            |
| Invariant_PriceCap                           | TLA+ invariant ensuring the maximum price cap is never exceeded.                                                                              |
| Key Performance Indicators (KPI)             | Summary metrics in the report, including final portfolio value, total return, max drawdown, and total trades.                                 |
| LLM + Symbolic                               | Agent type using a large language model along with symbolic rule checking.                                                                    |
| LLM-Only                                     | Baseline agent using only a Large Language Model without symbolic or causal modules.                                                          |
| MC.cfg                                       | Model configuration file specifying Init/Next actions and invariants for TLC runs.                                                            |
| MarketSimulatorV2                            | Simulator that models market behavior and tracks the agent’s portfolio over the simulation period.                                            |
| Maximum Price Cap                            | Upper limit on pricing enforced by Guardian logic.                                                                                            |
| Maximum price cap                            | Upper limit on product price enforced by the Guardian logic.                                                                                  |
| Minimum safe price threshold                 | The lowest allowable price for a product, protected by the safety buffer.                                                                     |
| Multi-Agent Competitive Simulations          | Future roadmap concept where multiple Chimera agents interact and compete in the same market.                                                 |
| Multi-Agent Simulation Suite                 | A descriptive subtitle for Project Chimera, highlighting its focus on multi-agent AI simulations.                                             |
| Multi-Hypothesis Reasoning                   | Feature where the agent generates multiple strategies and evaluates them before selecting the optimal one.                                    |
| MultiOutputRegressor                         | Wrapper allowing multi-output regression for the causal model’s treatment variables.                                                          |
| Neuro-Symbolic-Causal Agent                  | Hybrid AI architecture combining neural (LLM/GPT-4o), symbolic (rule-based safety/Guardian), and causal (profit prediction) components.       |
| Next                                         | Next-state relation defining system transitions in TLA+ model.                                                                                |
| NUM_WEEKS                                    | Global constant defining the number of simulation weeks to run.                                                                               |
| NUM_WEEKS_TO_SIMULATE                        | The total number of weeks each agent is simulated for in the benchmark.                                                                       |
| OUTPUT_DIR                                   | Directory path (`results/quant/causal_engine_tests_phase2`) where Phase 2 test outputs (plots, SHAP visualizations) are saved.                |
| Pandas TA (`pandas_ta`)                      | Technical analysis library for pandas; used in the project for generating trading indicators and strategies.                                  |
| Profit Maximization                          | A benchmark scenario prioritizing cumulative profit above other objectives.                                                                   |
| Project Chimera                              | The overarching AI project; a neuro-symbolic-causal agent for strategic decision-making in business environments.                             |
| Randomized augmentation                      | Process of probabilistically adding BUY or SHORT samples to the training dataset to increase representation of actionable scenarios.          |
| Relative advertising spend cap               | Maximum allowed advertising spend as a percentage of revenue or budget.                                                                       |
| SAFETY_BUFFER_ABS                            | Parameter defining the absolute safety buffer (0 in this model).                                                                              |
| SAFETY_BUFFER_RATIO                          | Parameter defining the proportional safety buffer (+1% in this model).                                                                        |
| SIMULATION_DAYS                              | Number of trading days to simulate in the backtest.                                                                                           |
| SYSTEM_PROMPT                                | Prompt template for Chimera-Quant defining the mandatory workflow, thought process, and JSON output format.                                   |
| Safety buffer                                | Configurable margin above minimum safe price to prevent rounding or precision errors.                                                         |
| Short Risk Penalty                           | Fixed penalty applied to SHORT trades during training data generation (`SHORT_RISK_PENALTY = 0.002`).                                         |
| StateProvider                                | Class that manages and provides the current state of an agent.                                                                                |
| Strategy Lab                                 | Streamlit-based interactive environment (`app.py`) for real-time agent interaction and analysis.                                              |
| Strategy Map                                 | Visualization of trades (BUY/SELL/SHORT) overlaid on the BTC price chart.                                                                     |
| Streamlit app (`app.py`)                     | The interactive front-end application where users can interact with Chimera agents.                                                           |
| StreamlitCallbackHandler                     | Callback utility for visualizing agent thought process inside Streamlit.                                                                      |
| Symbolic                                     | Rule-based component (“The Guardian”) ensuring safety and adherence to business rules.                                                        |
| SymbolicGuardianV3                           | A rule-based safety and validation module (defined in `components.py`) ensuring actions follow predefined constraints.                        |
| SymbolicGuardianV4                           | Version 4 of the Guardian module that enforces safety rules on pricing and advertising spend, formally verified in TLA+.                      |
| SymbolicGuardianV6                           | Component that validates trading actions against predefined rules and constraints (risk management).                                          |
| The Colosseum                                | A multi-agent competitive environment where AI agents (“gladiators”) compete in a live simulation.                                            |
| TLA+                                         | Temporal Logic of Actions, a formal specification language used to model and verify system behavior.                                          |
| TLC                                          | The TLA+ model checker used to verify invariants and explore possible system states.                                                          |
| TRAINING_HORIZON                             | Number of future days used to calculate profit/outcome for each training sample; set to 3 in this script.                                     |
| TRUST_VALUE_MULTIPLIER                       | A numeric multiplier used in the causal engine to weigh brand trust in decision-making.                                                       |
| Term                                         | Description                                                                                                                                   |
| War Chest                                    | Represents a gladiator’s “health” or accumulated resources during battle.                                                                     |
| Week / Market State                          | Simulation-specific concepts tracking the current week number, pricing, ad spend, and brand trust.                                            |
| What-If Analysis                             | Interactive simulator allowing users to explore counterfactual scenarios and agent reasoning.                                                 |
| XAI Suite                                    | Explainable AI features including SHAP panels for per-decision explainability.                                                                |
| XAI dashboard / explanation_obj              | Components and objects providing explainable AI insights into agent decisions (e.g., SHAP plots).                                             |

## Code Components & Implementation Details

| Term                                                         | Description                                                                                                                                                   |
|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| abs_ad_cap                                                   | Maximum allowed advertising spend set by SymbolicGuardianV4.                                                                                                  |
| abs_max_discount                                             | Maximum allowed discount set by SymbolicGuardianV4.                                                                                                           |
| abs_max_increase                                             | Maximum allowed price increase set by SymbolicGuardianV4.                                                                                                     |
| action_amount                                                | Randomized magnitude for the action in each sample (0–1) to simulate variable trade sizes.                                                                    |
| action_type_code                                             | Numerical encoding of actions: 1 for BUY, -1 for SHORT, 0 for HOLD/SELL (as used in training).                                                                |
| action_types                                                 | Dictionary mapping trade types (BUY, SELL, SHORT) to plotting markers in the Strategy Map.                                                                    |
| actions_for_turn`                                            | A collection of all agents’ proposed actions for a given simulation week.                                                                                     |
| actions_log                                                  | Log of all agent actions (BUY, SELL, SHORT, HOLD) during the simulation.                                                                                      |
| ad_cap, ad_increase_cap                                      | Maximum allowable weekly advertising spend and its weekly increment.                                                                                          |
| ad_log_scale, trust_ad_gain, trust_decay                     | Parameters controlling the effect of advertising on brand trust.                                                                                              |
| ad_spend                                                     | Action variable representing the advertising budget allocated for the week by the simulated agent.                                                            |
| agent_executor                                               | The object executing the agent’s reasoning loop with tools and prompts.                                                                                       |
| agent_goals                                                  | A list of strategic doctrines or objectives for each agent, defining their behavior (e.g., aggressive growth, brand custodianship, cautious market analysis). |
| agent_names                                                  | Human-readable names for the AI agents, corresponding to their strategic persona.                                                                             |
| agent_states                                                 | List of per-agent states in multi-agent simulations.                                                                                                          |
| ambiguous_scenario                                           | Example scenario with conflicting signals, used to test model decision consistency.                                                                           |
| analyze_and_report                                           | Function that generates and saves multi-panel performance report/dashboard including cumulative returns, KPIs, and visualizations.                            |
| apply_decision_and_prepare_experience                        | Applies the agent’s action to the simulator, logs results, and returns structured experience data.                                                            |
| base_context                                                 | Baseline feature values (mean of all features) used as reference when generating PDPs.                                                                        |
| base_demand, price_elasticity, seasonality_amp, `noise_sigma | Simulation parameters for demand modeling, price sensitivity, seasonal variations, and random noise.                                                          |
| brand_trust                                                  | Numeric value representing customer trust in the brand; affects sales.                                                                                        |
| btc_normalized                                               | Derived column representing the BTC price normalized to 100 at the start of the backtest for comparison.                                                      |
| bullish_scenario                                             | Example scenario with high momentum where model is likely to favor a BUY action.                                                                              |
| causal_training_data_balanced.csv                            | Output CSV file containing the full training dataset after augmentation and feature engineering.                                                              |
| check_action_validity                                        | Tool function that checks if a proposed trading action is valid according to SymbolicGuardianV6 rules.                                                        |
| check_business_rules                                         | Agent tool to validate proposed actions against symbolic business rules.                                                                                      |
| cumulative_profit                                            | Derived metric tracking the total profit accumulated by an agent over time.                                                                                   |
| decision_cache                                               | Session state dictionary to cache computed agent decisions.                                                                                                   |
| df_features                                                  | DataFrame containing all features created by `create_features()` for testing and analysis.                                                                    |
| drawdown_ratio                                               | Column representing portfolio drawdown relative to its running peak; used to calculate Max Drawdown.                                                          |
| env_check.py                                                 | Script used to verify that the Python environment is correctly set up and that required libraries (e.g., `pandas_ta`) are installed and functional.           |
| estimate_causal_effect                                       | Method to predict long-term value of an action using the causal model.                                                                                        |
| estimate_profit_impact                                       | Tool function that predicts the potential profit impact of a validated trading action using CausalEngineV7_Quant.                                             |
| experience_history                                           | Session state list of all applied agent experiences for potential retraining.                                                                                 |
| explain_decision                                             | Generates SHAP-based explanation of model predictions for a given context and action.                                                                         |
| featured_data                                                | DataFrame returned by `create_features()`, containing raw prices and engineered features.                                                                     |
| features_to_plot_pdp                                         | Key features analyzed via PDP: `"Rsi_14"`, `"Roc_5"`, `"Macdh_12_26_9"`, `"Bb_position"`, `"Price_vs_sma10"`, `"Sma10_vs_sma50"`.                             |
| final_backtest_actions.csv                                   | CSV file storing all executed trading actions (BUY, SELL, SHORT, HOLD) from the final backtest.                                                               |
| final_backtest_history.csv                                   | CSV file storing the historical portfolio values, market data, and other relevant metrics from the final backtest.                                            |
| generate_feedback_on_last_action                             | Function that provides feedback to the agent if its previous action was modified by the guardian.                                                             |
| generate_synthetic_data                                      | Function that runs multiple simulations to generate synthetic data for model training.                                                                        |
| get_agent_decisions                                          | Retrieves decisions from all active agents each week.                                                                                                         |
| get_decision_from_response                                   | Function that extracts structured decision information from the agent’s JSON output.                                                                          |
| get_default_gladiator                                        | Generates a default AI gladiator with a specified type and doctrine.                                                                                          |
| get_dynamic_trust_multiplier                                 | Function that interprets user strategic goals to generate a trust value multiplier.                                                                           |
| get_state_key                                                | Function to discretize continuous market state into a tuple for caching.                                                                                      |
| hide_default_sidebar_nav                                     | A custom function imported from `src.ui` that hides Streamlit's default sidebar navigation elements.                                                          |
| indices_short / indices_buy                                  | Indices of samples that satisfy technical conditions for SHORT or BUY augmentation.                                                                           |
| log_entry / decision_log                                     | Structure storing historical decisions, predicted values, and actual outcomes.                                                                                |
| market_share_evolution                                       | Visualization showing changes in each agent’s market share over time.                                                                                         |
| market_share                                                 | Fraction of total demand captured by an agent based on attractiveness score.                                                                                  |
| min_margin                                                   | Minimum allowed profit margin for pricing actions.                                                                                                            |
| multi-agent competitive analysis                             | Analytical mode involving multiple AI agents interacting competitively.                                                                                       |
| multi_agent_results.csv                                      | CSV file storing the full simulation results including weekly profit, price, ad spend, brand trust, and market share for each agent.                          |
| outcome_col                                                  | Column name representing the future profit change over `TRAINING_HORIZON` days (e.g., `profit_change_h3`).                                                    |
| outcome_profit_change                                        | Profit or loss resulting from the action over the defined horizon; clipped to [-0.15, 0.15] to limit extreme values.                                          |
| page_link                                                    | Streamlit component linking to individual app pages (e.g., Strategy Lab or Colosseum).                                                                        |
| parse_final_json                                             | Helper function to extract the final JSON decision block from the agent’s output.                                                                             |
| price_upper, price_lower                                     | Maximum and minimum allowed prices in simulators.                                                                                                             |
| portfolio_normalized                                         | Derived column in the report representing the portfolio value normalized to 100 at the start of the backtest.                                                 |
| prediction_wrapper                                           | Helper function that converts input features into predictions via the Causal Engine for SHAP analysis.                                                        |
| price_change                                                 | Action variable representing a change in product price applied by the simulated agent.                                                                        |
| print_daily_report                                           | Helper function to generate a formatted terminal report of daily market state, agent reasoning, and portfolio status.                                         |
| process_battle_mechanics                                     | Computes weekly outcomes, updates War Chests, and tracks eliminations.                                                                                        |
| profit_change                                                | Outcome variable measuring the change in profit after an action is applied in the simulation.                                                                 |
| render_battle_results                                        | Displays final leaderboard, charts, and allows sharing/export of results.                                                                                     |
| render_colored_progress_bar                                  | UI helper to visually represent War Chest or progress.                                                                                                        |
| repair_action                                                | Method in `SymbolicGuardianV4` that adjusts actions to satisfy constraints.                                                                                   |
| safe_action                                                  | Action modified or approved by SymbolicGuardian to comply with business rules.                                                                                |
| season_phase                                                 | A simulation parameter representing the week’s phase in the sales/season cycle.                                                                               |
| shap_values                                                  | Computed SHAP values for a scenario, indicating the contribution of each feature to the predicted causal effect.                                              |
| simulate_plan                                                | Simulates a sequence of actions and returns aggregated metrics (profit, sales, trust).                                                                        |
| single-agent analysis                                        | Analytical mode focusing on one AI agent at a time.                                                                                                           |
| state_with_competitors                                       | The agent’s current state enriched with information about competitors’ positions and actions.                                                                 |
| strategy_distributions                                       | Visualization showing distribution of key metrics (price, ad spend, profit) to analyze agent behavior and volatility.                                         |
| strategy_map                                                 | Visualization plotting agents’ price vs. ad spend to illustrate strategic tendencies.                                                                         |
| trades                                                       | Subset of `actions_df` filtering out HOLD actions for plotting strategy maps and metrics.                                                                     |
| trained_causal_model.pkl                                     | File where the trained causal model is serialized and saved for later use in the simulation or Streamlit app.                                                 |
| trust_adjusted_profit_change                                 | Profit change adjusted for brand trust changes, used in causal modeling.                                                                                      |
| trust_change                                                 | Outcome variable measuring the change in brand trust after an action is applied in the simulation.                                                            |
| unit_cost                                                    | Internal parameter for product cost, used to calculate margins and safe prices.                                                                               |
| update_live_view                                             | Renders live charts, metrics, and combat logs in the Streamlit interface.                                                                                     |
| validate_action                                              | Method in `SymbolicGuardianV4` that checks if a proposed action complies with business rules.                                                                 |
| wi_price / wi_trust / wi_ad                                  | Session state variables used in the What-If analysis to tweak market conditions.                                                                              |

---

*This glossary should be updated as new concepts and terms are introduced in Project-Chimera.*
