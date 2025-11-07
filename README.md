# Project Chimera: A Neuro-Symbolic-Causal AI Agent for Strategic Decision-Making
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

## ❤️ Our Contributors!

[![Contributors](https://contrib.rocks/image?repo=akarlaraytu/Project-Chimera&cache_bust=1)](...)

---

**Project Chimera is an advanced AI agent designed to overcome the critical limitations of standard Large Language Models (LLMs) in strategic environments like trading or business management. By integrating a hybrid Neuro-Symbolic-Causal architecture, this agent makes decisions that are not only intelligent but also safe, explainable, and designed for robustness.**

---

Research Update (Nov 2025): Project Chimera’s theoretical foundations are now live on arXiv — [arXiv:2510.23682](https://arxiv.org/abs/2510.23682)

---

### 🚀 Live Demo & Usage

#### Try the Interactive Lab (Business Context)

<a href="https://project-chimera.streamlit.app/" target="_blank"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Live Demo"></a>

---

## ❗ The Problem: Why Raw LLMs Can Be Risky

Modern LLMs are powerful, but when entrusted with critical decisions, they can be dangerously naive and unpredictable. Without proper guardrails, they can make catastrophic mistakes. Our benchmark experiment proved this: a pure `LLM-Only` agent, lacking rules or causal understanding, drove a simulated company into a multi-billion dollar loss.

---

## 💡 The Solution: The Chimera Agent in Action

Project Chimera solves this by providing the LLM with a **Symbolic** safety net and a **Causal** oracle. It doesn't just guess; it brainstorms multiple strategies, checks them against rules, and predicts their likely outcomes to find the optimal path.

You can try a **live demo** of the Strategy Lab here (Note: Demo might reflect business simulation context):

<a href="https://project-chimera.streamlit.app/" target="_blank"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Live Demo"></a>

**See Chimera in Action** (Business Simulation Demo)

![Chimera Demo](assets/all-demo-chimera-final.gif)

---

## 🧩 The Chimera Architecture

![The Chimera Architecture](assets/architecture.png)

* **🧠 Neuro (The Brain):** The creative core (e.g., GPT-4o) that understands goals and brainstorms diverse strategies.
* **🛡️ Symbolic (The Guardian):** A rule engine that acts as a safety net, preventing catastrophic, rule-breaking decisions.
* **🔮 Causal (The Oracle):** A data-driven causal inference engine (e.g., `EconML`, custom models) that predicts the real-world impact (like profit) of potential decisions.

---

### ✨ Key Features

* **Live 24/7 Trading Stream:** Real-time dashboard showcasing agent performance and decisions.
* **Multi-Hypothesis Reasoning:** Actively brainstorms and evaluates multiple strategies before making a data-driven recommendation.
* **Dynamic Learning (Optional):** Causal Engine can be retrained periodically on performance data for adaptation.
* **Advanced XAI Suite:** Includes per-decision explainability and an interactive 'What-If' simulator.
* **Sophisticated Simulation Environment:** Used for backtesting and benchmarking.
* **Interactive Strategy Lab:** A Streamlit application (`app.py`) for real-time interaction and analysis.
* **Automated Benchmarking Suite:** Rigorous comparison scripts (`benchmark.py`).

---

## 🔴 LIVE 24/7 Trading Stream Now Active!

**Watch the Chimera agent trade Bitcoin (Paper Trading) live on YouTube!** See its decisions, performance, and analysis updated in real-time on the dashboard.

[![Chimera Live Trading Stream](assets/chimera_live.png)](https://www.youtube.com/watch?v=fa2Skk3dKnk)

**(Click the image above to watch the live stream)**

---

## 📊 Latest Performance Results

Here's a look at the agent's performance in recent backtests:

**1. Long-Term Quant Agent Performance (200 Trading Days)**

This report highlights the agent's ability to significantly outperform a Buy & Hold strategy over an extended period, demonstrating robust cumulative returns while managing drawdowns.

![Project Chimera: Final Quant Agent Performance](results/quant/Chimera_Performance_Report_Final.png)

**2. Recent Performance Detail (30 Trading Days)**

This detailed analysis showcases the agent's decision-making on a shorter timeframe, including key performance indicators like Sharpe Ratio and a map of individual trade decisions on the price chart.

![Chimera-Quant: Recent Backtest Performance Analysis](results/quant/recency_test_report.png)

---

## 🔬 Advanced XAI: From Glass Box to Interactive Simulator
Version 1.2.1 introduced a full suite of XAI tools. The **"What-If Analysis"** tab allows you to explore the agent's reasoning by testing counterfactual scenarios live.

See the Interactive 'What-If' Simulator in Action (Business Simulation Demo):

![Chimera What-If Demo](assets/demo-what-if-final.gif)

---

## ✅ Formal Verification (TLA+)
SymbolicGuardianV4 introduced configurable safety buffers analyzed using TLA+ formal verification. This provides mathematical confidence that the safety logic consistently enforces defined constraints like minimum margins or maximum limits, even under complex scenarios.

![TLA+ Run Result](TLA+_verification/img/tla+_run_result.png)

* **Result:** Exhaustive exploration found **0 invariant violations**.
* **Interpretation:** High confidence in the safety guarantees provided by the Symbolic Guardian component. (See [V1.2.3 Release](https://github.com/akarlaraytu/Project-Chimera/releases/tag/v1.2.3 ) for details)

---

### 🚀 v1.4.0: The Colosseum

![The Chimera Colosseum](assets/the_chimera_colosseum.png)

**The arena is open!** This update transforms Project Chimera into a dynamic, multi-agent competitive ecosystem. Assemble AI gladiators and watch them battle in a live simulation.

This is currently an exclusive **Closed Beta**. Access details below. Learn more in the **[v1.4.0 release notes](https://github.com/akarlaraytu/Project-Chimera/releases/tag/v1.4.0)**.

---

## 📚 Documentation & Glossary

For a quick reference of key concepts, see our [Glossary](docs/GLOSSARY.md). Check the [latest release notes](https://github.com/akarlaraytu/Project-Chimera/releases) for detailed updates.

---

### 🗺️ Future Roadmap

Project Chimera is a living project. The roadmap outlines our vision for evolving this agent into a truly general-purpose decision-making engine and, ultimately, an open-source ecosystem.

---

**High-Level Phases**

* ✅ **Phase 1 — Strategic Depth & Provable Safety** — *COMPLETE*
    * *Theme:* Transform the agent from a simple optimizer into a strategist that is explainable, competitive, and mathematically verifiable.
* ✅ **Phase 2 — Architectural Abstraction & Cross-Domain Mastery** — *COMPLETE*
    * *Theme:* Decouple the agent's brain from any single environment, proving its universal capabilities in the high-stakes world of quantitative finance.
* ▶️ **Phase 3 — Ultimate Generalization & Autonomous Discovery** — *IN PROGRESS*
    * *Theme:* Test the limits of the architecture's flexibility (e.g., in Supply Chain simulations) and unlock the agent's ultimate intelligence layer: the ability to learn how the world works on its own.
* **Phase 4 — Ecosystem Leadership: The "Chimera Dev Kit"**
    * *Theme:* Package our proven technology into an open-source development kit for the world to build the next generation of trustworthy AI.

---

**Current Milestone:** **Phase 3 — Ultimate Generalization & Autonomous Discovery**

Having proven the architecture's core abstraction (Phase 1) and its effectiveness in quantitative finance (Phase 2), we now focus on pushing the boundaries of generalization and intelligence.

**Phase 3 Tasks:**

1.  **Final Flexibility Test (Supply Chain Simulation):**
    * *What:* Prove the architecture's universality by implementing a `SupplyChainSimulator` for a completely different complex domain.
    * *Status:* *Upcoming.*
2.  **The Revolutionary Leap: "Autonomous Causal Discovery"**
    * *What:* Evolve the Causal Engine from a system that learns from *given data* to one that *autonomously discovers new causal relationships* by designing and running its own experiments within an environment.
    * *Impact:* This fundamentally changes how AI understands the world. We no longer just tell it the rules; *it learns the changing rules of the game on its own*.
    * *Status:* *Research & Development in progress.*

---

**Future Phases**

* **Phase 4: Ecosystem Leadership: The "Chimera Dev Kit"**
    * *Goal:* Package the entire project into a clean, well-documented, `pip install chimera-agent` library.
    * *Includes:* Comprehensive documentation, tutorials for E-commerce, Quant Finance, and Supply Chain domains.
    * *Vision:* Transform the project into a living open-source ecosystem adopted by a global community.

---

### 🤝 How to Help

* **⭐ Star the repo** to follow development.
* Check **Discussions → Roadmap** for areas needing feedback.
* See **CONTRIBUTING.md** for setup and PR guidelines.
* Look for **`good first issue`** tasks if you want to contribute code.

---

#### Getting Started (Local Setup)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/akarlaraytu/Project-Chimera.git](https://github.com/akarlaraytu/Project-Chimera.git)
    cd Project-Chimera
    ```
2.  **Create environment & install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Set API Keys (e.g., OpenAI, Alpaca):**
    ```bash
    # Create a .env file in the root directory
    # Add your keys like this:
    # OPENAI_API_KEY='sk-...'
    # ALPACA_API_KEY='...'
    # ALPACA_SECRET_KEY='...'
    # YOUTUBE_STREAM_KEY='...'
    ```
4.  **Run Applications:**
    ```bash
    # Run the Interactive Demo (Streamlit App for E-commerce Simulation)
    streamlit run app.py

    # Run Automated Benchmarks (E-commerce Simulation)
    python3 benchmark.py
    python3 benchmark_learning.py

    # Run Quant Trading Backtest
    python3 quant_run_backtest.py

    # Run the Live Quant Trading Agent (Requires .env setup)
    # python3 live_paper_trader.py

    # Run the Live Quant Trading Dashboard Stream (Requires .env setup)
    # python3 chimera_live.py --stream
    ```

---

### 🤝 Contributing

Contributions, issues, and feature requests are welcome! Check the [issues page](https://github.com/akarlaraytu/Project-Chimera/issues).

### 📄 License

This project is licensed under the **GNU AGPLv3 License** - see the `LICENSE` file.

---

*Developed with passion by [Aytug Akarlar](https://www.linkedin.com/in/aytuakarlar/) *