# Contributing to Project Chimera

First off, thank you for considering contributing to Project Chimera! We're thrilled you're interested in helping us build the future of strategic AI agents. Every contribution, from a small typo fix to a new feature, is valuable.

This document provides guidelines for contributing to the project.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

There are many ways to contribute to Project Chimera, and all of them are appreciated:

* **🐛 Reporting Bugs:** Open an issue with detailed reproduction steps.
* **💡 Suggesting Enhancements:** Share ideas for improvements, referencing the [Public Roadmap](https://github.com/akarlaraytu/Project-Chimera/issues/4) where relevant.
* **📝 Improving Documentation:** Fix typos, clarify docstrings, or improve README/in-code docs.
* **💻 Writing Code:** Start with issues labeled `help wanted` or `good first issue`.

## Development Setup

1. **Fork & Clone the Repository**
    ```bash
    git clone https://github.com/akarlaraytu/Project-Chimera.git
    cd Project-Chimera
    ```

2. **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set Your API Key**
    ```bash
    export OPENAI_API_KEY='sk-...'
    ```

5. **Run the Applications**
    ```bash
    streamlit run app.py
    python3 benchmark.py
    ```

## Contribution Workflow

Follow the standard GitHub Fork & Pull Request workflow:

1. **Create a Branch**
    ```bash
    git checkout -b feature/YourAmazingFeature
    ```
2. **Make Your Changes**
3. **Commit Your Changes** (use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/))
    ```
    feat(causal): Add seasonality to CausalEngine training data
    fix(app): Correctly handle API key errors
    ```
4. **Push to Your Fork**
    ```bash
    git push origin feature/YourAmazingFeature
    ```
5. **Open a Pull Request** to `main` branch in the original repo.

## Pull Request Guidelines

* Clear, descriptive title.
* Detailed description of "what" and "why"; link issues if applicable (e.g., `Closes #123`).
* Self-review your code before submitting.
* Ensure tests/benchmarks pass; avoid regressions.

## Coding Style

* **Formatting:** Use `black`.
* **Linting:** Use `flake8`.
* **Docstrings & Comments:** Provide clear docstrings and comments for complex logic.

## Project Architecture Overview

* **`components.py`** – Core AI components: `EcommerceSimulatorV5`, `SymbolicGuardianV3`, `CausalEngineV5`.
* **`app.py`** – Interactive Streamlit app for user interaction.
* **`benchmark.py`** – Scripts for testing and benchmarking agent performance.

## References

* [Code of Conduct](CODE_OF_CONDUCT.md)  
* [Public Roadmap](https://github.com/akarlaraytu/Project-Chimera/issues/4)  
* Issues labeled [`good first issue`](https://github.com/akarlaraytu/Project-Chimera/labels/good%20first%20issue) or [`help wanted`](https://github.com/akarlaraytu/Project-Chimera/labels/help%20wanted)

Thank you again for your interest in Project Chimera! We look forward to your contributions.
