# Contributing to Project Chimera

First off, thank you for considering contributing to Project Chimera! We're thrilled you're interested in helping us build the future of strategic AI agents. Every contribution, from a small typo fix to a new feature, is valuable.

This document provides guidelines for contributing to the project.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

There are many ways to contribute to Project Chimera, and all of them are appreciated:

* **🐛 Reporting Bugs:** If you find a bug in the code, please open an issue and provide a detailed description, including steps to reproduce it.
* **💡 Suggesting Enhancements:** Have an idea to make the agent smarter, the simulation more realistic, or the UI better? Open an issue to share your thoughts. We'd love to hear your ideas, especially those related to our [Public Roadmap](https://github.com/akarlaraytu/Project-Chimera/issues/4) goals in the `README.md`.
* **📝 Improving Documentation:** If you find parts of the documentation unclear or see areas for improvement in the code comments, docstrings, or the `README.md`, feel free to suggest changes.
* **💻 Writing Code:** If you're ready to jump into the code, you can start by looking at existing issues labeled `help wanted` or `good first issue`.

## Development Setup

To get your local development environment set up, please follow these steps:

1.  **Fork & Clone the Repository**
    * Fork the repository on GitHub.
    * Clone your fork locally:
        ```bash
        git clone [https://github.com/akarlaraytu/Project-Chimera.git](https://github.com/akarlaraytu/Project-Chimera.git)
        cd Project-Chimera
        ```

2.  **Create a Virtual Environment**
    * We strongly recommend using a virtual environment to manage dependencies.
        ```bash
        python3 -m venv venv
        source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
        ```

3.  **Install Dependencies**
    * Install all required packages from the `requirements.txt` file.
        ```bash
        pip install -r requirements.txt
        ```

4.  **Set Your API Key**
    * The agent requires an OpenAI API key to function. Set it as an environment variable.
        ```bash
        export OPENAI_API_KEY='sk-...'
        ```

5.  **Run the Applications**
    * To run the interactive lab:
        ```bash
        streamlit run app.py
        ```
    * To run the full benchmark suite:
        ```bash
        python3 benchmark.py
        ```

## Contribution Workflow

We follow the standard GitHub Fork & Pull Request workflow.

1.  **Create a Branch:** Create a new branch from `main` for your changes. Please use a descriptive name.
    ```bash
    git checkout -b feature/YourAmazingFeature
    ```
2.  **Make Your Changes:** Write your code, fix the bug, or improve the documentation.
3.  **Commit Your Changes:** Commit your changes with a clear and descriptive commit message. We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
    * Example: `feat(causal): Add seasonality to CausalEngine training data`
    * Example: `fix(app): Correctly handle API key errors`
4.  **Push to Your Fork:** Push your branch to your forked repository.
    ```bash
    git push origin feature/YourAmazingFeature
    ```
5.  **Open a Pull Request:** Go to the original Project Chimera repository on GitHub and open a Pull Request from your forked branch to the `main` branch.

## Pull Request Guidelines

To help us review your PR efficiently, please ensure the following:

* **Clear Title:** The title should be a short, descriptive summary of the change.
* **Detailed Description:** Explain the "what" and "why" of your changes. If it fixes an issue, please link to it (e.g., `Closes #123`).
* **Self-Review:** Review your own code changes one last time before submitting.
* **Testing:** Confirm that you have tested your changes. For any significant modification, please run the benchmark to ensure you haven't introduced a performance regression.

## Coding Style

We aim to follow standard Python best practices.
* **Code Formatting:** We use the `black` code formatter. Please format your code before committing.
* **Linting:** We use `flake8` for linting to catch common issues.
* **Docstrings & Comments:** Please add clear docstrings to new functions and classes, and comment on complex parts of your code.

## Project Architecture Overview

To help you find your way around the codebase, here's a brief overview of the key files:

* **`components.py`:** The heart of the project. Contains the core AI components: `EcommerceSimulatorV5`, `SymbolicGuardianV3`, and `CausalEngineV5`.
* **`app.py`:** The interactive Streamlit application. The UI layout and user interaction logic reside here.
* **`benchmark.py`:** The research and testing script used to evaluate and compare the performance of the agents across different scenarios.

Thank you again for your interest in Project Chimera. We look forward to your contributions!
