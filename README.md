# MultiAgent Data Framework

This repository contains a simple yet powerful framework for working with
tabular data via large language models (LLMs). It demonstrates how to build
specialised agents for different datasets—supply chain, manufacturing and
sales—and coordinate them to answer complex questions, surface proactive
insights and expose an interactive dashboard.

## Contents

* **multiagent/** – Python package implementing the agent framework. The
  `agents` module defines the core classes (`BaseDataAgent` and
  `MultiAgentManager`), while the `datasets` folder contains example CSV
  files used by the default agents. The `__init__.py` file exposes the
  high‑level API.
* **dashboard.py** – Streamlit application for exploring your data in real
  time. It surfaces automated insights and lets you ask natural language
  questions.
* **README.md** – this file.

## Installation

Clone this repository and install the dependencies in a Python 3.9+
environment. You will need the `pandas`, `streamlit`, `langchain` and
`openai` packages:

```bash
pip install pandas streamlit langchain openai
```

You also need an OpenAI API key. Sign up at <https://platform.openai.com/>,
create a key and export it as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

## Usage

### Programmatic Access

Import the framework and create a manager using the bundled example
datasets:

```python
from multiagent import create_default_agents

manager, agents = create_default_agents()

print(manager.generate_trends())

answer = manager.ask("What were the total sales in March 2025?")
print(answer)
```

You can also work with individual agents:

```python
agent = agents["Sales"]
print(agent.generate_trend_report())
```

### Streamlit Dashboard

Launch the interactive dashboard to explore your data visually and via
natural language:

```bash
streamlit run dashboard.py
```

Use the **Generate Insights** button to display recent trends and
patterns across the supply chain, manufacturing and sales datasets. Then
enter a question into the text input to query the data directly.

## Customising the Framework

You can replace the bundled CSV files with your own data. Copy your
datasets into the `multiagent/datasets` directory or modify the
`create_default_agents()` function in `multiagent/agents.py` to load
data from other sources. Make sure the column names make sense for your
domain; for example, include a `date` column for temporal analyses.

## Deploying to GitHub

This repository is ready to be pushed to GitHub. Simply create a new
repository on GitHub (e.g. `multiagent-data-framework`) and push these
files. Once pushed, others can install your package via pip using a
`git+https` URL or by building a wheel.

## License

This project is provided for demonstration purposes and does not include
a license. Feel free to adapt it to your needs.