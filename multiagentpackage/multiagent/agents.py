"""
Agents Module
-------------

This module defines a simple framework for interacting with multiple tabular
datasets through large language models (LLMs). It provides two main classes:

* :class:`BaseDataAgent` – wraps a single pandas ``DataFrame`` together with
  a LangChain pandas agent, exposing methods for running natural language
  queries and generating basic trend reports.
* :class:`MultiAgentManager` – orchestrates several data agents. It
  automatically decides which dataset(s) are relevant for a user's question,
  delegates the question to the appropriate agents and fuses their answers.

The helpers :func:`create_default_agents` and :func:`load_csv_dataset` make it
easy to instantiate the framework with the packaged example datasets (stored
under ``multiagent/datasets``). You can replace these with your own data
sources as needed.
"""

from __future__ import annotations

import os
import importlib.resources
from typing import Dict, List, Tuple

import pandas as pd  # type: ignore

try:
    # LangChain v0.1 and later have refactored chat models into the
    # ``langchain_community`` package and moved experimental agents into
    # ``langchain_experimental``. Attempt to import from the new locations
    # first. Fall back to the legacy imports for backwards compatibility.
    try:
        from langchain_community.chat_models import ChatOpenAI  # type: ignore
    except ImportError:
        # Fallback for older versions of langchain where chat models were
        # exposed directly under ``langchain.chat_models``
        from langchain.chat_models import ChatOpenAI  # type: ignore

    try:
        # For LangChain >= 0.1.0 use the experimental pandas agent
        from langchain_experimental.agents.agent_toolkits.pandas.base import (
            create_pandas_dataframe_agent,
        )  # type: ignore
    except ImportError:
        # Fallback to the legacy location for create_pandas_dataframe_agent
        from langchain.agents import create_pandas_dataframe_agent  # type: ignore

    # SystemMessage and HumanMessage have not moved between versions, but
    # keep the import inside the try block to ensure langchain is available.
    from langchain.schema import SystemMessage, HumanMessage  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "multiagent.agents requires the 'langchain', 'langchain-community', "
        "'langchain-experimental' and 'openai' packages. Please install them "
        "using pip before using this module."
    ) from exc


class BaseDataAgent:
    """Wrapper around a LangChain pandas agent for a single DataFrame.

    See the module documentation for a high‑level overview.
    """

    def __init__(self, name: str, dataframe: pd.DataFrame, llm: ChatOpenAI) -> None:
        self.name: str = name
        self.dataframe: pd.DataFrame = dataframe
        self.agent = create_pandas_dataframe_agent(llm, dataframe, verbose=False)

    def run(self, query: str) -> str:
        return self.agent.run(query)

    def generate_trend_report(self) -> str:
        """
        Generate a dynamic trend report for this dataset by delegating analysis to the
        underlying LangChain pandas agent.

        This method formulates an open‑ended prompt asking the agent to analyse the
        current DataFrame for trends, patterns, anomalies or inconsistencies. The
        agent will consider temporal trends (if a date column exists), the distribution
        of numerical columns and relationships between categorical variables. The
        result is returned as a human‑readable summary. If the agent call fails for
        any reason (e.g. network errors, API limits), the method falls back to a simple
        message indicating that no insights could be generated.
        """
        analysis_prompt = (
            "You are an expert data analyst. Analyse the dataset provided to you and "
            "identify notable trends, patterns, anomalies or inconsistencies. "
            "Consider temporal trends (if there is a date column), distribution of numeric columns, "
            "and any interesting relationships across categorical columns. "
            "Summarise your findings in a few concise bullet points."
        )
        try:
            insights = self.agent.run(analysis_prompt)
            return str(insights)
        except Exception:
            return "Unable to generate dynamic insights for this dataset."


class MultiAgentManager:
    """Coordinates multiple data agents and fuses their responses."""

    _ROUTER_PROMPT = (
        "You are a routing assistant. There are three datasets:\n"
        "1. Supply chain dataset – contains information about suppliers, inventory,\n"
        "   logistics and procurement.\n"
        "2. Manufacturing dataset – contains information about production lines,\n"
        "   quality metrics, throughput and capacity.\n"
        "3. Sales dataset – contains information about customer orders, revenue,\n"
        "   product performance and market segmentation.\n\n"
        "When a user asks a question, decide which dataset(s) are needed to answer\n"
        "it. Respond with a comma‑separated list of dataset names (e.g. \"Supply\n"
        "chain\", \"Manufacturing\", \"Sales\") without any explanation. If multiple\n"
        "datasets are relevant, list each once. If none match, say \"None\"."
    )

    _FUSION_PROMPT_TEMPLATE = (
        "You are an analyst combining answers from multiple\n"
        "datasets to respond to a user's question. The user's question was:\n"
        "\"{question}\"\n\n"
        "You have the following partial answers, each produced by a specialised\n"
        "agent:\n"
        "{answers}\n\n"
        "Please provide a final, unified answer that addresses the user's question.\n"
        "Integrate information where relevant and avoid repetition. If there are\n"
        "discrepancies between answers, mention them explicitly. Your reply should\n"
        "be comprehensive yet concise."
    )

    def __init__(self, llm: ChatOpenAI, agents: Dict[str, BaseDataAgent]) -> None:
        self.llm = llm
        self.agents: Dict[str, BaseDataAgent] = agents

    def _determine_datasets(self, question: str) -> List[str]:
        messages = [
            SystemMessage(content=self._ROUTER_PROMPT),
            HumanMessage(content=question),
        ]
        response = self.llm(messages).content.strip()
        if response.lower() == "none":
            return []
        return [name.strip() for name in response.split(",") if name.strip()]

    def ask(self, question: str) -> str:
        dataset_names = self._determine_datasets(question)
        if not dataset_names:
            return "I could not determine which dataset contains the information you need."
        partial_answers: List[Tuple[str, str]] = []
        for name in dataset_names:
            agent = self.agents.get(name)
            if agent is None:
                continue
            answer = agent.run(question)
            partial_answers.append((name, answer))
        if not partial_answers:
            return "No data agents were able to answer your question."
        if len(partial_answers) == 1:
            return partial_answers[0][1]
        answers_text_lines = []
        for name, answer in partial_answers:
            answers_text_lines.append(f"From {name} dataset: {answer}")
        answers_text = "\n".join(answers_text_lines)
        fusion_prompt = self._FUSION_PROMPT_TEMPLATE.format(
            question=question,
            answers=answers_text,
        )
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=fusion_prompt),
        ]
        fused_response = self.llm(messages).content
        return fused_response

    def generate_trends(self) -> str:
        sections = []
        for name, agent in self.agents.items():
            try:
                report = agent.generate_trend_report()
            except Exception as exc:
                report = f"Failed to generate trends for {name}: {exc}"
            sections.append(f"=== {name} Dataset ===\n{report}")
        return "\n\n".join(sections)


def load_csv_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def create_default_agents() -> Tuple[MultiAgentManager, Dict[str, BaseDataAgent]]:
    """Instantiate a MultiAgentManager using packaged example datasets.

    This function locates the CSV files bundled in the ``datasets`` folder and
    loads them into DataFrames. You can copy this approach to load your own
    datasets from arbitrary locations.
    """
    # Locate packaged datasets using importlib.resources
    dataset_dir = importlib.resources.files(__package__).joinpath("datasets")
    datasets = {
        "Supply chain": dataset_dir / "supply_chain.csv",
        "Manufacturing": dataset_dir / "manufacturing.csv",
        "Sales": dataset_dir / "sales.csv",
    }
    dataframes: Dict[str, pd.DataFrame] = {}
    for name, path in datasets.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Expected dataset file '{path}' for '{name}' but it does not exist."
            )
        df = load_csv_dataset(str(path))
        dataframes[name] = df
    # Attempt to read the OpenAI API key from the environment. If it isn't
    # present, try to load it from a .env file located one directory up from
    # this module (i.e. the package root). Users should copy the provided
    # `.env` template and fill in their own key.
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        # Try to load from .env file manually without requiring python‑dotenv
        try:
            from pathlib import Path
            env_path = Path(__file__).resolve().parents[1] / ".env"
            if env_path.exists():
                with env_path.open("r") as f:
                    for line in f:
                        if line.strip().startswith("OPENAI_API_KEY="):
                            key_val = line.strip().split("=", 1)[1]
                            openai_api_key = key_val.strip().strip('"').strip("'")
                            # update environment for downstream users
                            os.environ["OPENAI_API_KEY"] = openai_api_key
                            break
        except Exception:
            pass
    if not openai_api_key:
        raise EnvironmentError(
            "Could not find OPENAI_API_KEY. Please set it in the environment or provide it in a .env file."
        )
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name="gpt-3.5-turbo",
    )
    agents: Dict[str, BaseDataAgent] = {}
    for name, df in dataframes.items():
        agents[name] = BaseDataAgent(name, df, llm)
    manager = MultiAgentManager(llm, agents)
    return manager, agents
