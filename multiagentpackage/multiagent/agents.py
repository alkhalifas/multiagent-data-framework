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
    from langchain.chat_models import ChatOpenAI  # type: ignore
    from langchain.agents import create_pandas_dataframe_agent  # type: ignore
    from langchain.schema import SystemMessage, HumanMessage  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "multiagent.agents requires the 'langchain' and 'openai' packages."
        " Please install them using pip before using this module."
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
        df = self.dataframe.copy()
        report_lines: List[str] = []
        date_col = None
        for col in df.columns:
            if col.lower() == 'date':
                date_col = col
                break
        if date_col is not None:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if date_col is not None and numeric_cols:
            monthly = df.dropna(subset=[date_col])
            if not monthly.empty:
                monthly = monthly.groupby(monthly[date_col].dt.to_period('M'))[numeric_cols].sum()
                if len(monthly) >= 2:
                    last_two = monthly.tail(2)
                    prev, current = last_two.iloc[0], last_two.iloc[1]
                    delta = current - prev
                    report_lines.append("Recent monthly trends:")
                    for col in numeric_cols:
                        change = delta[col]
                        direction = "increased" if change > 0 else ("decreased" if change < 0 else "remained stable")
                        percent = (change / prev[col] * 100) if prev[col] != 0 else None
                        if percent is not None:
                            report_lines.append(
                                f"• {col}: {direction} by {abs(change):.2f} ({abs(percent):.1f}% compared to the previous month)."
                            )
                        else:
                            report_lines.append(
                                f"• {col}: {direction} by {abs(change):.2f}."
                            )
        name = self.name.lower()
        if 'supply' in name:
            if 'supplier_id' in df.columns and 'order_quantity' in df.columns:
                top_supplier = df.groupby('supplier_id')['order_quantity'].sum().idxmax()
                total_orders = df.groupby('supplier_id')['order_quantity'].sum().max()
                report_lines.append(
                    f"Top supplier by order quantity: {top_supplier} with {total_orders} units ordered in total."
                )
            if 'inventory_level' in df.columns:
                avg_inventory = df['inventory_level'].mean()
                report_lines.append(
                    f"Average inventory level across all records: {avg_inventory:.1f} units."
                )
        elif 'manufacturing' in name:
            if 'production_line' in df.columns and 'units_produced' in df.columns:
                top_line = df.groupby('production_line')['units_produced'].sum().idxmax()
                total_units = df.groupby('production_line')['units_produced'].sum().max()
                report_lines.append(
                    f"Production line with highest output: {top_line} producing {total_units} units in total."
                )
            if 'defect_rate' in df.columns:
                avg_defect = df['defect_rate'].mean()
                report_lines.append(
                    f"Average defect rate: {avg_defect:.2%}."
                )
            if 'downtime_hours' in df.columns:
                avg_downtime = df['downtime_hours'].mean()
                report_lines.append(
                    f"Average downtime: {avg_downtime:.1f} hours."
                )
        elif 'sales' in name:
            if 'product' in df.columns and 'revenue' in df.columns:
                top_product = df.groupby('product')['revenue'].sum().idxmax()
                top_rev = df.groupby('product')['revenue'].sum().max()
                report_lines.append(
                    f"Top product by revenue: {top_product} generating ${top_rev:,.2f}."
                )
            if 'region' in df.columns and 'revenue' in df.columns:
                top_region = df.groupby('region')['revenue'].sum().idxmax()
                region_rev = df.groupby('region')['revenue'].sum().max()
                report_lines.append(
                    f"Top region by revenue: {top_region} with total sales of ${region_rev:,.2f}."
                )
            if 'quantity_sold' in df.columns:
                avg_qty = df['quantity_sold'].mean()
                report_lines.append(
                    f"Average quantity sold per order: {avg_qty:.1f} units."
                )
        if not report_lines:
            return "No trends could be derived from this dataset."
        return "\n".join(report_lines)


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
