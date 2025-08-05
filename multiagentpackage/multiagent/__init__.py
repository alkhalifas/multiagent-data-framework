"""Public API for the multiagent package.

This package exposes a set of classes and helper functions for working with
tabular data via large language models. See :mod:`multiagent.agents` for
implementation details.
"""

from .agents import BaseDataAgent, MultiAgentManager, create_default_agents  # noqa: F401

__all__ = [
    "BaseDataAgent",
    "MultiAgentManager",
    "create_default_agents",
]