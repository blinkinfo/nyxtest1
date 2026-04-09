from __future__ import annotations

from core.strategies.base import BaseStrategy
from core.strategies.pattern_strategy import PatternStrategy
from core.strategies.ml_strategy import MLStrategy


def get_strategy(name: str) -> BaseStrategy:
    """Return a strategy instance by name."""
    registry = {
        "pattern": PatternStrategy,
        "ml": MLStrategy,
    }
    cls = registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(registry.keys())}")
    return cls()
