"""Base signal interfaces for Execution Engine.

This module provides the abstract SignalGenerator and TradingSignal
dataclass. Any signal source can be integrated by implementing
the SignalGenerator interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class PositionSide(IntEnum):
    """Position side (duplicated here to avoid circular imports)."""

    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class TradingSignal:
    """Trading signal with directional score and confidence.

    Attributes:
        score: Directional score from -1.0 (strong sell) to +1.0 (strong buy).
        confidence: Confidence level from 0.0 to 1.0.
        metadata: Additional signal metadata.
    """

    score: float
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None

    def get_direction(self) -> PositionSide:
        """Get signal direction as PositionSide.

        Returns:
            LONG if score > 0, SHORT if score < 0, FLAT otherwise.
        """
        if self.score > 0:
            return PositionSide.LONG
        if self.score < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    def get_strength(self) -> float:
        """Get absolute signal strength (0.0 to 1.0)."""
        return abs(self.score)

    def weighted_score(self) -> float:
        """Get confidence-weighted score."""
        return self.score * self.confidence


class SignalGenerator(ABC):
    """Abstract base class for signal generators.

    Any signal source (DQN model, technical indicator, random baseline)
    can be integrated by implementing ``generate()`` and ``reset()``.

    Example::

        class MySMASignal(SignalGenerator):
            def generate(self, observation, **kwargs):
                sma_fast, sma_slow = observation[-2], observation[-1]
                score = 1.0 if sma_fast > sma_slow else -1.0
                return TradingSignal(score=score)

            def reset(self):
                pass
    """

    @abstractmethod
    def generate(self, observation: np.ndarray, **kwargs: Any) -> TradingSignal:
        """Generate a trading signal from observation.

        Args:
            observation: Input data (model features, price data, etc.).
            **kwargs: Additional arguments.

        Returns:
            TradingSignal with score and confidence.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset generator state."""
