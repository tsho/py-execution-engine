"""SMA crossover signal generator.

A classic technical analysis baseline: go long when the fast SMA
crosses above the slow SMA, and go short when it crosses below.

This demonstrates how a simple rule-based strategy plugs into
the signal-agnostic execution framework.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from signals.base import SignalGenerator, TradingSignal


class SMACrossSignalGenerator(SignalGenerator):
    """SMA crossover signal generator.

    Computes fast and slow simple moving averages from incoming
    close prices and generates directional signals based on their
    relative position.

    Signal scoring:
    - score = +1.0 when fast SMA > slow SMA (bullish)
    - score = -1.0 when fast SMA < slow SMA (bearish)
    - Confidence is proportional to the spread between the two SMAs.

    Attributes:
        fast_period: Period for the fast SMA.
        slow_period: Period for the slow SMA.
        prices: Rolling window of close prices.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
    ) -> None:
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prices: deque[float] = deque(maxlen=slow_period)

    def generate(self, observation: np.ndarray, **_kwargs: Any) -> TradingSignal:
        """Generate signal from observation.

        The observation is expected to contain the close price as
        the last element (or a single scalar).

        Args:
            observation: Array where the last element is the close price.
            **_kwargs: Unused.

        Returns:
            TradingSignal based on SMA crossover.
        """
        close = float(observation[-1]) if observation.ndim > 0 else float(observation)
        self.prices.append(close)

        if len(self.prices) < self.slow_period:
            return TradingSignal(
                score=0.0,
                confidence=0.0,
                metadata={"generator": "sma_cross", "warmup": True},
            )

        prices_arr = np.array(self.prices)
        fast_sma = float(prices_arr[-self.fast_period :].mean())
        slow_sma = float(prices_arr[-self.slow_period :].mean())

        spread = (fast_sma - slow_sma) / slow_sma
        score = 1.0 if fast_sma > slow_sma else -1.0

        confidence = min(abs(spread) * 100, 1.0)

        return TradingSignal(
            score=score,
            confidence=confidence,
            metadata={
                "generator": "sma_cross",
                "fast_sma": fast_sma,
                "slow_sma": slow_sma,
                "spread": spread,
            },
        )

    def reset(self) -> None:
        """Clear price history."""
        self.prices.clear()
