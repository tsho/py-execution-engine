"""Random signal generator (baseline).

This generator produces random buy/sell signals with uniform distribution.
It serves as a baseline to verify that the execution engine doesn't
generate profit from random noise.

A well-designed execution engine should produce ~zero expected PnL
with random signals (minus transaction costs).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from signals.base import SignalGenerator, TradingSignal


class RandomSignalGenerator(SignalGenerator):
    """Generates random trading signals.

    Useful as a baseline for validating that:
    1. The execution engine is not biased
    2. Transaction costs are properly accounted for
    3. Backtest results from real signals are statistically significant

    Attributes:
        rng: Random number generator.
        signal_probability: Probability of generating a non-zero signal.
    """

    def __init__(
        self,
        seed: int | None = None,
        signal_probability: float = 0.1,
    ) -> None:
        """Initialize RandomSignalGenerator.

        Args:
            seed: Random seed for reproducibility.
            signal_probability: Probability of generating a signal per bar.
        """
        self.rng = np.random.default_rng(seed)
        self.signal_probability = signal_probability

    def generate(self, _observation: np.ndarray, **_kwargs: Any) -> TradingSignal:
        """Generate a random trading signal.

        With probability ``signal_probability``, generates a signal
        with score uniformly drawn from [-1, +1]. Otherwise returns
        score=0 (no signal).
        """
        if self.rng.random() < self.signal_probability:
            score = float(self.rng.uniform(-1.0, 1.0))
            confidence = abs(score)
        else:
            score = 0.0
            confidence = 0.0

        return TradingSignal(
            score=score,
            confidence=confidence,
            metadata={"generator": "random"},
        )

    def reset(self) -> None:
        """Reset is a no-op for random generator."""
