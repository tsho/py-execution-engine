"""DQN Q-value to score signal generator.

This is the bridge between the RL model and the Execution Engine.
Instead of letting the DQN directly control position management
(which led to Buy & Hold and Zero-Trade behaviors), we extract
Q-values and convert them to a directional score.

Score calculation:
    score = tanh((Q_BUY - Q_SELL) / temperature)

This maps the raw Q-value difference to a [-1, +1] range:
- score ≈ +1.0: strong buy signal
- score ≈  0.0: no direction (hold)
- score ≈ -1.0: strong sell signal

Confidence is derived from the entropy of the Q-value distribution:
- Low entropy = high confidence (one action clearly dominates)
- High entropy = low confidence (all actions look similar)

This was the key insight: separate "what direction"
from "when to enter/exit".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from signals.base import SignalGenerator, TradingSignal


class DQNScoreSignalGenerator(SignalGenerator):
    """Convert DQN Q-values to directional trading signals.

    The DQN model outputs Q-values for 3 actions:
        Q[0] = HOLD, Q[1] = BUY, Q[2] = SELL

    We convert these to a score and confidence that the
    Execution Engine can use for its own entry/exit logic.

    In production, this wraps a trained DQN model.
    For demonstration, we provide a from_q_values() method.
    """

    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize.

        Args:
            temperature: Controls score sensitivity.
                Low temperature → score saturates at ±1 quickly.
                High temperature → score stays near 0 (less decisive).
        """
        self.temperature = temperature

    def generate(self, observation: np.ndarray, **_kwargs: Any) -> TradingSignal:
        """Generate signal from observation (requires model).

        In production, this would call:
            q_values = model.policy.q_net(obs_to_tensor(observation))

        For the demo, use generate_from_q_values() instead.
        """
        raise NotImplementedError(
            "Use generate_from_q_values() for demonstration, "
            "or set a model for production use."
        )

    def generate_from_q_values(self, q_values: np.ndarray) -> TradingSignal:
        """Generate signal directly from Q-values.

        This is the core algorithm that converts DQN output
        to a trading signal.

        Args:
            q_values: Array of Q-values [Q_HOLD, Q_BUY, Q_SELL].

        Returns:
            TradingSignal with score and confidence.
        """
        score = self._calculate_score(q_values)
        confidence = self._calculate_confidence(q_values)

        return TradingSignal(
            score=score,
            confidence=confidence,
            metadata={
                "q_hold": float(q_values[self.HOLD]),
                "q_buy": float(q_values[self.BUY]),
                "q_sell": float(q_values[self.SELL]),
            },
        )

    def _calculate_score(self, q_values: np.ndarray) -> float:
        """Calculate directional score from Q-values.

        score = tanh((Q_BUY - Q_SELL) / temperature)

        Why tanh?
        - Maps any Q-value difference to [-1, +1]
        - Smooth and differentiable
        - Saturates for large differences (strong conviction)

        Args:
            q_values: Q-values [HOLD, BUY, SELL].

        Returns:
            Score in [-1, +1].
        """
        q_buy = q_values[self.BUY]
        q_sell = q_values[self.SELL]
        diff = q_buy - q_sell
        scaled_diff = diff / self.temperature
        return float(np.tanh(scaled_diff))

    def _calculate_confidence(self, q_values: np.ndarray) -> float:
        """Calculate confidence from Q-value distribution entropy.

        High confidence = one action clearly dominates (low entropy).
        Low confidence = all actions look similar (high entropy).

        Args:
            q_values: Q-values array.

        Returns:
            Confidence in [0, 1].
        """
        # Softmax with numerical stability
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)

        # Entropy (0 = certain, log(3) ≈ 1.099 = uniform)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(q_values))

        # Invert: high entropy → low confidence
        return float(1.0 - entropy / max_entropy)

    def reset(self) -> None:
        """No state to reset."""
