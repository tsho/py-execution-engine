"""LightGBM-based signal generator.

LightGBM was tested as an alternative to DQN for signal generation.
It excels at learning from engineered features (105 indicators),
but has fundamental limitations for time-series trading:

Strengths:
- Fast training and inference
- Handles heterogeneous features well
- Built-in feature importance

Limitations:
- No temporal context: each prediction is independent
- Poor probability calibration: predict_proba ≠ true probability
- Can't learn "which past bars matter" (no attention mechanism)

Result: AUC 0.52 (barely above random chance)
→ Led to exploring Transformer architecture instead

This module shows how LightGBM plugs into the signal-agnostic
framework, and why it was eventually replaced.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from signals.base import SignalGenerator, TradingSignal


class LGBMSignalGenerator(SignalGenerator):
    """LightGBM-based signal generator.

    Wraps a trained LightGBM model to generate trading signals.
    The model predicts the probability of a profitable trade,
    which is converted to a directional score.

    This was used for SHORT signal classification:
    - label=1: price drops significantly (SELL)
    - label=0: price rises significantly (not-SELL)

    The score mapping:
    - prob(SELL) > threshold → negative score (sell signal)
    - prob(BUY) > threshold → positive score (buy signal)
    """

    def __init__(
        self,
        model: Any = None,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.6,
        n_features: int = 47,
    ) -> None:
        """Initialize.

        Args:
            model: Trained LightGBM model (or None for demo).
            buy_threshold: Probability threshold for buy signal.
            sell_threshold: Probability threshold for sell signal.
            n_features: Number of input features.
        """
        self.model = model
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.n_features = n_features

    def generate(self, observation: np.ndarray, **_kwargs: Any) -> TradingSignal:
        """Generate signal from features.

        LightGBM treats each observation independently -
        it cannot consider the sequence of past bars.
        This is its fundamental limitation.

        Args:
            observation: Feature vector (n_features,).
                Unlike Transformer which takes (window_size, n_features),
                LightGBM takes a flat vector for a single point in time.

        Returns:
            TradingSignal with calibrated score.
        """
        if self.model is None:
            return TradingSignal(score=0.0, confidence=0.0)

        # LightGBM predict_proba returns [P(class_0), P(class_1)]
        features = observation.reshape(1, -1)
        proba = self.model.predict_proba(features)[0]

        # Convert probability to directional score
        buy_prob = proba[1]  # P(price goes up)
        score = self._proba_to_score(buy_prob)
        confidence = abs(buy_prob - 0.5) * 2  # Distance from uncertain

        return TradingSignal(
            score=score,
            confidence=confidence,
            metadata={
                "generator": "lgbm",
                "buy_prob": float(buy_prob),
            },
        )

    def _proba_to_score(self, buy_prob: float) -> float:
        """Convert prediction probability to directional score.

        The key problem: LightGBM's predict_proba is poorly
        calibrated for trading. A predicted P=0.65 might only
        mean P_true=0.52 in practice.

        Args:
            buy_prob: Predicted probability of price increase.

        Returns:
            Score in [-1, +1].
        """
        if buy_prob > self.buy_threshold:
            # Scale from [threshold, 1.0] → [0, 1.0]
            return (buy_prob - self.buy_threshold) / (1.0 - self.buy_threshold)
        if buy_prob < (1.0 - self.sell_threshold):
            # Scale from [0, 1-threshold] → [-1.0, 0]
            return -(
                (1.0 - self.sell_threshold - buy_prob) / (1.0 - self.sell_threshold)
            )
        return 0.0  # No signal (uncertain zone)

    def generate_from_proba(self, buy_prob: float) -> TradingSignal:
        """Generate signal from pre-computed probability.

        Useful for backtesting with saved predictions.
        """
        score = self._proba_to_score(buy_prob)
        confidence = abs(buy_prob - 0.5) * 2
        return TradingSignal(
            score=score,
            confidence=confidence,
            metadata={"generator": "lgbm", "buy_prob": buy_prob},
        )

    def reset(self) -> None:
        """No state to reset."""
