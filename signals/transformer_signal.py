"""Transformer-based signal generator.

The Transformer signal generator was the breakthrough:
by using Self-Attention on a window of 50 bars × 105 features,
the model learns which historical bars are most relevant for
the current prediction.

Key advantages over LightGBM:
1. Temporal context: "which of the past 50 bars matter NOW?"
2. Long-range dependencies: can attend to support/resistance levels
3. Attention weights are interpretable (which bars got attention?)

Key advantages over LSTM:
1. Parallel processing (no sequential bottleneck)
2. Direct attention to any past bar (no forgetting)
3. +7.6% improvement over LSTM

The signal flow:
    Raw OHLCV → Feature Engineering (105 features)
    → Window (50 bars × 105 features)
    → Transformer (Self-Attention)
    → Score [-1, +1]
    → Execution Engine (SL/TP/Trailing)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from signals.base import SignalGenerator, TradingSignal

# Optional: torch for production use
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TransformerSignalGenerator(SignalGenerator):
    """Generate trading signals using a Transformer model.

    The model takes a window of recent bars with features and
    outputs a directional score via Self-Attention.

    Architecture:
        Input: (window_size=50, n_features=105)
        → Linear projection → Positional Encoding
        → 4× Transformer Encoder (8-head attention)
        → Last position → MLP → tanh → score

    The Self-Attention asks:
    "Given today's market conditions, which of the past 50 bars
     should I pay attention to for my directional prediction?"
    """

    def __init__(
        self,
        model: Any = None,
        window_size: int = 50,
        n_features: int = 105,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        """Initialize.

        Args:
            model: Trained TransformerScoreModel (or None for demo).
            window_size: Number of historical bars per prediction.
            n_features: Number of features per bar.
            feature_mean: Feature means for normalization.
            feature_std: Feature stds for normalization.
        """
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self._buffer: list[np.ndarray] = []

    def generate(self, observation: np.ndarray, **_kwargs: Any) -> TradingSignal:
        """Generate signal from feature observation.

        The observation can be either:
        - A single bar's features (n_features,) → buffered until
          window_size bars are accumulated
        - A full window (window_size, n_features) → used directly

        Args:
            observation: Feature array.

        Returns:
            TradingSignal with score from Transformer.
        """
        if self.model is None:
            return TradingSignal(score=0.0, confidence=0.0)

        # Handle different input shapes
        if observation.ndim == 1:
            self._buffer.append(observation)
            if len(self._buffer) < self.window_size:
                return TradingSignal(
                    score=0.0,
                    confidence=0.0,
                    metadata={"warmup": True},
                )
            window = np.array(self._buffer[-self.window_size :])
        else:
            window = observation

        # Normalize
        if self.feature_mean is not None and self.feature_std is not None:
            window = (window - self.feature_mean) / (self.feature_std + 1e-8)

        # Predict
        score = self._predict(window)
        confidence = abs(score)

        return TradingSignal(
            score=score,
            confidence=confidence,
            metadata={"generator": "transformer"},
        )

    def _predict(self, window: np.ndarray) -> float:
        """Run Transformer inference.

        Args:
            window: (window_size, n_features) array.

        Returns:
            Score in [-1, +1].
        """
        if not HAS_TORCH:
            return 0.0

        with torch.no_grad():
            x = torch.from_numpy(window).float().unsqueeze(0)
            score = self.model(x).item()
        return float(np.clip(score, -1.0, 1.0))

    def generate_from_score(self, raw_score: float) -> TradingSignal:
        """Generate signal from pre-computed score.

        Useful for backtesting with saved Transformer predictions.
        """
        score = float(np.clip(raw_score, -1.0, 1.0))
        return TradingSignal(
            score=score,
            confidence=abs(score),
            metadata={"generator": "transformer"},
        )

    def reset(self) -> None:
        """Clear the observation buffer."""
        self._buffer = []
