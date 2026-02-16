"""Transformer model for FX return prediction.

Architecture: Input (batch, window_size=50, n_features=105)
    → Linear projection (n_features → d_model=128)
    → Positional Encoding
    → Transformer Encoder (4 layers, 8 heads)
    → Regressor head (128 → 64 → 1)

The Self-Attention mechanism learns which of the past 50 bars
are most relevant for the current prediction. This captures
long-range dependencies (e.g., support/resistance levels from
20 bars ago) that LightGBM and LSTMs struggle with.

This model was the breakthrough:
- First Transformer → +7.6% improvement over LSTM
- Deeper (4 layers) → First 54% win rate
- LONG-only return prediction with FEP targets
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer.

    Adds positional information to the input embeddings so the
    Transformer knows the temporal order of the bars.

    For a window of 50 bars at 5-min intervals:
    - Position 0 = oldest bar (4h10m ago)
    - Position 49 = most recent bar (current)
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Position-encoded tensor.
        """
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransformerScoreModel(nn.Module):
    """Transformer for directional score prediction.

    Input:  (batch, window_size, n_features)
            e.g., (32, 50, 105) = 50 bars × 105 features

    Output: (batch,) score in range [-1, +1]

    Architecture:
    1. Linear projection: n_features → d_model
    2. Positional encoding: sinusoidal
    3. Transformer encoder: self-attention layers
    4. Take last position output (most recent bar)
    5. Regressor: d_model → 64 → 1 → tanh

    The Self-Attention mechanism answers:
    "Given the current market context, which of the past 50 bars
     should I pay attention to for my prediction?"
    """

    def __init__(
        self,
        n_features: int = 105,
        window_size: int = 50,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features

        # Project raw features to model dimension
        self.input_projection = nn.Linear(n_features, d_model)

        # Add temporal position information
        self.pos_encoding = PositionalEncoding(
            d_model, max_len=window_size + 10, dropout=dropout
        )

        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Predict score from the last position's representation
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output in [-1, +1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Can be:
               - (batch, window_size, n_features) - 3D
               - (batch, window_size * n_features) - 2D (flattened)

        Returns:
            Score tensor (batch,) in range [-1, +1].
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], self.window_size, self.n_features)

        # Project features: (batch, 50, 105) → (batch, 50, 128)
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Self-attention: each bar attends to all other bars
        # This is where the model learns temporal patterns
        x = self.transformer(x)

        # Take the last position's output (most recent bar)
        # This aggregates information from all attended positions
        last_hidden = x[:, -1, :]  # (batch, d_model)

        # Predict directional score
        return self.regressor(last_hidden).squeeze(-1)
