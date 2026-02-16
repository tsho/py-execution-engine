"""Act 3-1: Architecture Separation - The Execution Engine.

After the reward design failures of Act 2, we had a breakthrough
insight: separate signal generation from position management.

The DQN was failing because it was trying to simultaneously:
1. Learn WHAT direction the market will move (signal)
2. Learn WHEN to enter/exit trades (execution)
3. Manage risk (stop loss, take profit, position sizing)

These are fundamentally different problems. A market prediction
model shouldn't need to know about stop losses. And a risk
manager shouldn't need to predict price direction.

The solution: the Signal-Agnostic Execution Engine.

    Signal Generator → score [-1, +1] → Execution Engine
         ↓                                    ↓
    "Is the market              "Given this score, should I
     going up or down?"          enter? Where's my SL/TP?
                                 How big should the position be?"

This example shows:
1. How DQN Q-values are converted to a directional score
2. How the Execution Engine uses that score independently
3. Why this separation makes each component testable

Usage::

    python examples/act3_execution_engine.py

"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.dqn_score import DQNScoreSignalGenerator

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def demonstrate_q_value_to_score() -> None:
    """Show how DQN Q-values become directional scores."""
    logger.info("=" * 70)
    logger.info("Act 3: Architecture Separation - DQN Score Conversion")
    logger.info("=" * 70)
    logger.info("")
    logger.info("The key insight: extract a SCORE from the DQN,")
    logger.info("then let a separate Execution Engine handle everything else.")
    logger.info("")
    logger.info("  score = tanh((Q_BUY - Q_SELL) / temperature)")
    logger.info("")

    signal_gen = DQNScoreSignalGenerator(temperature=1.0)

    # Scenarios from real DQN training
    scenarios = [
        (
            "Strong BUY signal",
            np.array([0.5, 2.1, -0.3]),
            "Q_BUY >> Q_SELL → score ≈ +1",
        ),
        (
            "Weak BUY signal",
            np.array([0.5, 0.7, 0.4]),
            "Q_BUY slightly > Q_SELL → small positive",
        ),
        (
            "No signal (uncertain)",
            np.array([0.5, 0.5, 0.5]),
            "All Q-values equal → score ≈ 0",
        ),
        (
            "Weak SELL signal",
            np.array([0.5, 0.4, 0.7]),
            "Q_SELL slightly > Q_BUY → small negative",
        ),
        (
            "Strong SELL signal",
            np.array([0.5, -0.3, 2.1]),
            "Q_SELL >> Q_BUY → score ≈ -1",
        ),
        (
            "DQN output (broken)",
            np.array([0.42, 0.45, 0.38]),
            "All similar, BUY slightly higher → always long!",
        ),
    ]

    logger.info("-" * 70)
    logger.info(
        f"{'Scenario':<25} {'Q_HOLD':>7} {'Q_BUY':>7} "
        f"{'Q_SELL':>7} {'Score':>7} {'Conf':>6}"
    )
    logger.info("-" * 70)

    for name, q_values, _explanation in scenarios:
        signal = signal_gen.generate_from_q_values(q_values)
        logger.info(
            f"{name:<25} {q_values[0]:>+7.2f} {q_values[1]:>+7.2f} "
            f"{q_values[2]:>+7.2f} {signal.score:>+7.3f} "
            f"{signal.confidence:>5.3f}"
        )

    logger.info("-" * 70)
    logger.info("")

    logger.info("Notice the 'DQN output (broken)' row:")
    logger.info("  Q-values are nearly identical (0.42, 0.45, 0.38)")
    logger.info("  But Q_BUY is always slightly higher → score ≈ +0.07")
    logger.info("  The Execution Engine interprets this as a perpetual BUY signal.")
    logger.info("")
    logger.info("  This is how 'Always Long' behavior emerges from a DQN")
    logger.info("  that hasn't actually learned anything useful.")


def demonstrate_architecture_separation() -> None:
    """Show the clean separation between signal and execution."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Architecture Separation")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Before: End-to-end DQN")
    logger.info("")
    logger.info("  ┌─────────────────────────────────────┐")
    logger.info("  │           DQN (monolithic)           │")
    logger.info("  │                                      │")
    logger.info("  │  observation → Q-values → ACTION     │")
    logger.info("  │                           ↓          │")
    logger.info("  │              BUY / SELL / HOLD        │")
    logger.info("  │              (direct control)         │")
    logger.info("  └─────────────────────────────────────┘")
    logger.info("")
    logger.info("  Problems:")
    logger.info("  - BUY means 'open long AND hold AND manage risk'")
    logger.info("  - Can't separate 'good prediction' from 'good execution'")
    logger.info("  - Reward engineering nightmare (Act 2)")
    logger.info("")
    logger.info("")
    logger.info("After: Signal-Agnostic Architecture")
    logger.info("")
    logger.info("  ┌──────────────┐   score    ┌──────────────────┐")
    logger.info("  │    Signal     │──[-1,+1]──>│ Execution Engine │")
    logger.info("  │   Generator   │            │                  │")
    logger.info("  │              │            │  - Entry logic    │")
    logger.info("  │  (pluggable) │   conf     │  - Stop loss     │")
    logger.info("  │  DQN / LGBM / │──[0,1]───>│  - Take profit   │")
    logger.info("  │  Transformer  │            │  - Trailing stop │")
    logger.info("  │  / SMA / ...  │            │  - Position size │")
    logger.info("  └──────────────┘            └──────────────────┘")
    logger.info("")
    logger.info("  Benefits:")
    logger.info("  - Signal generator only answers: 'which direction?'")
    logger.info("  - Execution engine handles all position management")
    logger.info("  - Each component is independently testable")
    logger.info("  - Can swap signal source without changing execution")


def demonstrate_signal_flow() -> None:
    """Show the signal flow through the system."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Signal Flow Example")
    logger.info("=" * 70)
    logger.info("")

    signal_gen = DQNScoreSignalGenerator(temperature=1.0)

    # Simulate 10 bars of DQN output
    np.random.seed(42)
    logger.info("Simulating 10 bars of DQN output:")
    logger.info("")
    logger.info(
        f"{'Bar':>4} {'Q_HOLD':>8} {'Q_BUY':>8} {'Q_SELL':>8} "
        f"{'Score':>8} {'Conf':>6} {'Action':>10}"
    )
    logger.info("-" * 60)

    entry_threshold = 0.05

    for i in range(10):
        # Simulate Q-values with some noise
        base = np.random.normal(0.5, 0.3, 3)
        q_values = base.astype(np.float64)

        signal = signal_gen.generate_from_q_values(q_values)

        # Execution engine decision (simplified)
        if signal.score > entry_threshold:
            action = "→ LONG"
        elif signal.score < -entry_threshold:
            action = "→ SHORT"
        else:
            action = "  (wait)"

        logger.info(
            f"{i:>4} {q_values[0]:>+8.3f} {q_values[1]:>+8.3f} "
            f"{q_values[2]:>+8.3f} {signal.score:>+8.4f} "
            f"{signal.confidence:>5.3f} {action:>10}"
        )

    logger.info("")
    logger.info("The Execution Engine receives only the score and confidence.")
    logger.info("It doesn't know or care whether the signal came from a DQN,")
    logger.info("a Transformer, or a coin flip.")
    logger.info("")
    logger.info("This is the 'signal-agnostic' principle:")
    logger.info("  The engine's job is to EXECUTE well,")
    logger.info("  not to PREDICT well.")


if __name__ == "__main__":
    setup_logging()
    demonstrate_q_value_to_score()
    demonstrate_architecture_separation()
    demonstrate_signal_flow()
