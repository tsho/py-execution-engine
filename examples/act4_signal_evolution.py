"""Act 3-2: Signal Generator Evolution - LightGBM to Transformer.

With the signal-agnostic architecture in place (Act 3-1), we could
now focus purely on improving the signal quality. The execution
engine handles positions - we just need better predictions.

Evolution of signal generators:

1. DQN Score:
   - Q-values → tanh → score
   - Problem: Q-values were too similar, weak signals

2. LightGBM:
   - 105 engineered features → predict_proba → score
   - Fast training, built-in feature importance
   - Problem: each bar is independent (no temporal context)
   - AUC 0.52 (barely above random chance)

3. Transformer:
   - 50 bars × 105 features → Self-Attention → score
   - Breakthrough: "which of the past 50 bars matter NOW?"
   - Can attend to support/resistance levels from 20 bars ago
   - +7.6% improvement over LSTM

The key advantage of Self-Attention:

    LightGBM sees:  [bar_now] → prediction
    Transformer sees: [bar_1, bar_2, ..., bar_50] → attention → prediction

    The Transformer asks: "Given today's conditions, which of
    these 50 historical bars should I pay attention to?"

Usage::

    python examples/act4_signal_evolution.py

"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signals.lgbm_signal import LGBMSignalGenerator

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def demonstrate_signal_evolution() -> None:
    """Show the progression of signal generators."""
    logger.info("=" * 70)
    logger.info("Act 4: Signal Generator Evolution")
    logger.info("=" * 70)
    logger.info("")
    logger.info("With signal-agnostic execution (Act 3), we can now")
    logger.info("focus purely on improving signal quality.")
    logger.info("")

    logger.info("Evolution Timeline:")
    logger.info("")
    logger.info("  Step 1 │ DQN Score")
    logger.info("         │ Q-values → tanh → score")
    logger.info("         │ Problem: weak signals (Q-values too similar)")
    logger.info("         │")
    logger.info("  Step 2 │ LightGBM")
    logger.info("         │ 105 features → predict_proba → score")
    logger.info("         │ Problem: no temporal context (AUC 0.52)")
    logger.info("         │")
    logger.info("  Step 3 │ Transformer (breakthrough)")
    logger.info("         │ 50 bars × 105 features → Self-Attention → score")
    logger.info("         │ +7.6% improvement over LSTM")
    logger.info("")


def demonstrate_lgbm_limitation() -> None:
    """Show why LightGBM struggles without temporal context."""
    logger.info("=" * 70)
    logger.info("LightGBM: The Temporal Context Problem")
    logger.info("=" * 70)
    logger.info("")
    logger.info("LightGBM treats each bar independently:")
    logger.info("")
    logger.info("  Bar 1: [RSI=70, MACD=0.5, BB=0.8, ...] → P(up)=0.55")
    logger.info("  Bar 2: [RSI=65, MACD=0.3, BB=0.6, ...] → P(up)=0.52")
    logger.info("  Bar 3: [RSI=72, MACD=0.6, BB=0.9, ...] → P(up)=0.57")
    logger.info("")
    logger.info("  Each prediction is made in isolation.")
    logger.info("  It can't see that RSI has been rising for 3 bars,")
    logger.info("  or that we bounced off support 10 bars ago.")
    logger.info("")

    # Demonstrate with the LGBMSignalGenerator (no model = returns 0)
    lgbm = LGBMSignalGenerator(
        model=None,
        buy_threshold=0.6,
        sell_threshold=0.6,
    )

    # Show probability-to-score mapping
    logger.info("Probability → Score mapping (threshold=0.6):")
    logger.info("")
    logger.info(f"  {'P(buy)':>8} {'Score':>8} {'Signal':>10}")
    logger.info("  " + "-" * 30)

    for prob in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        signal = lgbm.generate_from_proba(prob)
        if signal.score > 0:
            label = "BUY"
        elif signal.score < 0:
            label = "SELL"
        else:
            label = "(no signal)"
        logger.info(f"  {prob:>8.2f} {signal.score:>+8.3f} {label:>10}")

    logger.info("")
    logger.info("  The 'dead zone' (0.4-0.6) means: 'I'm not confident enough.'")
    logger.info("  With AUC=0.52, most predictions fall in this zone.")
    logger.info("  Result: very few signals, and those few are unreliable.")


def demonstrate_transformer_attention() -> None:
    """Show how Transformer Self-Attention works for trading."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Transformer: Self-Attention for Time Series")
    logger.info("=" * 70)
    logger.info("")
    logger.info("The Transformer sees a WINDOW of 50 bars:")
    logger.info("")
    logger.info("  Input: (50 bars × 105 features)")
    logger.info("")
    logger.info("  ┌─────────────────────────────────────────────┐")
    logger.info("  │  bar_1   bar_2   ...  bar_49  bar_50        │")
    logger.info("  │  [105]   [105]        [105]   [105]         │")
    logger.info("  │    ↓       ↓            ↓       ↓           │")
    logger.info("  │  ┌─────────────────────────────────────┐    │")
    logger.info("  │  │        Self-Attention                │    │")
    logger.info("  │  │                                      │    │")
    logger.info("  │  │  'Which past bars matter for NOW?'   │    │")
    logger.info("  │  │                                      │    │")
    logger.info("  │  │  Attention weights:                   │    │")
    logger.info("  │  │  bar_1: 0.01  (irrelevant)           │    │")
    logger.info("  │  │  bar_15: 0.15 (support level!)       │    │")
    logger.info("  │  │  bar_30: 0.08 (resistance break)     │    │")
    logger.info("  │  │  bar_50: 0.25 (current context)      │    │")
    logger.info("  │  └─────────────────────────────────────┘    │")
    logger.info("  │                    ↓                        │")
    logger.info("  │              score: +0.73                   │")
    logger.info("  │              (strong buy signal)            │")
    logger.info("  └─────────────────────────────────────────────┘")
    logger.info("")

    # Architecture details
    logger.info("Architecture Details:")
    logger.info("")
    logger.info("  1. Linear Projection:  105 features → 128 dimensions")
    logger.info("  2. Positional Encoding: sinusoidal (temporal order)")
    logger.info("  3. Transformer Encoder: 4 layers × 8 attention heads")
    logger.info("  4. Last Position:       take bar_50's representation")
    logger.info("  5. Regressor:           128 → 64 → 1 → tanh")
    logger.info("  6. Output:              score in [-1, +1]")
    logger.info("")

    # Show the 105 features
    logger.info("The 105 Features (per bar):")
    logger.info("")
    logger.info("  Price-based (20):  returns, log returns, ATR ratios, ...")
    logger.info("  Momentum (25):     RSI, MACD, Stochastic, ROC, ...")
    logger.info("  Trend (15):        SMA ratios, EMA ratios, ADX, ...")
    logger.info("  Volatility (15):   Bollinger Band ratios, ATR, ...")
    logger.info("  Volume (10):       volume ratios, OBV, VWAP, ...")
    logger.info("  Microstructure (10): spread, tick direction, ...")
    logger.info("  Calendar (10):     hour, day of week, session, ...")


def demonstrate_comparison() -> None:
    """Side-by-side comparison of signal generators."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Side-by-Side Comparison")
    logger.info("=" * 70)
    logger.info("")

    headers = ["Aspect", "DQN Score", "LightGBM", "Transformer"]
    rows = [
        [
            "Input",
            "Q-values (3)",
            "Features (105)",
            "Window (50×105)",
        ],
        [
            "Temporal context",
            "None (single Q)",
            "None (single bar)",
            "50 bars (attention)",
        ],
        [
            "Key mechanism",
            "Q_BUY - Q_SELL",
            "predict_proba",
            "Self-Attention",
        ],
        [
            "Score range",
            "[-1, +1] (tanh)",
            "[-1, +1] (scaled)",
            "[-1, +1] (tanh)",
        ],
        [
            "Calibration",
            "Poor (Q-scale)",
            "Poor (AUC 0.52)",
            "Moderate",
        ],
        [
            "Training speed",
            "Slow (RL)",
            "Fast (GBDT)",
            "Moderate (GPU)",
        ],
        [
            "Interpretability",
            "Q-value gaps",
            "Feature importance",
            "Attention weights",
        ],
        [
            "Best for",
            "RL experiments",
            "Feature analysis",
            "Production signals",
        ],
    ]

    # Calculate column widths
    col_widths = [max(len(r[i]) for r in [headers] + rows) for i in range(4)]

    # Print table
    header_line = " | ".join(
        h.ljust(w) for h, w in zip(headers, col_widths, strict=True)
    )
    logger.info(f"  {header_line}")
    logger.info("  " + "-+-".join("-" * w for w in col_widths))

    for row in rows:
        line = " | ".join(
            cell.ljust(w) for cell, w in zip(row, col_widths, strict=True)
        )
        logger.info(f"  {line}")

    logger.info("")


def demonstrate_why_attention_matters() -> None:
    """Explain why Self-Attention is the breakthrough."""
    logger.info("=" * 70)
    logger.info("Why Self-Attention Was the Breakthrough")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Consider this market scenario:")
    logger.info("")
    logger.info("  Bar 15: Price hits 149.50 (support level)")
    logger.info("  Bar 16-29: Price rises to 150.80")
    logger.info("  Bar 30: Price breaks above 151.00 (resistance)")
    logger.info("  Bar 31-49: Price consolidates around 150.90")
    logger.info("  Bar 50: Price drops to 150.50 → NOW WHAT?")
    logger.info("")
    logger.info("")
    logger.info("  LightGBM (bar 50 only):")
    logger.info("    Sees: RSI=45, MACD=-0.1, Price=150.50")
    logger.info("    → 'Slightly bearish' (no context)")
    logger.info("")
    logger.info("  Transformer (bars 1-50):")
    logger.info("    Attends to bar 15 (support at 149.50)")
    logger.info("    Attends to bar 30 (resistance break at 151.00)")
    logger.info("    → 'Price is pulling back to support after breakout'")
    logger.info("    → score = +0.6 (buy the dip)")
    logger.info("")
    logger.info("")
    logger.info("The Transformer's advantage:")
    logger.info("")
    logger.info("  1. TEMPORAL CONTEXT: 'which of the past 50 bars matter NOW?'")
    logger.info("  2. LONG-RANGE DEPENDENCIES: attend to support from 35 bars ago")
    logger.info("  3. PARALLEL PROCESSING: no sequential bottleneck (unlike LSTM)")
    logger.info("  4. INTERPRETABLE: attention weights show what the model 'sees'")
    logger.info("")
    logger.info("")
    logger.info("This was the key insight that led to the signal-agnostic")
    logger.info("architecture paying off: once we had a good execution engine,")
    logger.info("we could focus entirely on improving the signal quality.")
    logger.info("The Transformer was the signal breakthrough.")


if __name__ == "__main__":
    setup_logging()
    demonstrate_signal_evolution()
    demonstrate_lgbm_limitation()
    demonstrate_transformer_attention()
    demonstrate_comparison()
    demonstrate_why_attention_matters()
