"""Example: Run backtests with different signal generators.

This script demonstrates the signal-agnostic execution framework
by comparing three signal sources on the same price data:

1. Random signals (baseline)
2. SMA crossover (classic technical analysis)
3. "Always-long" buy-and-hold (simulating a broken DQN model)

Background
----------
In our first attempt at FX trading with Deep RL (DQN), the model
learned to always output BUY. It looked profitable at first glance
because USD/JPY trended upward during the training period.

The model wasn't predicting the market - it was exploiting the trend
in the training data. This is equivalent to a buy-and-hold strategy
with extra steps.

This example shows how the execution framework helps you detect this:
- Separate the signal source from execution logic
- Compare against simple baselines (random, SMA)
- Verify that "alpha" isn't just a lucky trend

Usage::

    python examples/act1_buy_and_hold_trap.py

"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.config import (
    ExecutionConfig,
    RiskConfig,
    SignalThresholdConfig,
    StopLossConfig,
    StopLossType,
    TakeProfitConfig,
    TakeProfitType,
    TimeExitConfig,
    TrailingStopConfig,
)
from execution.engine import ExecutionEngine
from signals.base import SignalGenerator, TradingSignal
from signals.dummy_random import RandomSignalGenerator
from signals.dummy_sma_cross import SMACrossSignalGenerator

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


# ---------------------------------------------------------------------------
# "Always Long" signal - simulates what our initial DQN actually learned
# ---------------------------------------------------------------------------


class AlwaysLongSignalGenerator(SignalGenerator):
    """A signal generator that always outputs BUY.

    This simulates what our initial DQN model actually learned:
    it output nearly identical Q-values for all actions, but
    Q(BUY) was always slightly higher than Q(SELL) and Q(HOLD).

    When passed through tanh((Q_buy - Q_sell) / temperature),
    the score was always a small positive number (~0.05-0.08),
    resulting in a perpetual LONG position.
    """

    def __init__(self, score: float = 0.06) -> None:
        self.score = score

    def generate(self, _observation: np.ndarray, **_kwargs: Any) -> TradingSignal:
        return TradingSignal(
            score=self.score,
            confidence=0.5,
            metadata={"generator": "always_long"},
        )

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_usdjpy(
    n_bars: int = 10000,
    seed: int = 42,
    trend: float = 0.00001,
) -> pd.DataFrame:
    """Generate synthetic USD/JPY 5-min bar data.

    Creates realistic-looking OHLCV data with configurable trend.

    Args:
        n_bars: Number of 5-minute bars to generate.
        seed: Random seed.
        trend: Per-bar drift (positive = uptrend).

    Returns:
        DataFrame with OHLCV + ATR columns.
    """
    rng = np.random.default_rng(seed)

    base_price = 150.0
    volatility = 0.0003  # ~30 pips per day at 5-min bars

    # Generate returns with slight trend
    returns = rng.normal(trend, volatility, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    # Realistic 5-min bar: ~5-15 pip range (0.05-0.15 JPY at 150)
    high_spread = rng.uniform(0.0002, 0.0006, n_bars)  # 3-9 pips above close
    low_spread = rng.uniform(0.0002, 0.0006, n_bars)  # 3-9 pips below close

    timestamps = pd.date_range("2024-01-02 00:00", periods=n_bars, freq="5min")

    df = pd.DataFrame(
        {
            "open": np.roll(prices, 1),
            "high": prices * (1 + high_spread),
            "low": prices * (1 - low_spread),
            "close": prices,
            "volume": rng.uniform(100, 1000, n_bars),
        },
        index=timestamps,
    )
    df.iloc[0, df.columns.get_loc("open")] = base_price

    # Calculate ATR (14-period)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df = df.dropna()

    return df


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------


def run_single_backtest(
    name: str,
    signal_gen: SignalGenerator,
    df: pd.DataFrame,
    config: ExecutionConfig,
    risk_config: RiskConfig,
) -> dict[str, Any]:
    """Run backtest and return summary stats."""
    engine = ExecutionEngine(
        signal_generator=signal_gen,
        execution_config=config,
        risk_config=risk_config,
        transaction_cost=0.0003,  # 3 pips spread
    )

    # Generate observations (just close prices for signal generators)
    observations = df["close"].values.reshape(-1, 1)

    engine.run_backtest(
        df,
        observations=observations,
        verbose=False,
    )

    stats = engine.get_trade_stats()
    risk_stats = engine.get_risk_stats()

    return {
        "name": name,
        "total_trades": stats.total_trades,
        "win_rate": stats.win_rate,
        "total_pnl_pips": stats.total_pnl_pips,
        "profit_factor": stats.profit_factor,
        "total_return": risk_stats["total_return"],
        "max_drawdown": risk_stats["current_drawdown"],
        "exit_reasons": stats.trades_by_exit_reason,
    }


def main() -> None:
    """Run comparison backtests."""
    setup_logging()
    logger.info("=" * 70)
    logger.info("Signal-Agnostic Execution Framework - Backtest Comparison")
    logger.info("=" * 70)
    logger.info("")

    # --- Configuration ---
    # Use a low entry threshold so all signals trigger trades
    config = ExecutionConfig(
        signal=SignalThresholdConfig(
            entry_threshold=0.05,
            reversal_threshold=0.5,
        ),
        stop_loss=StopLossConfig(
            type=StopLossType.FIXED_PIPS,
            fixed_pips=30.0,
            enabled=True,
        ),
        take_profit=TakeProfitConfig(
            type=TakeProfitType.FIXED_PIPS,
            fixed_pips=60.0,
            enabled=True,
        ),
        trailing_stop=TrailingStopConfig(enabled=False),
        time_exit=TimeExitConfig(
            enabled=True,
            max_bars_in_trade=48,  # 4 hours
            close_before_weekend=False,
        ),
    )
    risk_config = RiskConfig(
        initial_balance=1_000_000.0,
        max_drawdown=0.20,
    )

    # --- Generate data with uptrend (like 2024 USD/JPY) ---
    logger.info("Generating synthetic USD/JPY data (uptrend)...")
    df_uptrend = generate_synthetic_usdjpy(
        n_bars=10000,
        trend=0.00002,  # slight uptrend
    )
    logger.info(f"  Period: {df_uptrend.index[0]} to {df_uptrend.index[-1]}")
    logger.info(
        f"  Price: {df_uptrend['close'].iloc[0]:.2f} -> "
        f"{df_uptrend['close'].iloc[-1]:.2f}"
    )
    trend_return = df_uptrend["close"].iloc[-1] / df_uptrend["close"].iloc[0] - 1
    logger.info(f"  Buy & Hold return: {trend_return:.2%}")
    logger.info("")

    # --- Run backtests ---
    signal_generators = [
        ("Random Baseline", RandomSignalGenerator(seed=42, signal_probability=0.1)),
        ("SMA Cross (10/30)", SMACrossSignalGenerator(fast_period=10, slow_period=30)),
        ("Always Long (DQN)", AlwaysLongSignalGenerator(score=0.06)),
    ]

    results = []
    for name, sig_gen in signal_generators:
        logger.info(f"Running: {name}...")
        result = run_single_backtest(name, sig_gen, df_uptrend, config, risk_config)
        results.append(result)

    # --- Print results ---
    logger.info("")
    logger.info("-" * 70)
    logger.info(
        f"{'Strategy':<25} {'Trades':>7} {'WinRate':>8} "
        f"{'PnL(pips)':>10} {'Return':>8} {'PF':>6}"
    )
    logger.info("-" * 70)

    for r in results:
        pf_str = f"{r['profit_factor']:.2f}" if r["profit_factor"] < 100 else "inf"
        logger.info(
            f"{r['name']:<25} {r['total_trades']:>7} "
            f"{r['win_rate']:>7.1%} {r['total_pnl_pips']:>10.1f} "
            f"{r['total_return']:>7.2%} {pf_str:>6}"
        )
    logger.info("-" * 70)
    logger.info("")

    # --- Analysis ---
    always_long = results[2]
    random = results[0]

    logger.info("Analysis:")
    logger.info("")
    if always_long["total_return"] > random["total_return"]:
        logger.info("  The 'Always Long' strategy outperformed Random.")
        logger.info("  But this is NOT because it learned to trade - it's because")
        logger.info("  the underlying data has an uptrend. The model just learned")
        logger.info("  to exploit the training data distribution.")
        logger.info("")
        logger.info("  This is exactly what happened with our initial DQN:")
        logger.info("  - 2024 USD/JPY rose from 141 to 157 (+11%)")
        logger.info("  - The DQN learned to always BUY")
        logger.info("  - It looked profitable in backtests")
        logger.info("  - But it had ZERO predictive power")
    logger.info("")
    logger.info("  Lesson: Always compare against baselines.")
    logger.info("  If your ML model can't beat SMA crossover,")
    logger.info("  it probably learned the trend, not the signal.")


if __name__ == "__main__":
    main()
