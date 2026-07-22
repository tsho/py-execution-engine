# Signal-Agnostic Execution Framework

> [!CAUTION]
> **Disclaimer:** This project is for **educational and demonstration purposes only**. It is designed to illustrate concepts in machine learning and reinforcement learning applied to trading systems. It does **not** constitute financial advice, does **not** guarantee any trading profits, and should **not** be used for live trading. Use at your own risk.

A modular execution engine for FX trading that cleanly separates **signal generation** from **position management** and **risk control**.

Born from real failures in building an FX trading system with Deep Reinforcement Learning.

## The Story (4 Acts)

### Act 1: The Buy & Hold Trap
We built an end-to-end DQN that handled everything. It looked great in backtests -- but it had simply learned to always output BUY, exploiting the uptrend in training data.

```bash
uv run python examples/act1_buy_and_hold_trap.py
```

### Act 2: Reward Design Quagmire
We tried to fix the DQN by redesigning the reward function. Adding penalties led to random trading. Removing unrealized PnL led to **Zero-Trade Collapse** -- the policy degenerated to a single constant action across the whole test set, booking zero completed trades and exactly 0.00% realized PnL.

We first read this as "the agent learned that inaction is optimal." On closer inspection it is subtler, and worth stating carefully: the constant-action behavior showed up **even with zero transaction cost, and even before training** -- a freshly initialized network already emitted one constant action, because its Q-values carried a systematic offset larger than the input-driven variation. So the honest takeaway is not a clean "trading is negative-EV, therefore do nothing" story. It is more uncomfortable: **on the two metrics a desk usually watches -- realized PnL and trade count -- a collapsed policy, an untrained model, and a genuinely idle one are indistinguishable.** (Action-distribution entropy does tell them apart.)

```bash
uv run python examples/act2_zero_trade_collapse.py
```

### Act 3-1: Architecture Separation
The breakthrough: stop trying to fix the reward. Instead, separate the architecture into a **Signal Generator** (predicts direction) and an **Execution Engine** (manages positions). Each component does one thing well.

```bash
uv run python examples/act3_execution_engine.py
```

### Act 3-2: Signal Generator Evolution
With signal-agnostic execution in place, we could focus purely on improving signals. LightGBM (105 features, no temporal context, AUC 0.52) gave way to **Transformer Self-Attention** (50 bars × 105 features), which learns "which of the past 50 bars matter NOW?"

```bash
uv run python examples/act4_signal_evolution.py
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│ Signal Generator │────>│ Execution Engine  │────>│ Risk Manager  │
│  (pluggable)     │     │  (SL/TP/Trailing) │     │  (sizing/DD)  │
└─────────────────┘     └──────────────────┘     └───────────────┘
        │                        │
   Any source:              Handles:
   - DQN Q-values          - Stop loss
   - LightGBM proba        - Take profit
   - Transformer score     - Trailing stop
   - SMA crossover         - Time-based exit
   - Random baseline       - Weekend close
```

## Quick Start

```python
from execution.engine import ExecutionEngine
from execution.config import ExecutionConfig, RiskConfig
from signals.dummy_sma_cross import SMACrossSignalGenerator

# 1. Choose a signal generator
signal_gen = SMACrossSignalGenerator(fast_period=10, slow_period=30)

# 2. Configure execution
config = ExecutionConfig()  # sensible defaults
risk_config = RiskConfig(initial_balance=1_000_000.0)

# 3. Create engine
engine = ExecutionEngine(
    signal_generator=signal_gen,
    execution_config=config,
    risk_config=risk_config,
    transaction_cost=0.0003,  # 3 pips spread
)

# 4. Run backtest
observations = df["close"].values.reshape(-1, 1)
results = engine.run_backtest(df, observations=observations)

# 5. Analyze
stats = engine.get_trade_stats()
print(f"Trades: {stats.total_trades}, Win rate: {stats.win_rate:.1%}")
```

## Creating a Custom Signal Generator

Implement the `SignalGenerator` interface:

```python
from signals.base import SignalGenerator, TradingSignal
import numpy as np

class MySignalGenerator(SignalGenerator):
    def generate(self, observation: np.ndarray, **kwargs) -> TradingSignal:
        score = your_model.predict(observation)
        return TradingSignal(score=score, confidence=abs(score))

    def reset(self) -> None:
        pass
```

The score is a float in [-1, +1]:
- `+1.0` = strong buy signal
- ` 0.0` = no signal
- `-1.0` = strong sell signal

## Project Structure

```
py-execution-engine/
├── execution/
│   ├── __init__.py
│   ├── config.py              # SL/TP/trailing/time exit configuration
│   ├── engine.py              # Main execution engine
│   ├── position_manager.py    # Position lifecycle + data classes
│   └── risk_manager.py        # Position sizing, daily limits, drawdown
├── signals/
│   ├── __init__.py
│   ├── base.py                # SignalGenerator ABC + TradingSignal
│   ├── dummy_random.py        # Random baseline
│   ├── dummy_sma_cross.py     # SMA crossover example
│   ├── dqn_score.py           # DQN Q-value → score converter
│   ├── lgbm_signal.py         # LightGBM signal generator
│   └── transformer_signal.py  # Transformer signal generator
├── models/
│   ├── __init__.py
│   ├── environment.py         # RL environment (reward design demo)
│   └── transformer.py         # Transformer architecture
├── features/
│   └── __init__.py
├── examples/
│   ├── act1_buy_and_hold_trap.py    # Act 1: Buy & Hold trap
│   ├── act2_zero_trade_collapse.py  # Act 2: Reward design failures
│   ├── act3_execution_engine.py     # Act 3: Architecture separation
│   └── act4_signal_evolution.py     # Act 4: LightGBM → Transformer
└── README.md
```

## Requirements

- Python 3.12+
- numpy
- pandas
- torch (optional, for Transformer inference)

## License

Apache 2.0