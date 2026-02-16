"""Main Execution Engine for FX trading system.

This module provides the ExecutionEngine class that coordinates
signal generation, position management, and risk management.

The engine is **signal-agnostic**: it accepts any SignalGenerator
implementation and handles the full trade lifecycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from execution.config import ExecutionConfig, RiskConfig
from execution.position_manager import (
    ClosedTrade,
    ExitReason,
    PositionManager,
    PositionSide,
    TradeStats,
)
from execution.risk_manager import RiskManager
from signals.base import SignalGenerator, TradingSignal

logger = logging.getLogger(__name__)


@dataclass
class BarData:
    """OHLCV bar data."""

    timestamp: datetime | None
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    atr: float | None = None
    bar_index: int = 0

    @classmethod
    def from_series(cls, row: pd.Series, bar_index: int = 0) -> BarData:
        """Create BarData from pandas Series."""
        timestamp = row.name if isinstance(row.name, datetime) else None
        return cls(
            timestamp=timestamp,
            open=float(row.get("open", row.get("Open", 0))),
            high=float(row.get("high", row.get("High", 0))),
            low=float(row.get("low", row.get("Low", 0))),
            close=float(row.get("close", row.get("Close", 0))),
            volume=float(row.get("volume", row.get("Volume", 0))),
            atr=(float(row["atr"]) if "atr" in row and pd.notna(row["atr"]) else None),
            bar_index=bar_index,
        )


@dataclass
class ExecutionResult:
    """Result of processing a bar."""

    bar_index: int = 0
    signal: TradingSignal | None = None
    action_taken: str = "none"
    position_opened: bool = False
    position_closed: bool = False
    closed_trade: ClosedTrade | None = None
    current_position_side: PositionSide = PositionSide.FLAT
    balance: float = 0.0
    unrealized_pnl: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    """Main execution engine for FX trading.

    Coordinates signal generation, position management, and risk management
    to execute trades based on model signals.

    The execution flow for each bar:
    1. Generate signal from observation
    2. If position exists, check exit conditions (SL/TP/time/trailing)
    3. If no position and signal is strong enough, open position
    4. Update trailing stops and position state
    """

    def __init__(
        self,
        signal_generator: SignalGenerator,
        execution_config: ExecutionConfig | None = None,
        risk_config: RiskConfig | None = None,
        transaction_cost: float = 0.0001,
    ) -> None:
        self.signal_generator = signal_generator
        self.config = execution_config or ExecutionConfig()
        self.risk_config = risk_config or RiskConfig()

        self.position_manager = PositionManager(
            config=self.config,
            transaction_cost=transaction_cost,
        )
        self.risk_manager = RiskManager(self.risk_config)

        self.current_bar = 0
        self.transaction_cost = transaction_cost
        self.results_history: list[ExecutionResult] = []

    def process_bar(
        self,
        bar: BarData,
        observation: np.ndarray | None = None,
        signal: TradingSignal | None = None,
    ) -> ExecutionResult:
        """Process a single bar and execute trading logic."""
        self.current_bar = bar.bar_index

        if bar.timestamp:
            self.risk_manager.on_new_day(bar.timestamp.date())

        if signal is None:
            if observation is None:
                raise ValueError("Either observation or signal must be provided")
            signal = self.signal_generator.generate(observation)

        result = ExecutionResult(
            bar_index=bar.bar_index,
            signal=signal,
            balance=self.risk_manager.get_balance(),
        )

        if self.position_manager.has_position():
            exit_result = self._check_and_execute_exits(bar, signal, result)
            if exit_result:
                return exit_result

            self.position_manager.update_position(
                high=bar.high,
                low=bar.low,
                current_price=bar.close,
                atr=bar.atr,
            )

            position = self.position_manager.current_position
            if position:
                result.unrealized_pnl = position.get_unrealized_pnl(bar.close)
                result.current_position_side = position.side
                result.action_taken = "hold"
        else:
            entry_result = self._check_and_execute_entry(bar, signal, result)
            if entry_result:
                return entry_result
            result.action_taken = "wait"

        self.results_history.append(result)
        return result

    def _check_and_execute_exits(
        self,
        bar: BarData,
        signal: TradingSignal,
        result: ExecutionResult,
    ) -> ExecutionResult | None:
        """Check exit conditions and execute if needed."""
        exit_reason = self.position_manager.check_exit_conditions(
            high=bar.high,
            low=bar.low,
            current_bar=bar.bar_index,
            current_time=bar.timestamp,
        )

        if exit_reason:
            return self._execute_exit(bar, exit_reason, signal.score, result)

        position = self.position_manager.current_position
        if position:
            should_exit, reason = self._check_signal_exit(signal, position.side)
            if should_exit:
                return self._execute_exit(bar, reason, signal.score, result)

        return None

    def _check_signal_exit(
        self,
        signal: TradingSignal,
        current_side: PositionSide,
    ) -> tuple[bool, ExitReason]:
        """Check if signal indicates position should be closed."""
        threshold = self.config.signal.reversal_threshold

        if current_side == PositionSide.LONG and signal.score < -threshold:
            return True, ExitReason.SIGNAL
        if current_side == PositionSide.SHORT and signal.score > threshold:
            return True, ExitReason.SIGNAL

        return False, ExitReason.SIGNAL

    def _execute_exit(
        self,
        bar: BarData,
        exit_reason: ExitReason,
        exit_score: float,
        result: ExecutionResult,
    ) -> ExecutionResult:
        """Execute position exit."""
        exit_price = self.position_manager.get_exit_price(exit_reason, bar.close)

        trade = self.position_manager.close_position(
            exit_price=exit_price,
            exit_time=bar.timestamp,
            exit_bar=bar.bar_index,
            exit_reason=exit_reason,
            exit_score=exit_score,
        )

        self.risk_manager.update_on_trade_close(trade, bar.timestamp)

        result.position_closed = True
        result.closed_trade = trade
        result.current_position_side = PositionSide.FLAT
        result.balance = self.risk_manager.get_balance()
        result.action_taken = f"close_{exit_reason.value}"

        logger.info(
            "Bar %d: Closed %s at %.3f, PnL=%.4f%%, reason=%s",
            bar.bar_index,
            trade.side.name,
            exit_price,
            trade.pnl * 100,
            exit_reason.value,
        )

        self.results_history.append(result)
        return result

    def _check_and_execute_entry(
        self,
        bar: BarData,
        signal: TradingSignal,
        result: ExecutionResult,
    ) -> ExecutionResult | None:
        """Check entry conditions and execute if appropriate."""
        threshold = self.config.signal.entry_threshold
        if signal.get_strength() < threshold:
            return None

        can_trade, reason = self.risk_manager.can_open_position(bar.timestamp)
        if not can_trade:
            logger.warning("Cannot open position: %s", reason)
            result.action_taken = f"blocked_{reason}"
            return None

        direction = signal.get_direction()
        if direction == PositionSide.FLAT:
            return None

        sl_distance = self.config.stop_loss.calculate_stop_distance(bar.close, bar.atr)
        if direction == PositionSide.LONG:
            stop_loss_price = bar.close - sl_distance
        else:
            stop_loss_price = bar.close + sl_distance

        position_size = self.risk_manager.calculate_position_size(
            entry_price=bar.close,
            stop_loss_price=stop_loss_price,
        )

        position = self.position_manager.open_position(
            side=direction,
            entry_price=bar.close,
            entry_time=bar.timestamp,
            entry_bar=bar.bar_index,
            size=position_size,
            atr=bar.atr,
            score=signal.score,
        )

        result.position_opened = True
        result.current_position_side = direction
        result.action_taken = f"open_{direction.name.lower()}"
        result.metadata["position_size"] = position_size

        logger.info(
            "Bar %d: Opened %s at %.3f, size=%.4f, SL=%s, TP=%s",
            bar.bar_index,
            direction.name,
            bar.close,
            position_size,
            f"{position.stop_loss:.3f}" if position.stop_loss else "None",
            f"{position.take_profit:.3f}" if position.take_profit else "None",
        )

        self.results_history.append(result)
        return result

    def run_backtest(
        self,
        df: pd.DataFrame,
        observations: np.ndarray | None = None,
        signals: list[TradingSignal] | None = None,
        atr_column: str = "atr",
        verbose: bool = False,
    ) -> list[ExecutionResult]:
        """Run backtest over a DataFrame.

        Args:
            df: OHLCV DataFrame with index as timestamps.
            observations: Pre-computed observations (one per row).
            signals: Pre-generated signals (one per row).
            atr_column: Column name for ATR values.
            verbose: Whether to log progress.

        Returns:
            List of ExecutionResults for each bar.
        """
        results = []
        n_bars = len(df)

        for i, (idx, row) in enumerate(df.iterrows()):
            bar = BarData(
                timestamp=idx if isinstance(idx, datetime) else None,
                open=float(row.get("open", row.get("Open", 0))),
                high=float(row.get("high", row.get("High", 0))),
                low=float(row.get("low", row.get("Low", 0))),
                close=float(row.get("close", row.get("Close", 0))),
                volume=float(row.get("volume", row.get("Volume", 0))),
                atr=(
                    float(row[atr_column])
                    if atr_column in row and pd.notna(row[atr_column])
                    else None
                ),
                bar_index=i,
            )

            obs = observations[i] if observations is not None else None
            sig = signals[i] if signals is not None else None

            result = self.process_bar(bar, observation=obs, signal=sig)
            results.append(result)

            if verbose and i % 1000 == 0:
                logger.info("Processed %d/%d bars", i, n_bars)

        if self.position_manager.has_position():
            last_bar = BarData.from_series(df.iloc[-1], len(df) - 1)
            self.position_manager.force_close_all(
                exit_price=last_bar.close,
                exit_time=last_bar.timestamp,
                exit_bar=last_bar.bar_index,
                exit_reason=ExitReason.MANUAL,
            )

        return results

    def get_trade_stats(self) -> TradeStats:
        """Get statistics for all closed trades."""
        trades = self.position_manager.get_trade_history()
        return TradeStats.from_trades(trades)

    def get_trades(self) -> list[ClosedTrade]:
        """Get all closed trades."""
        return self.position_manager.get_trade_history()

    def get_risk_stats(self) -> dict[str, Any]:
        """Get risk manager statistics."""
        return self.risk_manager.get_stats()

    def get_equity_curve(self) -> list[float]:
        """Get equity curve from backtest results."""
        initial = self.risk_config.initial_balance
        equity = [initial]
        current = initial

        for result in self.results_history:
            if result.closed_trade:
                current += result.closed_trade.pnl * current
            equity.append(current)

        return equity

    def reset(self) -> None:
        """Reset engine state for new backtest."""
        self.position_manager.reset()
        self.risk_manager.reset()
        self.signal_generator.reset()
        self.current_bar = 0
        self.results_history = []
