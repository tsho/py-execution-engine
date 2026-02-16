"""Position and trade management for Execution Engine.

This module handles position lifecycle including opening, updating,
and closing positions with SL/TP management. It also defines the
core data classes for positions and closed trades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any

from execution.config import ExecutionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class PositionSide(IntEnum):
    """Position side enumeration."""

    FLAT = 0
    LONG = 1
    SHORT = -1


class ExitReason(Enum):
    """Reason for exiting a position."""

    SIGNAL = "signal"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    WEEKEND_CLOSE = "weekend_close"
    MANUAL = "manual"


@dataclass
class Position:
    """Open position information.

    Attributes:
        side: Position side (LONG or SHORT).
        entry_price: Price at which position was opened.
        entry_time: Timestamp when position was opened.
        entry_bar: Bar index when position was opened.
        size: Position size in units.
        stop_loss: Current stop loss price level.
        take_profit: Current take profit price level.
        trailing_stop_active: Whether trailing stop is active.
        trailing_stop_level: Current trailing stop price level.
        highest_price: Highest price since entry (for trailing stop).
        lowest_price: Lowest price since entry (for trailing stop).
        entry_atr: ATR value at entry time.
        entry_score: Signal score at entry.
    """

    side: PositionSide
    entry_price: float
    entry_time: datetime | None = None
    entry_bar: int = 0
    size: float = 1.0
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop_active: bool = False
    trailing_stop_level: float | None = None
    highest_price: float | None = None
    lowest_price: float | None = None
    entry_atr: float | None = None
    entry_score: float = 0.0

    def __post_init__(self) -> None:
        if self.highest_price is None:
            self.highest_price = self.entry_price
        if self.lowest_price is None:
            self.lowest_price = self.entry_price

    def update_price_extremes(self, high: float, low: float) -> None:
        """Update highest and lowest prices since entry."""
        if self.highest_price is None or high > self.highest_price:
            self.highest_price = high
        if self.lowest_price is None or low < self.lowest_price:
            self.lowest_price = low

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL as a fraction of entry price."""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) / self.entry_price
        if self.side == PositionSide.SHORT:
            return (self.entry_price - current_price) / self.entry_price
        return 0.0

    def get_unrealized_pnl_pips(self, current_price: float) -> float:
        """Calculate unrealized PnL in pips (1 pip = 0.01 for USD/JPY)."""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) / 0.01
        if self.side == PositionSide.SHORT:
            return (self.entry_price - current_price) / 0.01
        return 0.0

    def check_stop_loss_hit(self, low: float, high: float) -> bool:
        """Check if stop loss was hit."""
        if self.stop_loss is None:
            return False
        if self.side == PositionSide.LONG:
            return low <= self.stop_loss
        if self.side == PositionSide.SHORT:
            return high >= self.stop_loss
        return False

    def check_take_profit_hit(self, low: float, high: float) -> bool:
        """Check if take profit was hit."""
        if self.take_profit is None:
            return False
        if self.side == PositionSide.LONG:
            return high >= self.take_profit
        if self.side == PositionSide.SHORT:
            return low <= self.take_profit
        return False

    def check_trailing_stop_hit(self, low: float, high: float) -> bool:
        """Check if trailing stop was hit."""
        if not self.trailing_stop_active or self.trailing_stop_level is None:
            return False
        if self.side == PositionSide.LONG:
            return low <= self.trailing_stop_level
        if self.side == PositionSide.SHORT:
            return high >= self.trailing_stop_level
        return False

    def bars_in_trade(self, current_bar: int) -> int:
        """Calculate number of bars in trade."""
        return current_bar - self.entry_bar

    def to_dict(self) -> dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "side": self.side.name,
            "entry_price": self.entry_price,
            "entry_time": (self.entry_time.isoformat() if self.entry_time else None),
            "entry_bar": self.entry_bar,
            "size": self.size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop_active": self.trailing_stop_active,
            "trailing_stop_level": self.trailing_stop_level,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "entry_atr": self.entry_atr,
            "entry_score": self.entry_score,
        }


@dataclass
class ClosedTrade:
    """Record of a closed trade.

    Attributes:
        side: Position side (LONG or SHORT).
        entry_price: Price at which position was opened.
        exit_price: Price at which position was closed.
        entry_time: Timestamp when position was opened.
        exit_time: Timestamp when position was closed.
        entry_bar: Bar index when position was opened.
        exit_bar: Bar index when position was closed.
        size: Position size.
        pnl: Realized PnL as a fraction.
        pnl_pips: Realized PnL in pips.
        exit_reason: Reason for exit.
        entry_score: Signal score at entry.
        exit_score: Signal score at exit.
        max_favorable_excursion: Maximum favorable price movement.
        max_adverse_excursion: Maximum adverse price movement.
        transaction_cost: Transaction cost fraction.
    """

    side: PositionSide
    entry_price: float
    exit_price: float
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    entry_bar: int = 0
    exit_bar: int = 0
    size: float = 1.0
    pnl: float = 0.0
    pnl_pips: float = 0.0
    exit_reason: ExitReason = ExitReason.SIGNAL
    entry_score: float = 0.0
    exit_score: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    transaction_cost: float = 0.0

    @classmethod
    def from_position(
        cls,
        position: Position,
        exit_price: float,
        exit_time: datetime | None,
        exit_bar: int,
        exit_reason: ExitReason,
        exit_score: float = 0.0,
        transaction_cost: float = 0.0,
    ) -> ClosedTrade:
        """Create ClosedTrade from Position."""
        if position.side == PositionSide.LONG:
            pnl = (exit_price - position.entry_price) / position.entry_price
            pnl_pips = (exit_price - position.entry_price) / 0.01
            mfe = (
                (position.highest_price - position.entry_price) / position.entry_price
                if position.highest_price
                else 0.0
            )
            mae = (
                (position.entry_price - position.lowest_price) / position.entry_price
                if position.lowest_price
                else 0.0
            )
        elif position.side == PositionSide.SHORT:
            pnl = (position.entry_price - exit_price) / position.entry_price
            pnl_pips = (position.entry_price - exit_price) / 0.01
            mfe = (
                (position.entry_price - position.lowest_price) / position.entry_price
                if position.lowest_price
                else 0.0
            )
            mae = (
                (position.highest_price - position.entry_price) / position.entry_price
                if position.highest_price
                else 0.0
            )
        else:
            pnl = 0.0
            pnl_pips = 0.0
            mfe = 0.0
            mae = 0.0

        pnl -= transaction_cost

        return cls(
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_bar=position.entry_bar,
            exit_bar=exit_bar,
            size=position.size,
            pnl=pnl,
            pnl_pips=pnl_pips,
            exit_reason=exit_reason,
            entry_score=position.entry_score,
            exit_score=exit_score,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            transaction_cost=transaction_cost,
        )

    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    def duration_bars(self) -> int:
        """Get trade duration in bars."""
        return self.exit_bar - self.entry_bar

    def to_dict(self) -> dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "side": self.side.name,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": (self.entry_time.isoformat() if self.entry_time else None),
            "exit_time": (self.exit_time.isoformat() if self.exit_time else None),
            "entry_bar": self.entry_bar,
            "exit_bar": self.exit_bar,
            "size": self.size,
            "pnl": self.pnl,
            "pnl_pips": self.pnl_pips,
            "exit_reason": self.exit_reason.value,
            "duration_bars": self.duration_bars(),
            "is_winner": self.is_winner(),
        }


@dataclass
class TradeStats:
    """Aggregate statistics for a series of trades."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pips: float = 0.0
    average_pnl: float = 0.0
    average_winner: float = 0.0
    average_loser: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    average_duration_bars: float = 0.0
    trades_by_exit_reason: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_trades(cls, trades: list[ClosedTrade]) -> TradeStats:
        """Calculate statistics from a list of trades."""
        if not trades:
            return cls()

        total_trades = len(trades)
        winners = [t for t in trades if t.is_winner()]
        losers = [t for t in trades if not t.is_winner()]

        winning_trades = len(winners)
        losing_trades = len(losers)

        total_pnl = sum(t.pnl for t in trades)
        total_pnl_pips = sum(t.pnl_pips for t in trades)
        average_pnl = total_pnl / total_trades

        average_winner = (
            sum(t.pnl for t in winners) / winning_trades if winners else 0.0
        )
        average_loser = sum(t.pnl for t in losers) / losing_trades if losers else 0.0

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        for trade in trades:
            if trade.is_winner():
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        average_duration = sum(t.duration_bars() for t in trades) / total_trades

        exit_counts: dict[str, int] = {}
        for trade in trades:
            reason = trade.exit_reason.value
            exit_counts[reason] = exit_counts.get(reason, 0) + 1

        return cls(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            total_pnl_pips=total_pnl_pips,
            average_pnl=average_pnl,
            average_winner=average_winner,
            average_loser=average_loser,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            average_duration_bars=average_duration,
            trades_by_exit_reason=exit_counts,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "total_pnl_pips": self.total_pnl_pips,
            "average_pnl": self.average_pnl,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "average_duration_bars": self.average_duration_bars,
            "trades_by_exit_reason": self.trades_by_exit_reason,
        }


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------


class PositionManager:
    """Manages position lifecycle and SL/TP levels.

    Handles opening positions with appropriate SL/TP levels,
    updating trailing stops, and closing positions with proper accounting.
    """

    def __init__(
        self,
        config: ExecutionConfig,
        transaction_cost: float = 0.0001,
    ) -> None:
        self.config = config
        self.transaction_cost = transaction_cost
        self.current_position: Position | None = None
        self.trade_history: list[ClosedTrade] = []

    def has_position(self) -> bool:
        """Check if there is an open position."""
        return self.current_position is not None

    def get_position_side(self) -> PositionSide:
        """Get current position side."""
        if self.current_position is None:
            return PositionSide.FLAT
        return self.current_position.side

    def open_position(
        self,
        side: PositionSide,
        entry_price: float,
        entry_time: datetime | None,
        entry_bar: int,
        size: float,
        atr: float | None = None,
        score: float = 0.0,
    ) -> Position:
        """Open a new position with SL/TP levels."""
        if self.current_position is not None:
            raise ValueError("Cannot open position while another is open")
        if side == PositionSide.FLAT:
            raise ValueError("Cannot open FLAT position")

        sl_distance = self.config.stop_loss.calculate_stop_distance(entry_price, atr)
        sl_valid = sl_distance < float("inf")
        if side == PositionSide.LONG:
            stop_loss = entry_price - sl_distance if sl_valid else None
        else:
            stop_loss = entry_price + sl_distance if sl_valid else None

        tp_distance = self.config.take_profit.calculate_profit_distance(
            sl_distance, atr
        )
        tp_valid = tp_distance < float("inf")
        if side == PositionSide.LONG:
            take_profit = entry_price + tp_distance if tp_valid else None
        else:
            take_profit = entry_price - tp_distance if tp_valid else None

        position = Position(
            side=side,
            entry_price=entry_price,
            entry_time=entry_time,
            entry_bar=entry_bar,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_atr=atr,
            entry_score=score,
        )
        self.current_position = position

        logger.debug(
            "Opened %s at %.3f, SL=%s, TP=%s",
            side.name,
            entry_price,
            f"{stop_loss:.3f}" if stop_loss is not None else "None",
            f"{take_profit:.3f}" if take_profit is not None else "None",
        )
        return position

    def update_position(
        self,
        high: float,
        low: float,
        current_price: float,
        atr: float | None = None,
    ) -> None:
        """Update position state with current bar data."""
        if self.current_position is None:
            return
        self.current_position.update_price_extremes(high, low)
        self._update_trailing_stop(current_price, atr)

    def _update_trailing_stop(
        self, current_price: float, atr: float | None = None
    ) -> None:
        """Update trailing stop level."""
        if self.current_position is None:
            return
        if not self.config.trailing_stop.enabled:
            return

        position = self.current_position
        activation_distance = self.config.trailing_stop.get_activation_distance()

        if not position.trailing_stop_active:
            pnl = position.get_unrealized_pnl(current_price)
            current_profit = pnl * position.entry_price
            if current_profit >= activation_distance:
                position.trailing_stop_active = True

        if position.trailing_stop_active:
            trail_distance = self.config.trailing_stop.calculate_trail_distance(
                current_price, atr
            )
            if position.side == PositionSide.LONG:
                high_price = position.highest_price
                new_level = high_price - trail_distance if high_price else None
                if new_level and (
                    position.trailing_stop_level is None
                    or new_level > position.trailing_stop_level
                ):
                    position.trailing_stop_level = new_level
            elif position.side == PositionSide.SHORT:
                low_price = position.lowest_price
                new_level = low_price + trail_distance if low_price else None
                if new_level and (
                    position.trailing_stop_level is None
                    or new_level < position.trailing_stop_level
                ):
                    position.trailing_stop_level = new_level

    def check_exit_conditions(
        self,
        high: float,
        low: float,
        current_bar: int,
        current_time: datetime | None = None,
    ) -> ExitReason | None:
        """Check all exit conditions (SL > TP > Trailing > Time)."""
        if self.current_position is None:
            return None

        position = self.current_position

        if position.check_stop_loss_hit(low, high):
            return ExitReason.STOP_LOSS
        if position.check_take_profit_hit(low, high):
            return ExitReason.TAKE_PROFIT
        if position.check_trailing_stop_hit(low, high):
            return ExitReason.TRAILING_STOP

        if self.config.time_exit.enabled:
            bars_in_trade = position.bars_in_trade(current_bar)
            if bars_in_trade >= self.config.time_exit.max_bars_in_trade:
                return ExitReason.TIME_EXIT
            if (
                self.config.time_exit.close_before_weekend
                and current_time
                and current_time.weekday() == 4
                and current_time.hour >= self.config.time_exit.weekend_close_hour
            ):
                return ExitReason.WEEKEND_CLOSE

        return None

    def close_position(
        self,
        exit_price: float,
        exit_time: datetime | None,
        exit_bar: int,
        exit_reason: ExitReason,
        exit_score: float = 0.0,
    ) -> ClosedTrade:
        """Close the current position."""
        if self.current_position is None:
            raise ValueError("No position to close")

        trade = ClosedTrade.from_position(
            position=self.current_position,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_bar=exit_bar,
            exit_reason=exit_reason,
            exit_score=exit_score,
            transaction_cost=self.transaction_cost,
        )
        self.trade_history.append(trade)
        self.current_position = None

        logger.debug(
            "Closed %s at %.3f, PnL=%.4f%%, reason=%s",
            trade.side.name,
            exit_price,
            trade.pnl * 100,
            exit_reason.value,
        )
        return trade

    def get_exit_price(self, exit_reason: ExitReason, close_price: float) -> float:
        """Get exit price based on exit reason."""
        if self.current_position is None:
            return close_price

        position = self.current_position
        if exit_reason == ExitReason.STOP_LOSS and position.stop_loss:
            return position.stop_loss
        if exit_reason == ExitReason.TAKE_PROFIT and position.take_profit:
            return position.take_profit
        if exit_reason == ExitReason.TRAILING_STOP and position.trailing_stop_level:
            return position.trailing_stop_level
        return close_price

    def force_close_all(
        self,
        exit_price: float,
        exit_time: datetime | None,
        exit_bar: int,
        exit_reason: ExitReason = ExitReason.MANUAL,
    ) -> ClosedTrade | None:
        """Force close any open position."""
        if self.current_position is None:
            return None
        return self.close_position(
            exit_price=exit_price,
            exit_time=exit_time,
            exit_bar=exit_bar,
            exit_reason=exit_reason,
        )

    def get_trade_history(self) -> list[ClosedTrade]:
        """Get all closed trades."""
        return self.trade_history.copy()

    def get_position_info(self) -> dict[str, Any] | None:
        """Get current position info as dictionary."""
        if self.current_position is None:
            return None
        return self.current_position.to_dict()

    def reset(self) -> None:
        """Reset position manager state."""
        self.current_position = None
        self.trade_history = []
