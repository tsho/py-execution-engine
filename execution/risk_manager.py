"""Risk management for Execution Engine.

This module handles position sizing, daily loss limits,
and maximum drawdown controls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from execution.config import RiskConfig
from execution.position_manager import ClosedTrade

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: date
    pnl: float = 0.0
    trades: int = 0
    winners: int = 0
    losers: int = 0


class RiskManager:
    """Manages trading risk and position sizing.

    Enforces risk limits including:
    - Maximum position size
    - Risk per trade
    - Daily loss limit
    - Maximum drawdown
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.balance = config.initial_balance
        self.peak_balance = config.initial_balance
        self.daily_stats: dict[date, DailyStats] = {}
        self.current_date: date | None = None
        self.is_daily_limit_hit = False
        self.is_drawdown_limit_hit = False

    def can_open_position(
        self, current_time: datetime | None = None
    ) -> tuple[bool, str]:
        """Check if new position can be opened."""
        current_drawdown = self.get_current_drawdown()
        max_dd = self.config.max_drawdown
        if current_drawdown >= max_dd:
            self.is_drawdown_limit_hit = True
            return False, f"Max drawdown limit ({max_dd:.1%}) hit"

        if current_time:
            current_dt = current_time.date()
            if current_dt in self.daily_stats:
                daily_pnl = self.daily_stats[current_dt].pnl
                daily_loss = -daily_pnl / self.config.initial_balance
                max_daily = self.config.max_daily_loss
                if daily_loss >= max_daily:
                    self.is_daily_limit_hit = True
                    return False, f"Daily loss limit ({max_daily:.1%}) hit"

        return True, ""

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
    ) -> float:
        """Calculate position size based on risk per trade.

        Uses fixed fractional position sizing:
        size = (balance * risk_per_trade) / stop_distance
        """
        stop_distance = abs(entry_price - stop_loss_price) / entry_price

        if stop_distance <= 0:
            logger.warning("Invalid stop distance, using max position size")
            return self.config.max_position_size

        position_size = self.config.risk_per_trade / stop_distance
        position_size = min(position_size, self.config.max_position_size)
        return position_size

    def update_on_trade_close(
        self,
        trade: ClosedTrade,
        current_time: datetime | None = None,
    ) -> None:
        """Update risk manager state after trade close."""
        pnl_amount = trade.pnl * self.balance
        self.balance += pnl_amount
        self.peak_balance = max(self.peak_balance, self.balance)

        if current_time:
            current_dt = current_time.date()
            if current_dt not in self.daily_stats:
                self.daily_stats[current_dt] = DailyStats(date=current_dt)

            stats = self.daily_stats[current_dt]
            stats.pnl += trade.pnl
            stats.trades += 1
            if trade.is_winner():
                stats.winners += 1
            else:
                stats.losers += 1

    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.balance) / self.peak_balance

    def get_total_return(self) -> float:
        """Get total return since start."""
        initial = self.config.initial_balance
        return (self.balance - initial) / initial

    def get_balance(self) -> float:
        """Get current balance."""
        return self.balance

    def get_stats(self) -> dict[str, Any]:
        """Get risk manager statistics."""
        return {
            "balance": self.balance,
            "initial_balance": self.config.initial_balance,
            "peak_balance": self.peak_balance,
            "total_return": self.get_total_return(),
            "current_drawdown": self.get_current_drawdown(),
            "max_drawdown_limit": self.config.max_drawdown,
            "is_drawdown_limit_hit": self.is_drawdown_limit_hit,
            "is_daily_limit_hit": self.is_daily_limit_hit,
        }

    def reset(self) -> None:
        """Reset risk manager state."""
        self.balance = self.config.initial_balance
        self.peak_balance = self.config.initial_balance
        self.daily_stats = {}
        self.current_date = None
        self.is_daily_limit_hit = False
        self.is_drawdown_limit_hit = False

    def on_new_day(self, new_date: date) -> None:
        """Handle transition to new trading day."""
        if self.current_date != new_date:
            self.current_date = new_date
            self.is_daily_limit_hit = False
