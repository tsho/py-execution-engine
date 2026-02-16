"""Configuration classes for Execution Engine.

This module defines dataclasses for configuring stop loss, take profit,
trailing stop, time-based exit, and risk management parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StopLossType(Enum):
    """Type of stop loss calculation."""

    FIXED_PIPS = "fixed_pips"
    ATR_BASED = "atr_based"
    PERCENTAGE = "percentage"


class TakeProfitType(Enum):
    """Type of take profit calculation."""

    FIXED_PIPS = "fixed_pips"
    ATR_BASED = "atr_based"
    RISK_REWARD = "risk_reward"


class TrailingStopType(Enum):
    """Type of trailing stop calculation."""

    FIXED_PIPS = "fixed_pips"
    ATR_BASED = "atr_based"
    PERCENTAGE = "percentage"


@dataclass
class StopLossConfig:
    """Configuration for stop loss calculation.

    Attributes:
        type: Method of stop loss calculation.
        fixed_pips: Fixed stop loss in pips (for FIXED_PIPS type).
        atr_multiplier: ATR multiplier (for ATR_BASED type).
        percentage: Percentage of entry price (for PERCENTAGE type).
        enabled: Whether stop loss is enabled.
    """

    type: StopLossType = StopLossType.ATR_BASED
    fixed_pips: float = 20.0
    atr_multiplier: float = 2.0
    percentage: float = 0.005  # 0.5%
    enabled: bool = True

    def calculate_stop_distance(
        self, entry_price: float, atr: float | None = None
    ) -> float:
        """Calculate stop loss distance from entry price.

        Args:
            entry_price: Entry price of the position.
            atr: Average True Range value (required for ATR_BASED).

        Returns:
            Stop loss distance in price units.
        """
        if not self.enabled:
            return float("inf")

        if self.type == StopLossType.FIXED_PIPS:
            # For USD/JPY, 1 pip = 0.01
            return self.fixed_pips * 0.01
        if self.type == StopLossType.ATR_BASED:
            if atr is None:
                raise ValueError("ATR value required for ATR_BASED stop loss")
            return atr * self.atr_multiplier
        if self.type == StopLossType.PERCENTAGE:
            return entry_price * self.percentage
        raise ValueError(f"Unknown stop loss type: {self.type}")


@dataclass
class TakeProfitConfig:
    """Configuration for take profit calculation.

    Attributes:
        type: Method of take profit calculation.
        fixed_pips: Fixed take profit in pips (for FIXED_PIPS type).
        atr_multiplier: ATR multiplier (for ATR_BASED type).
        risk_reward_ratio: Risk-reward ratio (for RISK_REWARD type).
        enabled: Whether take profit is enabled.
    """

    type: TakeProfitType = TakeProfitType.RISK_REWARD
    fixed_pips: float = 40.0
    atr_multiplier: float = 3.0
    risk_reward_ratio: float = 2.0
    enabled: bool = True

    def calculate_profit_distance(
        self,
        stop_distance: float | None = None,
        atr: float | None = None,
    ) -> float:
        """Calculate take profit distance from entry price.

        Args:
            stop_distance: Stop loss distance (required for RISK_REWARD).
            atr: Average True Range value (required for ATR_BASED).

        Returns:
            Take profit distance in price units.
        """
        if not self.enabled:
            return float("inf")

        if self.type == TakeProfitType.FIXED_PIPS:
            return self.fixed_pips * 0.01
        if self.type == TakeProfitType.ATR_BASED:
            if atr is None:
                raise ValueError("ATR value required for ATR_BASED take profit")
            return atr * self.atr_multiplier
        if self.type == TakeProfitType.RISK_REWARD:
            if stop_distance is None:
                raise ValueError("Stop distance required for RISK_REWARD take profit")
            return stop_distance * self.risk_reward_ratio
        raise ValueError(f"Unknown take profit type: {self.type}")


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop.

    Attributes:
        type: Method of trailing stop calculation.
        activation_profit_pips: Profit in pips to activate trailing stop.
        trail_distance_pips: Fixed trail distance in pips (for FIXED_PIPS).
        atr_multiplier: ATR multiplier for trail distance (for ATR_BASED).
        trail_percentage: Percentage for trail distance (for PERCENTAGE).
        enabled: Whether trailing stop is enabled.
    """

    type: TrailingStopType = TrailingStopType.ATR_BASED
    activation_profit_pips: float = 10.0
    trail_distance_pips: float = 15.0
    atr_multiplier: float = 1.5
    trail_percentage: float = 0.003  # 0.3%
    enabled: bool = True

    def get_activation_distance(self) -> float:
        """Get activation distance in price units."""
        return self.activation_profit_pips * 0.01

    def calculate_trail_distance(
        self, current_price: float, atr: float | None = None
    ) -> float:
        """Calculate trailing stop distance.

        Args:
            current_price: Current market price.
            atr: Average True Range value (required for ATR_BASED).

        Returns:
            Trail distance in price units.
        """
        if not self.enabled:
            return float("inf")

        if self.type == TrailingStopType.FIXED_PIPS:
            return self.trail_distance_pips * 0.01
        if self.type == TrailingStopType.ATR_BASED:
            if atr is None:
                raise ValueError("ATR value required for ATR_BASED trailing stop")
            return atr * self.atr_multiplier
        if self.type == TrailingStopType.PERCENTAGE:
            return current_price * self.trail_percentage
        raise ValueError(f"Unknown trailing stop type: {self.type}")


@dataclass
class TimeExitConfig:
    """Configuration for time-based exit.

    Attributes:
        enabled: Whether time-based exit is enabled.
        max_bars_in_trade: Maximum number of bars to hold a position.
        close_before_weekend: Whether to close positions before weekend.
        weekend_close_hour: Hour to close positions on Friday (UTC).
    """

    enabled: bool = True
    max_bars_in_trade: int = 48  # 4 hours for 5-min bars
    close_before_weekend: bool = True
    weekend_close_hour: int = 21  # 21:00 UTC Friday


@dataclass
class SignalThresholdConfig:
    """Configuration for signal thresholds.

    Attributes:
        entry_threshold: Minimum |score| to enter a position.
        exit_threshold: |score| threshold for signal-based exit.
        reversal_threshold: |score| threshold for position reversal.
        use_confidence_weight: Whether to weight signals by confidence.
    """

    entry_threshold: float = 0.3
    exit_threshold: float = 0.1
    reversal_threshold: float = 0.5
    use_confidence_weight: bool = True


@dataclass
class ExecutionConfig:
    """Combined configuration for execution engine.

    Attributes:
        signal: Signal threshold configuration.
        stop_loss: Stop loss configuration.
        take_profit: Take profit configuration.
        trailing_stop: Trailing stop configuration.
        time_exit: Time-based exit configuration.
    """

    signal: SignalThresholdConfig = field(default_factory=SignalThresholdConfig)
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = field(default_factory=TakeProfitConfig)
    trailing_stop: TrailingStopConfig = field(default_factory=TrailingStopConfig)
    time_exit: TimeExitConfig = field(default_factory=TimeExitConfig)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ExecutionConfig:
        """Create ExecutionConfig from dictionary."""
        signal_dict = config_dict.get("signal", {})
        sl_dict = config_dict.get("stop_loss", {})
        tp_dict = config_dict.get("take_profit", {})
        ts_dict = config_dict.get("trailing_stop", {})
        te_dict = config_dict.get("time_exit", {})

        sl_type = sl_dict.get("type", "atr_based")
        sl_config = StopLossConfig(
            type=StopLossType(sl_type) if isinstance(sl_type, str) else sl_type,
            fixed_pips=sl_dict.get("fixed_pips", 20.0),
            atr_multiplier=sl_dict.get("atr_multiplier", 2.0),
            percentage=sl_dict.get("percentage", 0.005),
            enabled=sl_dict.get("enabled", True),
        )

        tp_type = tp_dict.get("type", "risk_reward")
        tp_config = TakeProfitConfig(
            type=TakeProfitType(tp_type) if isinstance(tp_type, str) else tp_type,
            fixed_pips=tp_dict.get("fixed_pips", 40.0),
            atr_multiplier=tp_dict.get("atr_multiplier", 3.0),
            risk_reward_ratio=tp_dict.get("risk_reward_ratio", 2.0),
            enabled=tp_dict.get("enabled", True),
        )

        ts_type = ts_dict.get("type", "atr_based")
        ts_config = TrailingStopConfig(
            type=TrailingStopType(ts_type) if isinstance(ts_type, str) else ts_type,
            activation_profit_pips=ts_dict.get("activation_profit_pips", 10.0),
            trail_distance_pips=ts_dict.get("trail_distance_pips", 15.0),
            atr_multiplier=ts_dict.get("atr_multiplier", 1.5),
            trail_percentage=ts_dict.get("trail_percentage", 0.003),
            enabled=ts_dict.get("enabled", True),
        )

        te_config = TimeExitConfig(
            enabled=te_dict.get("enabled", True),
            max_bars_in_trade=te_dict.get("max_bars_in_trade", 48),
            close_before_weekend=te_dict.get("close_before_weekend", True),
            weekend_close_hour=te_dict.get("weekend_close_hour", 21),
        )

        signal_config = SignalThresholdConfig(
            entry_threshold=signal_dict.get("entry_threshold", 0.3),
            exit_threshold=signal_dict.get("exit_threshold", 0.1),
            reversal_threshold=signal_dict.get("reversal_threshold", 0.5),
            use_confidence_weight=signal_dict.get("use_confidence_weight", True),
        )

        return cls(
            signal=signal_config,
            stop_loss=sl_config,
            take_profit=tp_config,
            trailing_stop=ts_config,
            time_exit=te_config,
        )


@dataclass
class RiskConfig:
    """Configuration for risk management.

    Attributes:
        max_position_size: Maximum position size as fraction of balance.
        risk_per_trade: Risk per trade as fraction of balance.
        max_daily_loss: Maximum daily loss as fraction of balance.
        max_drawdown: Maximum drawdown as fraction of peak balance.
        initial_balance: Initial account balance.
    """

    max_position_size: float = 0.10
    risk_per_trade: float = 0.01
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.10
    initial_balance: float = 1_000_000.0

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> RiskConfig:
        """Create RiskConfig from dictionary."""
        return cls(
            max_position_size=config_dict.get("max_position_size", 0.10),
            risk_per_trade=config_dict.get("risk_per_trade", 0.01),
            max_daily_loss=config_dict.get("max_daily_loss", 0.02),
            max_drawdown=config_dict.get("max_drawdown", 0.10),
            initial_balance=config_dict.get("initial_balance", 1_000_000.0),
        )
