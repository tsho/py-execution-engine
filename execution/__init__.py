"""Signal-agnostic execution engine for FX trading.

This package provides a modular execution framework that separates
signal generation from position management and risk control.
"""

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
    TrailingStopType,
)
from execution.engine import BarData, ExecutionEngine, ExecutionResult
from execution.position_manager import (
    ClosedTrade,
    ExitReason,
    Position,
    PositionManager,
    PositionSide,
    TradeStats,
)
from execution.risk_manager import RiskManager

__all__ = [
    "BarData",
    "ClosedTrade",
    "ExecutionConfig",
    "ExecutionEngine",
    "ExecutionResult",
    "ExitReason",
    "Position",
    "PositionManager",
    "PositionSide",
    "RiskConfig",
    "RiskManager",
    "SignalThresholdConfig",
    "StopLossConfig",
    "StopLossType",
    "TakeProfitConfig",
    "TakeProfitType",
    "TimeExitConfig",
    "TradeStats",
    "TrailingStopConfig",
    "TrailingStopType",
]
