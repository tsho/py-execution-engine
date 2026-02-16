"""Signal generators for the execution engine.

This package provides the base signal interface and example implementations.
Any signal source (ML model, technical indicator, random) can be plugged
into the execution engine through the SignalGenerator interface.
"""

from signals.base import SignalGenerator, TradingSignal

__all__ = [
    "SignalGenerator",
    "TradingSignal",
]
