"""Trading environment for reinforcement learning (simplified).

This module demonstrates the RL environment used in the early experiments.
It shows how different reward designs led to pathological behaviors:

- unrealized_pnl_weight=1.0 → Buy & Hold (agent holds forever)
- heavy penalties → random trading, -6% loss
- unrealized_pnl_weight=0.0 → Zero-Trade Collapse (do nothing is optimal)

The key insight: when transaction_cost > 0 and holding reward = 0,
the optimal policy is to never trade (cost-free).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RewardConfig:
    """Reward function configuration.

    Different configurations lead to different pathological behaviors:

    Buy & Hold:
        unrealized_pnl_weight=1.0, transaction_cost=0.0001
        → Agent learned to hold positions for unrealized gains

    Penalty Hell:
        holding_penalty=0.0005, inactivity_penalty=0.0001
        → Over-penalization led to indiscriminate trading

    Zero-Trade Collapse:
        unrealized_pnl_weight=0.0, transaction_cost=0.0003
        → "Do nothing" became optimal (zero cost)

    Trade Completion Bonus:
        trade_completion_bonus=0.001, profitable_trade_bonus=0.002
        → Encouraged trading but didn't fix signal quality
    """

    unrealized_pnl_weight: float = 1.0
    transaction_cost: float = 0.0001
    reward_scaling: float = 100.0
    holding_penalty: float = 0.0
    inactivity_penalty: float = 0.0
    trade_completion_bonus: float = 0.0
    profitable_trade_bonus: float = 0.0


@dataclass
class TradeRecord:
    """Record of a completed trade."""

    entry_price: float
    exit_price: float
    position: int  # 1=long, -1=short
    pnl: float
    step: int


@dataclass
class EnvironmentState:
    """Internal state of the trading environment."""

    step: int = 0
    position: int = 0  # -1=short, 0=neutral, 1=long
    entry_price: float | None = None
    total_pnl: float = 0.0
    trade_count: int = 0
    winning_trades: int = 0
    trade_history: list[TradeRecord] = field(default_factory=list)


class SimpleTradingEnvironment:
    """Simplified trading environment for demonstrating reward design issues.

    This is a stripped-down version of the full Gymnasium environment,
    focusing on the reward calculation logic that caused the pathological
    behaviors documented in the early experiments.

    Actions:
        0: HOLD - do nothing
        1: BUY  - open/flip to long
        2: SELL - open/flip to short
    """

    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        prices: np.ndarray,
        config: RewardConfig | None = None,
    ) -> None:
        """Initialize environment.

        Args:
            prices: Array of close prices.
            config: Reward configuration.
        """
        self.prices = prices
        self.config = config or RewardConfig()
        self.state = EnvironmentState()

    def reset(self) -> None:
        """Reset environment to initial state."""
        self.state = EnvironmentState()

    def step(self, action: int) -> tuple[float, bool, dict[str, Any]]:
        """Execute one step.

        Args:
            action: Trading action (0=hold, 1=buy, 2=sell).

        Returns:
            Tuple of (reward, done, info).
        """
        i = self.state.step
        if i >= len(self.prices) - 1:
            return 0.0, True, self._get_info()

        current_price = self.prices[i]
        next_price = self.prices[i + 1]
        reward = self._calculate_reward(action, current_price, next_price)

        self.state.step += 1
        done = self.state.step >= len(self.prices) - 1

        return reward, done, self._get_info()

    def _calculate_reward(
        self, action: int, current_price: float, next_price: float
    ) -> float:
        """Calculate reward - the core of the problem.

        This function shows exactly how different reward designs
        lead to different pathological behaviors.
        """
        cfg = self.config
        reward = 0.0
        price_change = (next_price - current_price) / current_price

        # --- Execute action ---
        if action == self.BUY:
            if self.state.position == 0:
                # Open long
                self.state.position = 1
                self.state.entry_price = current_price
                reward -= cfg.transaction_cost
            elif self.state.position == -1:
                # Close short + open long
                realized = self._close_position(current_price)
                reward += realized - cfg.transaction_cost
                reward += cfg.trade_completion_bonus
                if realized > 0:
                    reward += cfg.profitable_trade_bonus
                self.state.position = 1
                self.state.entry_price = current_price
                reward -= cfg.transaction_cost

        elif action == self.SELL:
            if self.state.position == 0:
                # Open short
                self.state.position = -1
                self.state.entry_price = current_price
                reward -= cfg.transaction_cost
            elif self.state.position == 1:
                # Close long + open short
                realized = self._close_position(current_price)
                reward += realized - cfg.transaction_cost
                reward += cfg.trade_completion_bonus
                if realized > 0:
                    reward += cfg.profitable_trade_bonus
                self.state.position = -1
                self.state.entry_price = current_price
                reward -= cfg.transaction_cost

        else:
            # HOLD action
            if self.state.position == 0:
                reward -= cfg.inactivity_penalty

        # --- Holding reward/penalty ---
        if self.state.position != 0:
            # This is what made Buy & Hold optimal
            # When weight=1.0, holding a profitable position gives
            # continuous reward without any cost.
            unrealized_pnl = price_change * self.state.position
            reward += unrealized_pnl * cfg.unrealized_pnl_weight

            # Adding penalties didn't fix the problem
            reward -= cfg.holding_penalty

        return reward * cfg.reward_scaling

    def _close_position(self, current_price: float) -> float:
        """Close current position and return realized PnL."""
        if self.state.position == 0 or self.state.entry_price is None:
            return 0.0

        pnl = (
            (current_price - self.state.entry_price)
            / self.state.entry_price
            * self.state.position
        )
        self.state.total_pnl += pnl
        self.state.trade_count += 1
        if pnl > 0:
            self.state.winning_trades += 1

        self.state.trade_history.append(
            TradeRecord(
                entry_price=self.state.entry_price,
                exit_price=current_price,
                position=self.state.position,
                pnl=pnl,
                step=self.state.step,
            )
        )

        self.state.entry_price = None
        return pnl

    def _get_info(self) -> dict[str, Any]:
        """Get current info."""
        return {
            "step": self.state.step,
            "position": self.state.position,
            "total_pnl": self.state.total_pnl,
            "trade_count": self.state.trade_count,
            "win_rate": (
                self.state.winning_trades / self.state.trade_count
                if self.state.trade_count > 0
                else 0.0
            ),
        }


def simulate_agent_policy(
    env: SimpleTradingEnvironment,
    policy: str = "always_buy",
) -> dict[str, Any]:
    """Simulate a fixed policy to demonstrate pathological behaviors.

    Args:
        env: Trading environment.
        policy: One of 'always_buy', 'always_hold', 'random'.

    Returns:
        Final info dict.
    """
    env.reset()
    rng = np.random.default_rng(42)
    total_reward = 0.0

    while True:
        if policy == "always_buy":
            action = SimpleTradingEnvironment.BUY
        elif policy == "always_hold":
            action = SimpleTradingEnvironment.HOLD
        elif policy == "random":
            action = int(rng.integers(0, 3))
        else:
            action = SimpleTradingEnvironment.HOLD

        reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    info["total_reward"] = total_reward
    return info
