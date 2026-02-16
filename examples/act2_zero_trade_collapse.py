"""Act 2: The Reward Design Quagmire and Zero-Trade Collapse.

After discovering the Buy & Hold trap in Act 1, we tried to fix
the DQN by redesigning the reward function. This led us through
a painful series of failures:

Buy & Hold (unrealized_pnl_weight=1.0)
    → Agent holds forever for unrealized gains

Penalty Hell (holding_penalty + inactivity_penalty)
    → Over-penalization led to random trading, -6% loss

Balanced penalties
    → Still random, marginal improvement

Zero-Trade Collapse (unrealized_pnl_weight=0.0)
    → "Do nothing" becomes the optimal policy!
    → When there's a transaction cost but no reward for holding,
       the rational strategy is to never trade.

Trade completion bonus
    → Agent trades just to collect the bonus,
       not because it found good entries.

The key insight: reward engineering is a dead end.
No matter how we tuned the reward, the DQN was trying to
simultaneously learn WHAT to trade and HOW to manage positions.
These are fundamentally different problems.

→ This led to Act 3: separating architecture.

Usage::

    python examples/act2_zero_trade_collapse.py

"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.environment import (
    RewardConfig,
    SimpleTradingEnvironment,
    simulate_agent_policy,
)

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def generate_price_data(
    n_steps: int = 2000,
    seed: int = 42,
    trend: float = 0.00001,
) -> np.ndarray:
    """Generate synthetic price series.

    Args:
        n_steps: Number of price points.
        seed: Random seed.
        trend: Per-step drift.

    Returns:
        Array of close prices.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, 0.0003, n_steps)
    return 150.0 * np.exp(np.cumsum(returns))


def demonstrate_reward_configs() -> None:
    """Show how different reward configs lead to different failures."""
    prices = generate_price_data()
    buy_hold_return = prices[-1] / prices[0] - 1

    logger.info("=" * 70)
    logger.info("Act 2: Reward Design Quagmire & Zero-Trade Collapse")
    logger.info("=" * 70)
    logger.info("")
    logger.info(
        f"Price data: {prices[0]:.2f} -> {prices[-1]:.2f} "
        f"(Buy & Hold return: {buy_hold_return:+.2%})"
    )
    logger.info("")

    # Define reward configs that mirror our historical experiments
    configs = [
        (
            "Buy & Hold (unrealized_pnl_weight=1.0)",
            RewardConfig(
                unrealized_pnl_weight=1.0,
                transaction_cost=0.0001,
            ),
            "always_buy",
            "Agent holds forever for unrealized gains",
        ),
        (
            "Penalty Hell",
            RewardConfig(
                unrealized_pnl_weight=1.0,
                transaction_cost=0.0001,
                holding_penalty=0.0005,
                inactivity_penalty=0.0001,
            ),
            "random",
            "Over-penalized → trades randomly",
        ),
        (
            "Zero-Trade Collapse (unrealized_pnl=0)",
            RewardConfig(
                unrealized_pnl_weight=0.0,
                transaction_cost=0.0003,
            ),
            "always_hold",
            "Do nothing = optimal (no cost!)",
        ),
        (
            "Trade Completion Bonus",
            RewardConfig(
                unrealized_pnl_weight=0.0,
                transaction_cost=0.0003,
                trade_completion_bonus=0.001,
                profitable_trade_bonus=0.002,
            ),
            "random",
            "Trades for bonus, not alpha",
        ),
    ]

    # Run each configuration
    logger.info("-" * 70)
    logger.info(
        f"{'Version':<45} {'Trades':>7} {'WinRate':>8} {'PnL':>8} {'Reward':>10}"
    )
    logger.info("-" * 70)

    for name, config, policy, _description in configs:
        env = SimpleTradingEnvironment(prices, config)
        result = simulate_agent_policy(env, policy=policy)

        win_rate = result["win_rate"]
        pnl = result["total_pnl"]
        trades = result["trade_count"]
        reward = result["total_reward"]

        logger.info(
            f"{name:<45} {trades:>7} {win_rate:>7.1%} {pnl:>+7.4f} {reward:>+10.2f}"
        )

    logger.info("-" * 70)
    logger.info("")

    # Explain the Zero-Trade Collapse
    logger.info("=" * 70)
    logger.info("Zero-Trade Collapse Explained")
    logger.info("=" * 70)
    logger.info("")
    logger.info("When unrealized_pnl_weight = 0 and transaction_cost > 0:")
    logger.info("")
    logger.info("  Expected reward of TRADING:")
    logger.info("    = E[realized_pnl] - transaction_cost")
    logger.info("    = 0 - 0.0003  (if model has no edge)")
    logger.info("    = -0.0003 per trade")
    logger.info("")
    logger.info("  Expected reward of DOING NOTHING:")
    logger.info("    = 0  (no cost, no reward)")
    logger.info("")
    logger.info("  Since -0.0003 < 0, the optimal policy is: NEVER TRADE.")
    logger.info("")
    logger.info("This is mathematically optimal! The DQN correctly learned")
    logger.info("that trading has negative expected value when it has no")
    logger.info("predictive edge. The problem isn't the model - it's that")
    logger.info("we're asking one system to do two things:")
    logger.info("")
    logger.info("  1. PREDICT market direction (signal generation)")
    logger.info("  2. MANAGE positions (entry/exit/risk)")
    logger.info("")
    logger.info("→ Solution: Separate these into independent components.")
    logger.info("  This is what we do in Act 3.")


def demonstrate_reward_landscape() -> None:
    """Show the reward landscape that causes Zero-Trade Collapse."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Reward Landscape Visualization")
    logger.info("=" * 70)
    logger.info("")
    logger.info("For a single trade with transaction_cost=0.0003:")
    logger.info("")
    logger.info("  Price Change (pips) | Realized PnL | Net Reward")
    logger.info("  " + "-" * 50)

    transaction_cost = 0.0003
    for pips in [-20, -10, -5, 0, 5, 10, 20]:
        price_change = pips * 0.01 / 150  # Convert pips to return
        realized = price_change
        net = realized - transaction_cost

        # Format with color-like indicators
        indicator = "+" if net > 0 else "-"
        if net == 0:
            indicator = "="

        logger.info(f"  {pips:>+18} | {realized:>+11.6f} | {net:>+10.6f} [{indicator}]")

    logger.info("")
    logger.info("  The transaction cost creates a 'dead zone' around zero")
    logger.info("  where trading always loses money. Without predictive")
    logger.info("  power, the DQN correctly avoids this zone entirely.")


if __name__ == "__main__":
    setup_logging()
    demonstrate_reward_configs()
    demonstrate_reward_landscape()
