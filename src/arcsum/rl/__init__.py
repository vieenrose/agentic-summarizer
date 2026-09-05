"""Reinforcement learning against the harness's own deterministic gates (SPEC §5.2.2).

Separate from `evalkit` on purpose: `evalkit` MEASURES a finished checkpoint and must stay
free of anything that could be tuned against, while this package OPTIMISES. They share the
metric implementations so the training signal and the gate cannot drift apart, but the
dependency runs one way — `rl` imports `evalkit`, never the reverse.
"""

from arcsum.rl.reward import RewardBreakdown, score

__all__ = ["RewardBreakdown", "score"]
