"""Agents that play Hanoi Crossing.

An agent is any callable taking an `Observation` and returning an `Action`. It
never sees a `GameState`, so it cannot read the opponent's poles or hand —
the same interface an external reinforcement-learning policy would use.
"""

from __future__ import annotations

import random
from typing import Protocol

from hanoi_crossing.engine import legal_actions
from hanoi_crossing.model import Action, Observation


class Agent(Protocol):
    """The interface a policy must satisfy to play a game."""

    def __call__(self, observation: Observation) -> Action: ...


class RandomAgent:
    """Picks uniformly at random from the legal actions.

    The random source is injected rather than taken from the `random` module's
    global state, so a game is reproducible from its seed and concurrent games
    do not contend on a shared generator.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()

    def __call__(self, observation: Observation) -> Action:
        actions = legal_actions(observation)
        return self._rng.choice(actions)


class SkipAgent:
    """An agent that always skips.

    A second implementation of the `Agent` protocol, confirming the interface
    is not shaped around `RandomAgent` alone.
    """

    def __call__(self, _observation: Observation) -> Action:
        return Action.skip()
