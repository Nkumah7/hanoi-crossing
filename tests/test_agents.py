"""Tests for agents.

The important property is not which action an agent picks, but that it only
ever picks legal ones and only ever sees its own observation.
"""

import random

from hanoi_crossing.agents import RandomAgent, SkipAgent
from hanoi_crossing.engine import initial_state, legal_actions, observe, step
from hanoi_crossing.model import Action, ActionKind, Player


def test_random_agent_only_returns_legal_actions():
    agent = RandomAgent(random.Random(0))
    state = initial_state(n=3)

    for turn, player in enumerate(
        [Player.A, Player.B] * 25
    ):  # alternating is arbitrary; the engine imposes no pattern
        obs = observe(state, player)
        action = agent(obs)
        assert action in legal_actions(obs), f"illegal action on turn {turn}"
        state = step(state, player, action)


def test_same_seed_produces_the_same_game():
    def play(seed: int) -> list[str]:
        agent = RandomAgent(random.Random(seed))
        state = initial_state(n=2)
        actions = []
        for player in [Player.A, Player.B] * 20:
            action = agent(observe(state, player))
            actions.append(str(action))
            state = step(state, player, action)
        return actions

    assert play(42) == play(42)


def test_different_seeds_diverge():
    def play(seed: int) -> list[str]:
        agent = RandomAgent(random.Random(seed))
        state = initial_state(n=2)
        actions = []
        for player in [Player.A, Player.B] * 20:
            action = agent(observe(state, player))
            actions.append(str(action))
            state = step(state, player, action)
        return actions

    assert play(1) != play(2)


def test_skip_agent_always_skips():
    agent = SkipAgent()
    state = initial_state(n=2)
    assert agent(observe(state, Player.A)).kind is ActionKind.SKIP


def test_agent_receives_only_its_own_observation():
    """An agent cannot reach the opponent's poles: they are not in the type."""
    captured = {}

    def spy(observation):
        captured["poles"] = set(observation.poles)
        return Action.skip()

    state = initial_state(n=2)
    spy(observe(state, Player.A))

    from hanoi_crossing.model import Pole

    assert Pole.P1B not in captured["poles"]
    assert Pole.P3B not in captured["poles"]