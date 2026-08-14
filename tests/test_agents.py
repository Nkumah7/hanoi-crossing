"""Tests for agents.

The important property is not which action an agent picks, but that it only
ever picks legal ones and only ever sees its own observation.
"""

import random

from hanoi_crossing.agents import RandomAgent, SkipAgent
from hanoi_crossing.cli import run_random
from hanoi_crossing.engine import initial_state, legal_actions, observe, step
from hanoi_crossing.model import Action, ActionKind, Outcome, Player, Pole


def _play(seed: int, turns: int = 40) -> list[str]:
    """Play a fixed number of turns and return the action sequence."""
    agent = RandomAgent(random.Random(seed))
    state = initial_state(n=2)
    actions = []
    for i in range(turns):
        player = Player.A if i % 2 == 0 else Player.B
        action = agent(observe(state, player))
        actions.append(str(action))
        state = step(state, player, action)
    return actions


def test_random_agent_only_returns_legal_actions():
    agent = RandomAgent(random.Random(0))
    state = initial_state(n=3)

    # Alternating is arbitrary; the engine imposes no turn-order pattern.
    for turn, player in enumerate([Player.A, Player.B] * 25):
        obs = observe(state, player)
        action = agent(obs)
        assert action in legal_actions(obs), f"illegal action on turn {turn}"
        state = step(state, player, action)


def test_same_seed_produces_the_same_game():
    assert _play(42) == _play(42)


def test_different_seeds_diverge():
    assert _play(1) != _play(2)


def test_skip_agent_always_skips():
    agent = SkipAgent()
    state = initial_state(n=2)
    assert agent(observe(state, Player.A)).kind is ActionKind.SKIP


def test_a_plain_function_satisfies_the_agent_protocol():
    """No inheritance required — any callable of the right shape works.

    This is what "the engine must serve an external agent unchanged" means in
    practice: a policy written elsewhere plugs in without subclassing
    anything from this package.
    """

    def always_skip(observation) -> Action:
        return Action.skip()

    state, outcome = run_random(n=2, max_turns=20, agent=always_skip)
    assert state.turn == 20
    assert outcome is Outcome.IN_PROGRESS


def test_skip_agent_can_drive_a_full_game():
    state, outcome = run_random(n=2, max_turns=20, agent=SkipAgent())
    assert state.turn == 20
    assert outcome is Outcome.IN_PROGRESS


def test_agent_sees_only_its_own_poles():
    """The opponent's poles are absent from the observation type entirely."""
    seen: dict[str, set[Pole]] = {}

    class Spy:
        def __call__(self, observation):
            seen["poles"] = set(observation.poles)
            return SkipAgent()(observation)

    run_random(n=2, max_turns=1, agent=Spy())

    assert Pole.P1B not in seen["poles"]
    assert Pole.P3B not in seen["poles"]
