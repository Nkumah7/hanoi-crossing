"""Tests for the frontends.

These cover input parsing and the loop that drives the engine. The rules
themselves are tested in `test_engine.py`.
"""

import json

import pytest

from hanoi_crossing.cli import parse_action, run_random, run_replay
from hanoi_crossing.model import Action, Outcome, Pole


def test_parse_action_string_form():
    assert parse_action("lift 1a") == Action.lift(Pole.P1A)
    assert parse_action("place 3b") == Action.place(Pole.P3B)
    assert parse_action("skip") == Action.skip()


def test_parse_action_object_form():
    assert parse_action({"kind": "lift", "pole": "2"}) == Action.lift(Pole.SHARED)
    assert parse_action({"kind": "skip"}) == Action.skip()


def test_parse_action_rejects_unknown_kind():
    with pytest.raises(ValueError):
        parse_action("jump 1a")


def test_replay_reproduces_the_spec_example(tmp_path):
    game = tmp_path / "game.json"
    game.write_text(
        json.dumps(
            {
                "n": 1,
                "turn_order": ["A", "B", "A"],
                "moves": ["lift 1a", "lift 1b", "place 3a"],
            }
        )
    )
    _, outcome = run_replay(game)
    assert outcome is Outcome.A_WINS


def test_replay_stops_once_the_game_is_decided(tmp_path):
    """Trailing moves must not overwrite a result already reached."""
    game = tmp_path / "game.json"
    game.write_text(
        json.dumps(
            {
                "n": 1,
                "turn_order": ["A", "B", "A", "B"],
                "moves": ["lift 1a", "lift 1b", "place 3a", "place 2"],
            }
        )
    )
    state, outcome = run_replay(game)
    assert outcome is Outcome.A_WINS
    assert state.poles[Pole.SHARED] == ()


def test_replay_rejects_mismatched_lengths(tmp_path):
    game = tmp_path / "game.json"
    game.write_text(
        json.dumps({"n": 1, "turn_order": ["A", "B"], "moves": ["lift 1a"]})
    )
    with pytest.raises(ValueError):
        run_replay(game)


def test_random_play_terminates_within_the_turn_limit():
    _, outcome = run_random(n=2, seed=1, max_turns=200)
    assert outcome in {
        Outcome.A_WINS,
        Outcome.B_WINS,
        Outcome.DRAW,
        Outcome.IN_PROGRESS,
    }


def test_random_play_respects_max_turns():
    state, _ = run_random(n=6, seed=1, max_turns=10)
    assert state.turn <= 10


def test_random_play_is_reproducible():
    first, outcome_a = run_random(n=3, seed=7, max_turns=100)
    second, outcome_b = run_random(n=3, seed=7, max_turns=100)
    assert first == second
    assert outcome_a is outcome_b