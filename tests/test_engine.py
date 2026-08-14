"""Tests for the game engine.

Tests exercise the engine directly, without going through either frontend.
Because state is immutable, each test constructs exactly the position it needs
— there is no setup, teardown, or ordering dependency between tests.
"""

from hanoi_crossing.engine import (
    check_outcome,
    initial_state,
    legal_actions,
    observe,
    step,
)
from hanoi_crossing.model import (
    Action,
    ActionKind,
    GameState,
    Outcome,
    Player,
    Pole,
)


def _state(
    p1a: tuple[int, ...] = (),
    shared: tuple[int, ...] = (),
    p3a: tuple[int, ...] = (),
    p1b: tuple[int, ...] = (),
    p3b: tuple[int, ...] = (),
    hand_a: int | None = None,
    hand_b: int | None = None,
    turn: int = 0,
) -> GameState:
    """Build an arbitrary position for testing.

    Immutability is what makes this possible: any legal-looking arrangement is
    a valid state, so tests can jump straight to the position of interest
    instead of playing towards it.
    """
    return GameState(
        poles={
            Pole.P1A: p1a,
            Pole.SHARED: shared,
            Pole.P3A: p3a,
            Pole.P1B: p1b,
            Pole.P3B: p3b,
        },
        hands={Player.A: hand_a, Player.B: hand_b},
        starting_disks={Player.A: frozenset({1, 3}), Player.B: frozenset({2, 4})},
        turn=turn,
    )


# --- the worked example from the specification -----------------------------


def test_spec_example_player_a_wins():
    """N=1, turn order [A, B, A], as given in the brief."""
    state = initial_state(n=1)
    state = step(state, Player.A, Action.lift(Pole.P1A))
    state = step(state, Player.B, Action.lift(Pole.P1B))
    state = step(state, Player.A, Action.place(Pole.P3A))
    assert check_outcome(state) is Outcome.A_WINS


# --- initial state ---------------------------------------------------------


def test_initial_state_stacks_largest_at_bottom():
    state = initial_state(n=2)
    assert state.poles[Pole.P1A] == (3, 1)
    assert state.poles[Pole.P1B] == (4, 2)


def test_initial_state_other_poles_empty_and_hands_free():
    state = initial_state(n=2)
    assert state.poles[Pole.SHARED] == ()
    assert state.poles[Pole.P3A] == ()
    assert state.poles[Pole.P3B] == ()
    assert state.hands[Player.A] is None
    assert state.hands[Player.B] is None
    assert state.turn == 0


def test_initial_state_rejects_zero_disks():
    import pytest

    with pytest.raises(ValueError):
        initial_state(n=0)


# --- partial observability -------------------------------------------------


def test_observation_excludes_opponent_poles():
    state = initial_state(n=2)
    seen = observe(state, Player.A).poles
    assert set(seen) == {Pole.P1A, Pole.SHARED, Pole.P3A}
    assert Pole.P1B not in seen
    assert Pole.P3B not in seen


def test_observation_carries_only_own_hand():
    state = _state(hand_a=1, hand_b=2)
    assert observe(state, Player.A).hand == 1
    assert observe(state, Player.B).hand == 2


# --- legal actions ---------------------------------------------------------


def test_empty_hand_can_only_lift_or_skip():
    state = initial_state(n=2)
    kinds = {a.kind for a in legal_actions(observe(state, Player.A))}
    assert kinds == {ActionKind.LIFT, ActionKind.SKIP}


def test_full_hand_can_only_place_or_skip():
    state = _state(p1a=(3,), hand_a=1)
    kinds = {a.kind for a in legal_actions(observe(state, Player.A))}
    assert kinds == {ActionKind.PLACE, ActionKind.SKIP}


def test_skip_is_always_legal():
    for state in (initial_state(n=2), _state(hand_a=1)):
        assert Action.skip() in legal_actions(observe(state, Player.A))


def test_cannot_place_larger_disk_on_smaller():
    # A holds disk 3; pole 3a has disk 1 on top, so the placement is illegal.
    state = _state(p3a=(1,), hand_a=3)
    actions = legal_actions(observe(state, Player.A))
    assert Action.place(Pole.P3A) not in actions


def test_can_place_smaller_disk_on_larger():
    state = _state(p3a=(3,), hand_a=1)
    assert Action.place(Pole.P3A) in legal_actions(observe(state, Player.A))


def test_can_place_on_empty_pole():
    state = _state(hand_a=5)
    assert Action.place(Pole.SHARED) in legal_actions(observe(state, Player.A))


def test_lift_ignores_disk_size():
    """Lifting takes whatever is on top; only placing checks size."""
    state = _state(p1a=(5, 3, 1))
    assert Action.lift(Pole.P1A) in legal_actions(observe(state, Player.A))


def test_cannot_lift_from_empty_pole():
    state = _state(p1a=(1,))
    assert Action.lift(Pole.P3A) not in legal_actions(observe(state, Player.A))


# --- applying actions ------------------------------------------------------


def test_lift_moves_top_disk_to_hand():
    state = step(_state(p1a=(3, 1)), Player.A, Action.lift(Pole.P1A))
    assert state.poles[Pole.P1A] == (3,)
    assert state.hands[Player.A] == 1


def test_place_empties_hand_onto_pole():
    state = step(_state(hand_a=1), Player.A, Action.place(Pole.P3A))
    assert state.poles[Pole.P3A] == (1,)
    assert state.hands[Player.A] is None


def test_illegal_action_wastes_the_turn():
    before = _state(p3a=(1,), hand_a=3)
    after = step(before, Player.A, Action.place(Pole.P3A))
    assert after.poles == before.poles
    assert after.hands == before.hands
    assert after.turn == before.turn + 1


def test_skip_advances_turn_only():
    before = initial_state(n=2)
    after = step(before, Player.A, Action.skip())
    assert after.poles == before.poles
    assert after.turn == before.turn + 1


def test_step_does_not_mutate_the_input_state():
    before = initial_state(n=2)
    snapshot = dict(before.poles)
    step(before, Player.A, Action.lift(Pole.P1A))
    assert before.poles == snapshot
    assert before.hands[Player.A] is None


# --- win conditions --------------------------------------------------------


def test_player_a_wins_when_only_target_pole_holds_disks():
    state = _state(p3a=(3, 1))
    assert check_outcome(state) is Outcome.A_WINS


def test_disk_on_shared_pole_blocks_the_win():
    """Decision 1: the shared pole is visible to both, so it blocks both."""
    blocked = _state(p3a=(3, 1), shared=(5,))
    assert check_outcome(blocked) is Outcome.IN_PROGRESS

    cleared = _state(p3a=(3, 1))
    assert check_outcome(cleared) is Outcome.A_WINS


def test_win_can_be_completed_by_the_opponent():
    """Decision 2: B lifting the blocking disk hands A the win."""
    blocked = _state(p3a=(3, 1), shared=(5,), p1b=(4, 2))
    assert check_outcome(blocked) is Outcome.IN_PROGRESS

    after = step(blocked, Player.B, Action.lift(Pole.SHARED))
    assert check_outcome(after) is Outcome.A_WINS


def test_holding_a_disk_prevents_winning():
    state = _state(p3a=(3,), hand_a=1)
    assert check_outcome(state) is Outcome.IN_PROGRESS


def test_disks_left_on_home_pole_prevent_winning():
    state = _state(p1a=(3,), p3a=(1,))
    assert check_outcome(state) is Outcome.IN_PROGRESS


def test_empty_target_pole_is_not_a_win():
    assert check_outcome(_state()) is Outcome.IN_PROGRESS


def test_opening_position_is_in_progress():
    assert check_outcome(initial_state(n=3)) is Outcome.IN_PROGRESS


# --- the optional stricter reading (Decision 4) ----------------------------


def test_incomplete_tower_wins_under_the_literal_reading():
    """A has lost disk 3 to B, but still satisfies the stated condition."""
    state = _state(p3a=(1,), p3b=(3,), p1b=(4, 2))
    assert check_outcome(state) is Outcome.A_WINS


def test_incomplete_tower_does_not_win_when_all_disks_required():
    state = _state(p3a=(1,), p3b=(4, 3, 2))
    assert check_outcome(state, require_all_disks=True) is Outcome.IN_PROGRESS


def test_complete_tower_wins_when_all_disks_required():
    state = _state(p3a=(3, 1))
    assert check_outcome(state, require_all_disks=True) is Outcome.A_WINS


# --- hostage strategy (Decision 5) -----------------------------------------


def test_hostage_strategy_prevents_both_players_winning():
    state = _state(p1a=(3,), shared=(1,), p1b=(4, 2))
    state = step(state, Player.B, Action.lift(Pole.SHARED))
    for _ in range(50):
        state = step(state, Player.B, Action.skip())
    assert check_outcome(state) is Outcome.IN_PROGRESS


def test_hostage_is_decisive_when_all_disks_required():
    state = _state(p3a=(3,), shared=(1,), p1b=(4, 2))
    state = step(state, Player.B, Action.lift(Pole.SHARED))
    assert check_outcome(state, require_all_disks=True) is Outcome.IN_PROGRESS