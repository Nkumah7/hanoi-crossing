"""Game rules for Hanoi Crossing.

Every function here is pure: no I/O, no mutation, no global state. `step`
returns a new `GameState` rather than modifying the one it was given, so a
state can be snapshotted, replayed, or shared across concurrent games without
defensive copying.
"""

from __future__ import annotations

from dataclasses import replace

from hanoi_crossing.model import (
    HOME_POLE,
    TARGET_POLE,
    VISIBLE_POLES,
    Action,
    ActionKind,
    GameState,
    Observation,
    Outcome,
    Player,
    Pole,
)


def initial_state(n: int) -> GameState:
    """Return the opening position for a game with `n` disks per player.

    Player A holds the odd sizes (1, 3, 5, ...) and player B the even sizes
    (2, 4, 6, ...). Each stack is ordered bottom-to-top, so the largest disk
    is first and the smallest is on top.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    odd = tuple(2 * i - 1 for i in range(1, n + 1))  # 1, 3, 5, ...
    even = tuple(2 * i for i in range(1, n + 1))  # 2, 4, 6, ...

    return GameState(
        poles={
            Pole.P1A: tuple(reversed(odd)),  # largest at bottom
            Pole.P1B: tuple(reversed(even)),
            Pole.SHARED: (),
            Pole.P3A: (),
            Pole.P3B: (),
        },
        hands={Player.A: None, Player.B: None},
        starting_disks={Player.A: frozenset(odd), Player.B: frozenset(even)},
        turn=0,
    )


def observe(state: GameState, player: Player) -> Observation:
    """Return what `player` can see.

    Only the three poles visible to that player are included, along with their
    own hand. The opponent's poles and hand are absent from the returned type
    entirely, so an agent cannot read them even by accident.
    """
    return Observation(
        player=player,
        poles={pole: state.poles[pole] for pole in VISIBLE_POLES[player]},
        hand=state.hands[player],
        turn=state.turn,
    )


def can_place(disk: int, top: int | None) -> bool:
    """Standard Hanoi placement rule: empty pole, or strictly larger disk."""
    return top is None or disk < top


def legal_actions(observation: Observation) -> tuple[Action, ...]:
    """Return every action available to the observing player.

    Depends only on the observation, never on the full state — a player's
    options are determined by their visible poles and their own hand, which is
    exactly what an `Observation` contains.

    Skip is always legal. Lift and place are mutually exclusive: a player may
    hold at most one disk, so an empty hand can only lift and a full hand can
    only place.
    """
    actions: list[Action] = [Action.skip()]
    visible = VISIBLE_POLES[observation.player]

    if observation.hand is None:
        actions.extend(
            Action.lift(pole) for pole in visible if observation.poles[pole]
        )
    else:
        actions.extend(
            Action.place(pole)
            for pole in visible
            if can_place(observation.hand, observation.top(pole))
        )

    return tuple(actions)


def is_legal(state: GameState, player: Player, action: Action) -> bool:
    """Whether `action` is available to `player` in `state`."""
    return action in legal_actions(observe(state, player))


def step(state: GameState, player: Player, action: Action) -> GameState:
    """Apply `action` and return the resulting state.

    An illegal action leaves the position unchanged but still consumes the
    turn, as the rules require. The turn counter always advances, so a game
    cannot stall on repeated illegal input.
    """
    if not is_legal(state, player, action):
        return replace(state, turn=state.turn + 1)

    if action.kind is ActionKind.SKIP:
        return replace(state, turn=state.turn + 1)

    assert action.pole is not None  # guaranteed by is_legal
    poles = dict(state.poles)
    hands = dict(state.hands)

    if action.kind is ActionKind.LIFT:
        stack = poles[action.pole]
        hands[player] = stack[-1]
        poles[action.pole] = stack[:-1]
    else:  # PLACE
        disk = hands[player]
        assert disk is not None  # guaranteed by is_legal
        poles[action.pole] = poles[action.pole] + (disk,)
        hands[player] = None

    return GameState(
        poles=poles,
        hands=hands,
        starting_disks=state.starting_disks,
        turn=state.turn + 1,
    )


def has_won(state: GameState, player: Player, require_all_disks: bool = False) -> bool:
    """Whether `player` has met the win condition.

    The condition is that their hand is empty and, among their *visible*
    poles, only pole 3 holds disks. Since the shared pole is visible to both
    players, a disk left there blocks both of them (see README, Decision 1).

    With `require_all_disks`, the player must additionally have every disk
    they started with on their target pole. The spec does not require this;
    see README, Decision 4.
    """
    if state.hands[player] is not None:
        return False
    if state.poles[HOME_POLE[player]]:
        return False
    if state.poles[Pole.SHARED]:
        return False

    target = state.poles[TARGET_POLE[player]]
    if not target:
        return False

    if require_all_disks:
        return frozenset(target) == state.starting_disks[player]
    return True


def check_outcome(state: GameState, require_all_disks: bool = False) -> Outcome:
    """Classify the position.

    Evaluated for both players regardless of who moved last, because a win can
    be completed by the opponent's action — lifting the last disk off the
    shared pole can satisfy the other player's condition (README, Decision 2).
    """
    a_won = has_won(state, Player.A, require_all_disks)
    b_won = has_won(state, Player.B, require_all_disks)

    if a_won and b_won:
        # Believed unreachable: the action that clears the shared pole leaves
        # the actor holding a disk. Guarded rather than left undefined.
        return Outcome.DRAW
    if a_won:
        return Outcome.A_WINS
    if b_won:
        return Outcome.B_WINS
    return Outcome.IN_PROGRESS